# SPDX-License-Identifier: Apache-2.0
import os
from collections import OrderedDict
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.distributed.spmd as xs
import torchax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)

logger = init_logger(__name__)

VLLM_TORCHAX_ENABLED = os.environ.get('VLLM_TORCHAX_ENABLED', '0') == '1'


def create_torchax_tensor_with_partition_spec(
        weight_t: torch.Tensor,
        mesh: Optional[Union["xs.Mesh", Mesh]] = None,
        partition_spec: Optional[Tuple] = None) -> torch.Tensor:
    # Validate that if mesh is None, sharding must also be None
    if mesh is None and partition_spec is not None:
        raise ValueError("if mesh is None, sharding must also be None")

    if VLLM_TORCHAX_ENABLED:
        if mesh is None:
            # Single chip case.
            return weight_t.to('jax')

        if weight_t.dtype == torch.bfloat16:
            jax_t = jnp.array(weight_t.to(torch.float32).numpy()).astype(
                jnp.bfloat16)
        else:
            jax_t = jnp.array(weight_t.numpy())

        p = P()
        if partition_spec is not None:
            p = P(*partition_spec)
        sharding = NamedSharding(mesh, p)

        jax_t = jax.device_put(jax_t, sharding)

        torchax_t = torchax.tensor.Tensor(jax_t, torchax.default_env())
        return torchax_t
    else:
        if mesh is None:
            return weight_t.to('xla')

        # If mesh is provided, we use the XLA sharding
        xs.mark_sharding(weight_t, mesh, sharding)
        return weight_t


class XlaQKVParallelLinear(nn.Module):

    def __init__(self,
                 qkv_linear: nn.Module,
                 mesh: Optional["xs.Mesh"] = None):
        super().__init__()
        assert isinstance(qkv_linear, QKVParallelLinear)
        self.skip_bias_add = qkv_linear.skip_bias_add
        self.return_bias = qkv_linear.return_bias
        assert qkv_linear.tp_size == 1, "TP > 1 is only supported under SPMD."

        self.q_weight: Parameter
        self.k_weight: Parameter
        self.v_weight: Parameter
        self.q_bias: Optional[Parameter]
        self.k_bias: Optional[Parameter]
        self.v_bias: Optional[Parameter]
        self._load_weights_from_qkv_linear(qkv_linear)
        if mesh is not None:
            self._shard_weight(mesh)

    def _shard_weight(self, mesh: "xs.Mesh"):
        self.q_weight = Parameter(create_torchax_tensor_with_partition_spec(
            self.q_weight, mesh, ('x', None)),
                                  requires_grad=False)
        self.k_weight = Parameter(create_torchax_tensor_with_partition_spec(
            self.k_weight, mesh, ('x', None)),
                                  requires_grad=False)
        self.v_weight = Parameter(create_torchax_tensor_with_partition_spec(
            self.v_weight, mesh, ('x', None)),
                                  requires_grad=False)

        if self.q_bias is not None:
            assert self.k_bias is not None and self.v_bias is not None, \
                "QKVParallelLinear should have q, k, and v biases together."
            self.q_bias = Parameter(create_torchax_tensor_with_partition_spec(
                self.q_bias, mesh, ('x', )),
                                    requires_grad=False)
            self.k_bias = Parameter(create_torchax_tensor_with_partition_spec(
                self.k_bias, mesh, ('x', )),
                                    requires_grad=False)
            self.v_bias = Parameter(create_torchax_tensor_with_partition_spec(
                self.v_bias, mesh, ('x', )),
                                    requires_grad=False)

    def _load_weights_from_qkv_linear(self, qkv_linear: nn.Module):
        q_proj_size, k_proj_size, _ = qkv_linear.output_sizes
        # The weight of qkv linear is a concatenation of q, k, and v weights
        # along the output dimension.
        qkv_weight = qkv_linear.weight.data.cpu()
        q_weight = Parameter(qkv_weight[:q_proj_size], requires_grad=False)
        k_weight = Parameter(qkv_weight[q_proj_size:q_proj_size + k_proj_size],
                             requires_grad=False)
        v_weight = Parameter(qkv_weight[q_proj_size + k_proj_size:],
                             requires_grad=False)
        self.register_parameter("q_weight", q_weight)
        self.register_parameter("k_weight", k_weight)
        self.register_parameter("v_weight", v_weight)

        if qkv_linear.bias is not None:
            q_bias = Parameter(qkv_linear.bias[:q_proj_size],
                               requires_grad=False)
            k_bias = Parameter(qkv_linear.bias[q_proj_size:q_proj_size +
                                               k_proj_size],
                               requires_grad=False)
            v_bias = Parameter(qkv_linear.bias[q_proj_size + k_proj_size:],
                               requires_grad=False)
            self.register_parameter("q_bias", q_bias)
            self.register_parameter("k_bias", k_bias)
            self.register_parameter("v_bias", v_bias)
        else:
            self.register_parameter("q_bias", None)
            self.register_parameter("k_bias", None)
            self.register_parameter("v_bias", None)

    def forward(self, input):
        # Same forward functionality as QKVParallelLinear, but doing qkv porj
        # separately.
        q_bias = self.q_bias if not self.skip_bias_add else None
        k_bias = self.k_bias if not self.skip_bias_add else None
        v_bias = self.v_bias if not self.skip_bias_add else None
        q_proj = F.linear(input, self.q_weight, q_bias)
        k_proj = F.linear(input, self.k_weight, k_bias)
        v_proj = F.linear(input, self.v_weight, v_bias)
        # The q/k/v projections will be split outside of the QKVParallelLinear.
        # Because we are replacing XlaQKVParallelLinear with the
        # QKVParallelLinear, we need to concatenate q, k, and v projections to
        # match the output shape of the QKVParallelLinear implementation even if
        # it seems to be redundant.
        # The concat and the following split will be noop, and should be
        # optimized away by the compiler.
        qkv_proj = torch.cat([q_proj, k_proj, v_proj], dim=-1)
        output_bias = torch.cat([q_bias, k_bias, v_bias], dim=-1) if \
                            self.skip_bias_add else None
        if not self.return_bias:
            return qkv_proj
        return qkv_proj, output_bias


def partition_column_parallel_linear(layer: torch.nn.Module,
                                     mesh: xs.Mesh) -> torch.nn.Module:
    assert isinstance(layer, ColumnParallelLinear)
    if VLLM_TORCHAX_ENABLED:

        # weight_t = layer.weight.data
        # if weight_t.dtype == torch.bfloat16:
        #     jax_t = jnp.array(weight_t.to(torch.float32).numpy()).astype(jnp.bfloat16)
        # else:
        #     jax_t = jnp.array(weight_t.numpy())
        sharding = NamedSharding(mesh, P('x', None))
        # jax_t = jax.device_put(jax_t, sharding)
        # torchax_t = torchax.tensor.Tensor(jax_t, torchax.default_env())
        torchax_t = create_torchax_tensor_with_partition_spec(
            layer.weight.data, mesh, ('x', None))
        layer.weight = Parameter(torchax_t,
                                 requires_grad=layer.weight.requires_grad)
        logger.info("Applied column-parallel sharding to %s", layer)
    else:
        xs.mark_sharding(layer.weight, mesh, ('x', None))
        logger.info("Applied column-parallel sharding to %s", layer)
    return layer


def partition_row_parallel_linear(layer: torch.nn.Module,
                                  mesh: xs.Mesh) -> torch.nn.Module:
    assert isinstance(layer, RowParallelLinear)
    if VLLM_TORCHAX_ENABLED:
        # weight_t = layer.weight.data
        # if weight_t.dtype == torch.bfloat16:
        #     jax_t = jnp.array(weight_t.to(torch.float32).numpy()).astype(jnp.bfloat16)
        # else:
        #     jax_t = jnp.array(weight_t.numpy())
        sharding = NamedSharding(mesh, P(None, 'x'))
        # jax_t = jax.device_put(jax_t, sharding)
        # torchax_t = torchax.tensor.Tensor(jax_t, torchax.default_env())
        torchax_t = create_torchax_tensor_with_partition_spec(
            layer.weight.data, mesh, (None, 'x'))
        layer.weight = Parameter(torchax_t,
                                 requires_grad=layer.weight.requires_grad)
        logger.info("Applied row-parallel sharding to %s", layer)
    else:
        xs.mark_sharding(layer.weight, mesh, (None, 'x'))
        logger.info("Applied row-parallel sharding to %s", layer)
    return layer


def partition_qkv_parallel_linear(layer: torch.nn.Module,
                                  mesh: xs.Mesh) -> torch.nn.Module:
    assert isinstance(layer, QKVParallelLinear)
    xla_layer = XlaQKVParallelLinear(layer, mesh)
    logger.info("Applied qkv parallel sharding to %s", layer)
    return xla_layer


def replicate_weights_buffers(module: torch.nn.Module, mesh) -> None:
    logger.info("Replicating weights and buffers for module %s", module)
    for name, param in module.named_parameters(recurse=False):
        torchax_t = create_torchax_tensor_with_partition_spec(
            param.data, mesh, ())
        torchax_param = Parameter(torchax_t, requires_grad=param.requires_grad)
        setattr(module, name, torchax_param)

    for name, buffer in module.named_buffers(recurse=False):
        torchax_t = create_torchax_tensor_with_partition_spec(buffer, mesh, ())
        # TODO: handle persistent buffer
        setattr(module, name, torchax_t)


MODULE_TYPE_TO_WRAPPING_FUNC = OrderedDict([
    ("QKVParallelLinear", partition_qkv_parallel_linear),
    ("ColumnParallelLinear", partition_column_parallel_linear),
    ("MergedColumnParallelLinear", partition_column_parallel_linear),
    ("RowParallelLinear", partition_row_parallel_linear),
    # (ColumnParallelLinear, partition_column_parallel_linear),
    # (RowParallelLinear, partition_row_parallel_linear),
])


def get_fqn(module):
    # Get the fully qualified name of the module
    return module.__class__.__qualname__


def shard_model(model: torch.nn.Module, mesh: "xs.Mesh") -> None:
    """
    Recursively check a PyTorch model and apply appropriate sharding based on 
    the MODULE_TYPE_TO_WRAPPING_FUNC mapping.
    
    Args:
        model: torch.nn.Module to process
        mesh: An XLA SPMD mesh object used for sharding
    """

    def _process_module(module, name=None, parent=None):
        model_processed = False
        for module_type, wrapping_func in MODULE_TYPE_TO_WRAPPING_FUNC.items():
            if get_fqn(module) == module_type:
                logger.info("processing module %s with type %s", module,
                            module_type)
                # if isinstance(module, module_type):
                wrapped_module = wrapping_func(module, mesh)

                assert parent is not None and name is not None, (
                    "Top Level module is not expected to be wrapped.")
                if wrapped_module is not module:
                    # Wrapped module and module are different py object.
                    # The original module should be replaced by the
                    # wrapped_module.
                    logger.info("replace %s with %s", module, wrapped_module)
                    setattr(parent, name, wrapped_module)

                module = wrapped_module
                model_processed = True
                break

        # Replicate the weights and buffers if the module is not processed
        if not model_processed and VLLM_TORCHAX_ENABLED and mesh is not None:
            replicate_weights_buffers(module, mesh)

        for child_name, child_module in list(module.named_children()):
            _process_module(child_module, child_name, module)

    # for name, tensor in model.named_parameters():
    #     logger.info("weight %s: %s %s", name, tensor.shape, tensor.dtype)

    # for name, tensor in model.named_buffers():
    #     logger.info("buffer %s: %s %s", name, tensor.shape, tensor.dtype)

    assert mesh is not None, "Mesh must be provided for sharding."
    _process_module(model)
