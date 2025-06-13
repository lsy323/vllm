# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import jax
import numpy as np
import pytest
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from jax.sharding import Mesh

from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.distributed.tpu_distributed_utils import (
    XlaQKVParallelLinear, create_torchax_tensor_with_partition_spec)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import QKVParallelLinear


@pytest.fixture(autouse=True)
def setup_environment():
    # This is a fake config used for init dist env.
    # QKVParallelLinear needs dist env to be initialized.
    engine_args = EngineArgs(
        model="Qwen/Qwen2-1.5B-Instruct",
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )

    vllm_config = engine_args.create_engine_config()

    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)
        yield


MESH = None


def _get_spmd_mesh():
    global MESH
    if MESH is None:
        if os.environ.get('VLLM_TORCHAX_ENABLED', '0') == '1':
            devices = jax.devices()
            MESH = Mesh(devices, axis_names=('x', ))
        else:
            xr.use_spmd()
            num_devices = xr.global_runtime_device_count()
            mesh_shape = (num_devices, 1)
            device_ids = np.array(range(num_devices))
            MESH = xs.Mesh(device_ids, mesh_shape, ('x', 'y'))
    return MESH


@pytest.mark.parametrize("bias", [False, True])
# `xr.use_spmd()` will set a global state, and this state is not reversible.
# Therefore, non-SPMD tests should be run before SPMD tests.
@pytest.mark.parametrize("mesh", [_get_spmd_mesh()])
@pytest.mark.parametrize("device", ['jax'])
@torch.no_grad()
def test_xla_qkv_linear(bias, mesh, device):
    if device == 'jax':
        import torchax
        torchax.enable_globally()

    torch.manual_seed(123)

    qkv_linear = QKVParallelLinear(
        hidden_size=4096,
        head_size=128,
        total_num_heads=32,
        total_num_kv_heads=8,
        bias=bias,
        params_dtype=torch.bfloat16,
        return_bias=False,
    )

    qkv_linear.weight.data = torch.rand_like(qkv_linear.weight.data) / 10
    if bias:
        qkv_linear.bias.data = torch.rand_like(qkv_linear.bias.data)

    xla_qkv_linear = XlaQKVParallelLinear(qkv_linear, mesh=mesh)
    # If jitted got small relative error
    # if device == 'jax':
    # xla_qkv_linear = JittableModule(xla_qkv_linear)

    input_tensor = torch.rand(10, 4096, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to(device)
    qkv_linear = qkv_linear.to(device)
    xla_qkv_linear = xla_qkv_linear.to(device)
    if device != 'cpu':
        input_tensor_rep = create_torchax_tensor_with_partition_spec(
            input_tensor, mesh=mesh)

    output = qkv_linear(input_tensor)
    xla_output = xla_qkv_linear(input_tensor_rep)
    if device == 'jax':
        # Convert both to float32 to avoid dtype promotion issues between float16 and bfloat16
        output_np = np.asarray(output.jax()).astype(np.float32)
        xla_output_np = np.asarray(xla_output.jax()).astype(np.float32)
        max_abs_error = np.max(np.abs(output_np - xla_output_np))
        print(f"Max absolute error: {max_abs_error}")
        max_rel_error = np.max(
            np.abs((output_np - xla_output_np) / (output_np + 1e-8)))
        print(f"Max relative error: {max_rel_error}")
        assert torch.allclose(torch.from_numpy(output_np),
                              torch.from_numpy(xla_output_np))
    else:
        assert torch.allclose(output.cpu(), xla_output.cpu())
