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
from torchax.interop import extract_all_buffers, jax_jit

from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.distributed.tpu_distributed_utils import (
    XlaMergedColumnParallelLinear, create_torchax_tensor_with_partition_spec)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import MergedColumnParallelLinear


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


@pytest.mark.parametrize("bias", [False])
# `xr.use_spmd()` will set a global state, and this state is not reversible.
# Therefore, non-SPMD tests should be run before SPMD tests.
@pytest.mark.parametrize("mesh", [_get_spmd_mesh()])
@pytest.mark.parametrize("device", ['jax'])
@torch.no_grad()
def test_xla_merged_col_parallel_linear(bias, mesh, device):
    if device == 'jax':
        import torchax
        torchax.enable_globally()

    torch.manual_seed(123)

    merged_col_parallel_linear = MergedColumnParallelLinear(
        input_size=4096,
        output_sizes=[14336, 14336],
        bias=bias,
        params_dtype=torch.bfloat16,
        return_bias=False,
    )

    merged_col_parallel_linear.weight.data = torch.rand_like(
        merged_col_parallel_linear.weight.data) / 10
    if bias:
        merged_col_parallel_linear.bias.data = torch.rand_like(
            merged_col_parallel_linear.bias.data)

    xla_merged_col_parallel_linear = XlaMergedColumnParallelLinear(
        merged_col_parallel_linear, mesh=mesh)

    params, buffers = extract_all_buffers(xla_merged_col_parallel_linear)
    params_and_buffers = {**params, **buffers}

    @jax_jit
    def wrapped_func(params_and_buffers, args, **kwargs):
        return torch.func.functional_call(xla_merged_col_parallel_linear,
                                          params_and_buffers,
                                          args=args,
                                          kwargs=kwargs,
                                          strict=True)

    # breakpoint()
    print(f"check {params_and_buffers}")
    print(
        f"check model statedict {xla_merged_col_parallel_linear.state_dict()}")

    # If jitted got small relative error
    # if device == 'jax':
    # xla_merged_col_parallel_linear = JittableModule(xla_merged_col_parallel_linear)

    input_tensor = torch.rand(10, 4096, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to(device)
    merged_col_parallel_linear = merged_col_parallel_linear.to(device)
    xla_merged_col_parallel_linear = xla_merged_col_parallel_linear.to(device)
    if device != 'cpu':
        input_tensor_rep = create_torchax_tensor_with_partition_spec(
            input_tensor, mesh=mesh)

    output = merged_col_parallel_linear(input_tensor)
    # xla_output = xla_merged_col_parallel_linear(input_tensor_rep)
    xla_output = wrapped_func(params_and_buffers, (input_tensor_rep, ))
    if device == 'jax':
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
