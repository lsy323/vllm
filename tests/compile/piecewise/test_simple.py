# SPDX-License-Identifier: Apache-2.0
"""
Test the piecewise compilation with a simple model so that we
can exactly calculate the expected output and side effects.
"""

import torch
from torch import nn
from torch.library import Library
from typing import TYPE_CHECKING, Optional, cast

from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.compilation.counter import compilation_counter
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CompilationConfig, CompilationLevel, VllmConfig,
                         set_current_vllm_config)


def silly_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    out: torch.Tensor) -> None:
    out.copy_(q)
    out[0] += 1


def silly_attention_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         out: torch.Tensor) -> None:
    return


@support_torch_compile
class SillyModel(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = '',
                 **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overall effect:
        x += 1
        x[0] += 2
        global_counter += 2
        """
        x = x + 1
        x = x + 2
        out = torch.empty_like(x)
        # torch.ops.silly.attention(x, x, x, out)
        silly_attention(x, x, x, out)
        x = out
        x = x - 2
        x = x - 1
        out = torch.empty_like(x)
        # torch.ops.silly.attention(x, x, x, out)
        silly_attention(x, x, x, out)
        x = out
        x = x + 1
        return x


class ModelWrapperV1(TorchCompileWrapperWithCustomDispatcher):

    def __init__(self, model: nn.Module):
        self.model = model
        compiled_callable = torch.compile(self.forward, backend="openxla", fullgraph=True, dynamic=False)
        super().__init__(compiled_callable,
                         compilation_level=CompilationLevel.DYNAMO_ONCE,
                         register_hook=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model(
            x
        )

        return x

    def __call__(
        self,
        x: torch.Tensor,
    ):
        print(len(self.compiled_codes))
        # return self.compiled_callable(input_ids, positions, kv_caches, inputs_embeds)
        # return self.forward(input_ids, positions, kv_caches, inputs_embeds)
        if len(self.compiled_codes) > 0:
            dispatch_id = 0
            with self.dispatch_to_code(dispatch_id):
                return self.forward(x)
        else:
            return self.compiled_callable(x)
            
def test_simple_piecewise_compile():

    vllm_config = VllmConfig(compilation_config=CompilationConfig(
        level=CompilationLevel.DYNAMO_ONCE,
        backend="openxla",
    ))
    with set_current_vllm_config(vllm_config):
        model = SillyModel(vllm_config=vllm_config, prefix='')

    model = model.to('xla')
    # model = ModelWrapperV1(model)
    inputs = torch.randn(100).to('xla')
    # torch._dynamo.mark_dynamic(inputs, 0)
    model(inputs)
    print("remove")
    # if isinstance(model, TorchCompileWrapperWithCustomDispatcher):
    #     torch._dynamo.eval_frame.remove_from_cache(model.original_code_object)

    model(torch.randn(2).to('xla'))   
    model(torch.randn(1).to('xla'))

    input = torch.zeros(2).to('xla')
    output = model(input)
    assert torch.allclose(output.cpu(), torch.tensor([3., 1.]))
