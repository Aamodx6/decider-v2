import pytest
import torch

from network import build_network

# Tiny networks (2 backbone layers, 64-d head, frozen Qwen) so the tests run on CPU in seconds.
# fp32 on CPU: autocast is only enabled for cuda/mps in inference.autocast, so probabilities are fp32.
META = {"max_state_tokens": 128, "max_choice_tokens": 32}


@pytest.fixture(scope="session")
def networks():
    built = {}
    for ctx_queries in (0, 8):
        torch.manual_seed(0)
        built[ctx_queries] = build_network(
            "Qwen/Qwen3-0.6B", 2, 8, 16, 0.0, 0,
            {"new_dim": 64, "num_layers": 2, "num_task_types": 3, "ctx_queries": ctx_queries},
        ).eval()
    return built
