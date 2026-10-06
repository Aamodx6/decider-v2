"""Order invariance: per-key probabilities must not depend on option order.

fp32 on CPU, random option permutations, 3..12 options, threshold <= 1e-5.
The measured maximum difference is printed so it can be reported, not just asserted.
"""
import random

import torch

from inference import answer

META = {"max_state_tokens": 128, "max_choice_tokens": 32}
DEVICE = torch.device("cpu")
STATE = ("The customer ordered on 2026-03-15. The warranty was valid through 2026-03-14. "
         "The order total was 108.40 with free shipping over 100.00. The customer emailed "
         "asking whether a repair is still free of charge.")
INSTRUCTIONS = "Which key best describes the situation?"


def make_question(k):
    return {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {f"key_{i}": f"description of candidate {i} for this state" for i in range(k)},
        "label": "key_0",
    }


def run(network, question):
    return answer(network, META, STATE, question, DEVICE)["probabilities"]


def test_order_invariance(networks):
    max_diff = 0.0
    for ctx_queries, network in networks.items():
        for k in range(3, 13):  # 3 to 12 options
            question = make_question(k)
            keys = list(question["criteria"])
            base = run(network, question)
            rng = random.Random(k)
            for _ in range(4):  # random permutations
                order = keys[:]
                rng.shuffle(order)
                permuted = dict(question)
                permuted["criteria"] = {key: question["criteria"][key] for key in order}
                shuffled = run(network, permuted)
                for key in keys:
                    max_diff = max(max_diff, abs(base[key] - shuffled[key]))
    print(f"\norder invariance: max |Δ probability| over all permutations = {max_diff:.3e} (threshold 1e-5)")
    assert max_diff <= 1e-5, f"max difference {max_diff:.3e} exceeds the 1e-5 threshold"
