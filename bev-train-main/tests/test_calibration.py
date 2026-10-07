"""calibrate.py helpers: temperature fitting, ECE, and temperature application in answer()."""
import math

import torch
import torch.nn.functional as F

from calibrate import fit_temperature
from inference import answer, top_label_ece

META = {"max_state_tokens": 128, "max_choice_tokens": 32}
DEVICE = torch.device("cpu")
STATE = "The customer ordered on 2026-03-15. The warranty ended 2026-03-14."
QUESTION = {"type": "choice", "instructions": "Which statement applies?",
            "criteria": {"a": "first option", "b": "second option", "c": "third option"}, "label": "a"}


def test_fit_temperature_recovers_known_temperature():
    torch.manual_seed(0)
    z = torch.randn(4096, 4) * 2.0                     # fixed logits
    t_true = 2.0
    labels = torch.multinomial(F.softmax(z / t_true, dim=-1), 1).squeeze(1)  # labels ~ softmax(z / T)

    fitted = fit_temperature(z, labels)
    nll_before = F.cross_entropy(z, labels)
    nll_after = F.cross_entropy(z / fitted, labels)
    assert math.isclose(fitted, t_true, rel_tol=0.15), f"fitted T={fitted}, expected ~{t_true}"
    assert nll_after <= nll_before + 1e-6              # scaling can only help or tie on the fit data


def test_top_label_ece_bounds():
    # confidence 0.9 with exactly 90% accuracy -> perfectly calibrated
    hits = [1.0] * 90 + [0.0] * 10
    assert top_label_ece([0.9] * 100, hits) < 1e-6
    assert top_label_ece([0.9] * 100, [0.0] * 100) > 0.8   # confident + wrong -> high ECE
    assert math.isnan(top_label_ece([], []))


def test_answer_applies_temperatures(networks):
    network = networks[0]
    raw = answer(network, META, STATE, QUESTION, DEVICE)["probabilities"]
    hot = answer(network, {**META, "temperatures": {"choice": 4.0}}, STATE, QUESTION, DEVICE)["probabilities"]

    # dividing logits by T > 1 flattens the distribution but keeps the argmax
    assert max(raw, key=raw.get) == max(hot, key=hot.get)
    assert max(abs(raw[k] - hot[k]) for k in raw) > 1e-4
    assert abs(sum(hot.values()) - 1.0) < 1e-6
