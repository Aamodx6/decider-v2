"""answer() probabilities must match the evaluate() path (same dataset + collate + forward)."""
import torch

from dataset import BEVDataset, collate_fn, question_to_choices
from inference import answer, evaluate

META = {"max_state_tokens": 128, "max_choice_tokens": 32}
DEVICE = torch.device("cpu")
STATE = ("The customer ordered on 2026-03-15. The warranty was valid through 2026-03-14. "
         "The order total was 108.40 with free shipping over 100.00.")
DOMAIN = "Parity test domain"

QUESTIONS = [
    {"type": "choice", "instructions": "Which statement applies?",
     "criteria": {"covered": "the warranty still covers it", "expired": "the warranty expired",
                  "unknown": "not stated"}, "label": "expired"},
    {"type": "noul", "instructions": "Is the order still under warranty?", "label": False},
    {"type": "score", "instructions": "How urgent is this request?",
     "criteria": ["calm", "annoyed", "furious"], "label": 2},
]


def make_row(question):
    choices, label, label_probs = question_to_choices(question)
    return {"state": STATE, "task_type": {"choice": 0, "noul": 1, "score": 2}[question["type"]],
            "instructions": question["instructions"], "choices": choices, "label": label,
            "domain": DOMAIN, "label_probs": label_probs or []}


def test_answer_matches_evaluate_path(networks):
    rows = [make_row(q) for q in QUESTIONS]
    dataset = BEVDataset(rows, META["max_state_tokens"], META["max_choice_tokens"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(rows), collate_fn=collate_fn)

    max_diff = 0.0
    for ctx_queries, network in networks.items():
        # the exact forward pass evaluate() runs (only logits -> probs differ)
        batch = next(iter(loader))
        with torch.autocast(DEVICE.type, enabled=False):
            batched = torch.softmax(network(**batch).float(), dim=-1).tolist()

        for i, question in enumerate(QUESTIONS):
            single = answer(network, META, STATE, question, DEVICE)
            if question["type"] == "choice":
                got = [single["probabilities"][key] for key in question["criteria"]]
            elif question["type"] == "noul":
                got = [1.0 - single["noul"], single["noul"]]
            else:
                got = single["probabilities"]
            for a, b in zip(got, batched[i]):
                max_diff = max(max_diff, abs(a - b))

        # evaluate() reports metrics over the same forward pass: its loss is the CE of those probs
        metrics = evaluate(network, loader, DEVICE)
        expected_loss = sum(-torch.log(torch.tensor(batched[i][rows[i]["label"]]))
                            for i in range(len(rows))).item() / len(rows)
        assert abs(metrics["loss"] - expected_loss) < 1e-4
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "ece" in metrics and "accuracy_domain_parity_test_domain" in metrics

    print(f"\nparity: max |answer() - evaluate path| = {max_diff:.3e} (threshold 1e-5)")
    assert max_diff <= 1e-5, f"max difference {max_diff:.3e} exceeds the 1e-5 threshold"
