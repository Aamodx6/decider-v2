"""Losses: label-smoothed CE, EMD for ordinal `score`, and per-type weighting."""

import torch
import torch.nn.functional as F

from serialize import TASK_TYPES


NEG_INF = -1e30  # finite stand-in for -inf: exp() underflows to exactly 0.0, so p == 0 exactly and
                # the gradient w.r.t. that logit is exactly 0, while log p stays finite.


def smoothed_cross_entropy(logits, labels, option_mask, label_smoothing=0.0):
    """Label-smoothed CE with the smoothing term restricted to the VALID options.

    torch's F.cross_entropy(..., label_smoothing=eps) spreads eps over all C columns, so a masked
    padding column with logit -1e9 contributes eps/K * 1e9 to the loss.  That is not a rounding
    detail: on this task 45% of the questions are 2-option noul, so padded columns are the norm and
    the naive form returns ~1e7.  Renormalising the smoothing over the K real options makes the two
    forms identical when every column is valid.
    """
    valid = option_mask
    z = torch.where(valid, logits, torch.full_like(logits, NEG_INF))
    logp = torch.log_softmax(z, dim=-1)
    n = valid.sum(-1).clamp(min=1)
    nll = -logp.gather(1, labels.clamp(min=0).unsqueeze(1)).squeeze(1)
    if label_smoothing <= 0:
        return nll
    uniform = -(logp * valid.to(logp.dtype)).sum(-1) / n
    return (1.0 - label_smoothing) * nll + label_smoothing * uniform


def emd_loss(probs, target_idx, num_levels):
    """Earth-mover distance between the predicted level distribution and the one-hot target.

    (1/K) * sum_i |cdf_p(i) - cdf_t(i)|.  Zero exactly when probs is one-hot at target_idx.
    """
    K = num_levels
    cdf_p = probs[:, :K].cumsum(-1)
    ar = torch.arange(K, device=probs.device)
    cdf_t = (ar[None, :] >= target_idx[:, None]).to(probs.dtype)
    return (cdf_p - cdf_t).abs().mean(-1)


def compute_loss(logits, labels, task_type, num_options, label_smoothing=0.05, emd_lambda=0.5,
                 type_weights=None, emd_on="score"):
    """Per-example losses, weighted and averaged.

    logits [B, C] (padding slots are -inf / -1e9 and are never a label), labels [B], task_type [B].
    type_weights: optional dict {0: w0, 1: w1, 2: w2} applied to the per-example mean.
    """
    B, C = logits.shape
    # validity comes from the option count, not from the logit value: the head writes -inf into the
    # padding slots but callers also pass -1e9 / -1e30 sentinels, and isfinite() cannot tell those
    # apart from a real logit.
    valid = torch.arange(C, device=logits.device)[None, :] < num_options.unsqueeze(-1)
    ce = smoothed_cross_entropy(logits, labels, valid, label_smoothing)

    is_score = task_type == TASK_TYPES["score"]
    if emd_lambda > 0 and is_score.any():
        z = torch.where(valid, logits, torch.full_like(logits, NEG_INF))
        probs = torch.softmax(z, dim=-1)
        lvl = torch.arange(C, device=logits.device)[None, :].expand(B, C)
        k_eff = num_options.clamp(min=1)
        cdf_p = (probs * (lvl < k_eff[:, None]).float()).cumsum(-1)
        cdf_t = (lvl >= labels[:, None]).float() * (lvl < k_eff[:, None]).float()
        emd = (cdf_p - cdf_t).abs().sum(-1) / k_eff.float()
        loss = torch.where(is_score, ce + emd_lambda * emd, ce)
    else:
        loss = ce

    if type_weights:
        w = torch.tensor([type_weights.get(int(t), 1.0) for t in task_type.tolist()],
                         device=logits.device, dtype=loss.dtype)
        return (loss * w).sum() / w.sum().clamp(min=1e-6), loss.detach()
    return loss.mean(), loss.detach()


@torch.no_grad()
def expected_calibration_error(probs, labels, num_bins=10):
    """Top-label ECE. probs [N, C] (already softmaxed, padding columns zeroed), labels [N]."""
    conf, pred = probs.max(-1)
    correct = (pred == labels).float()
    n = labels.numel()
    ece = torch.zeros((), device=probs.device, dtype=torch.float64)
    edges = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    for i in range(num_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece = ece + m.sum().double() / n * (conf[m].mean().double() - correct[m].mean().double()).abs()
    return ece


@torch.no_grad()
def negative_log_likelihood(probs, labels):
    n = labels.numel()
    if n == 0:
        return torch.zeros(())
    idx = torch.arange(n, device=probs.device)
    return -probs[idx, labels].clamp_min(1e-12).log().mean()