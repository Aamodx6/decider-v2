"""The network: truncated Qwen3 backbone + LoRA, and the two decision heads.

DecisionHeadV1 is a byte-faithful replica of bev-decider's ChoiceHead (verified against
github.com/avbiswas/bev-decider/src/bev_decider/model.py and bev-train/network.py), so the
baseline arm is the real thing rather than a paraphrase of it.

DecisionHeadV2 adds the pieces decider-v2 is testing: a Perceiver-style context summarizer over
the prefix hidden states, option self-attention that is explicitly position-free, cross-attention
to that summary, a type embedding and a rank embedding for `score`.
"""

import json
import math
import os

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model

from serialize import (NUM_TASK_TYPES, POOL_BEV, POOL_MEAN, POOL_OPT_END, TASK_TYPES, pad_ranks,
                        segment_mean)


# --------------------------------------------------------------------------- backbone
def build_backbone(model_name="Qwen/Qwen3-0.6B", num_layers=20, attn_implementation="sdpa",
                   dtype=torch.float32, config=None, random_init=False, vocab_size=512):
    """Qwen3 truncated to its first `num_layers` decoder layers.

    Truncating `num_hidden_layers` keeps Qwen3Model.norm: the final RMSNorm IS applied after the
    last kept layer (the norm is a property of Qwen3Model, not of the last layer).
    """
    if config is None:
        if random_init:
            config = Qwen3Config(
                vocab_size=vocab_size, hidden_size=64, intermediate_size=128,
                num_hidden_layers=num_layers,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                max_position_embeddings=8192, tie_word_embeddings=True,
            )
            model = Qwen3Model(config)
        else:
            overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
            model = Qwen3Model.from_pretrained(model_name, dtype=dtype, **overrides)
    else:
        model = Qwen3Model(config)
    model.config._attn_implementation = attn_implementation
    for m in model.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = attn_implementation
    return model


def lora_target_regex(num_layers, last_k_layers, include_mlp):
    """PEFT target regex.  last_k_layers=0 -> no LoRA.  include_mlp adds gate/up/down_proj."""
    lo = 0 if last_k_layers in (0, None) else max(0, num_layers - last_k_layers)
    if last_k_layers == 0:
        raise ValueError("last_k_layers=0 means a frozen backbone; no LoRA regex is built")
    layers = "|".join(str(i) for i in range(lo, num_layers))
    pat = rf".*layers\.({layers})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    if include_mlp:
        pat = f"({pat})|.*layers\\.({layers})\\.mlp\\.(gate_proj|up_proj|down_proj)"
    return pat


def attach_lora(backbone, num_layers, r, alpha, dropout, last_k_layers, include_mlp):
    from peft import LoraConfig, get_peft_model
    if last_k_layers == 0:
        backbone.requires_grad_(False)
        return backbone
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=lora_target_regex(backbone.config.num_hidden_layers, last_k_layers, include_mlp),
        bias="none",
    )
    return get_peft_model(backbone, cfg)


# --------------------------------------------------------------------------- head v1 (bev replica)
class SelfAttentionV1(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.new_dim = new_dim
        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        self.o = nn.Linear(new_dim, hidden_dim)

    def forward(self, x, padding_mask):
        scores = torch.einsum("bqd,bkd->bqk", self.q(x), self.k(x)) / math.sqrt(self.new_dim)
        scores = scores.masked_fill(~padding_mask[:, None, :], -float("inf"))
        return self.o(torch.einsum("bqk,bkd->bqd", torch.softmax(scores, dim=-1), self.v(x)))


class AttentionBlockV1(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttentionV1(hidden_dim, new_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, new_dim), nn.GELU(), nn.Linear(new_dim, hidden_dim))

    def forward(self, x, padding_mask):
        x = x + self.attn(self.norm1(x), padding_mask)
        return x + self.mlp(self.norm2(x))


class DecisionHeadV1(nn.Module):
    """Exact replica of bev-decider ChoiceHead: [task token, options..., answer] -> bilinear logits."""

    def __init__(self, hidden_dim, new_dim=512, num_layers=2, num_task_types=NUM_TASK_TYPES):
        super().__init__()
        self.new_dim = new_dim
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([AttentionBlockV1(hidden_dim, new_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, new_dim)
        self.choice_proj = nn.Linear(hidden_dim, new_dim)

    def forward(self, option_h, answer_h, option_mask, task_type, **kw):
        B = option_h.shape[0]
        task_token = self.task_embedding(task_type).unsqueeze(1)
        x = self.input_norm(torch.cat([task_token, option_h, answer_h.unsqueeze(1)], dim=1))
        always = torch.ones(B, 1, dtype=torch.bool, device=option_mask.device)
        padding_mask = torch.cat([always, option_mask, always], dim=1)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.final_norm(x)
        answer = self.answer_proj(x[:, -1])
        choices = self.choice_proj(x[:, 1:-1])
        logits = torch.einsum("bd,bcd->bc", answer, choices) / math.sqrt(self.new_dim)
        return logits.masked_fill(~option_mask, -float("inf"))


# --------------------------------------------------------------------------- head v2
class HeadBlockV2(nn.Module):
    """option self-attn (no positions) -> cross-attn to the context summary -> FFN."""

    def __init__(self, d, h, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, ctx, opt_pad):
        h = self.n1(x)
        x = x + self.sa(h, h, h, key_padding_mask=opt_pad, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.ca(h, ctx, ctx, need_weights=False)[0]
        return x + self.ff(self.n3(x))


class DecisionHeadV2(nn.Module):
    def __init__(self, d_model=1024, d=512, h=8, num_layers=2, n_ctx=8,
                 num_task_types=NUM_TASK_TYPES, use_null=False, use_ctx=True, use_rank=True,
                 dropout=0.0):
        super().__init__()
        self.d = d
        self.use_ctx = use_ctx
        self.use_rank = use_rank
        self.use_null = use_null
        self.proj_opt = nn.Linear(d_model, d)
        self.type_emb = nn.Embedding(num_task_types, d)
        if use_ctx:
            self.proj_ctx = nn.Linear(d_model, d)
            self.ctx_q = nn.Parameter(torch.randn(n_ctx, d) * 0.02)
            self.ctx_attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        if use_rank:
            self.rank_mlp = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        if use_null:
            self.null_opt = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([HeadBlockV2(d, h, dropout) for _ in range(num_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def context_summary(self, prefix_h, prefix_pad):
        if not self.use_ctx:
            return None
        c = self.proj_ctx(prefix_h)
        ctx, _ = self.ctx_attn(self.ctx_q.unsqueeze(0).expand(c.shape[0], -1, -1), c, c,
                               key_padding_mask=prefix_pad, need_weights=False)
        return ctx

    def forward(self, option_h, prefix_h=None, prefix_pad=None, option_mask=None, task_type=None,
                ranks=None, **kw):
        B, K, _ = option_h.shape
        x = self.proj_opt(option_h) + self.type_emb(task_type)[:, None, :]
        if self.use_rank and ranks is not None:
            x = x + self.rank_mlp(pad_ranks(option_mask, ranks).unsqueeze(-1))
        opt_pad = ~option_mask
        if self.use_null:
            null = self.null_opt.to(x.dtype).expand(B, 1, -1)
            x = torch.cat([null, x], dim=1)
            opt_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), opt_pad], dim=1)
        ctx = self.context_summary(prefix_h, prefix_pad) if prefix_h is not None else None
        for blk in self.blocks:
            if ctx is None:
                h = blk.n1(x)
                x = x + blk.sa(h, h, h, key_padding_mask=opt_pad, need_weights=False)[0]
                x = x + blk.ff(blk.n2(x))
            else:
                x = blk(x, ctx, opt_pad)
        logits = self.out(x).squeeze(-1)
        return logits.masked_fill(opt_pad, -float("inf"))


# --------------------------------------------------------------------------- network
class DeciderV2(nn.Module):
    def __init__(self, backbone, head, pool=POOL_BEV):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pool = pool

    def forward(self, input_ids, position_ids, attention_mask, option_read_idx, option_mask,
                option_seg, answer_end_idx, task_type, token_valid=None, ranks=None,
                prefix_len=None, **kw):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, use_cache=False)
        hidden = out.last_hidden_state.float()
        B = hidden.shape[0]
        bidx = torch.arange(B, device=hidden.device)

        if self.pool == POOL_MEAN:
            option_h = segment_mean(hidden, option_seg, option_mask)
        else:
            option_h = hidden[bidx[:, None], option_read_idx]
        answer_h = hidden[bidx, answer_end_idx]

        # The context summarizer sees the PREFIX ONLY.  Letting it see the option blocks too would
        # still be permutation-invariant in exact arithmetic (the learned queries are a fixed
        # function of the key set) but it makes the guarantee depend on summation order, and it
        # contradicts the design: the summary is of state + question, not of the candidate answers.
        prefix_h = prefix_pad = None
        if token_valid is not None and prefix_len is not None:
            P = int(prefix_len.max())
            prefix_h = hidden[:, :P]
            prefix_pad = ~token_valid[:, :P]

        # the head always runs in fp32: bf16 logits are too coarse for a stable loss (as in bev)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head(option_h=option_h.float(), answer_h=answer_h.float(),
                             prefix_h=None if prefix_h is None else prefix_h.float(),
                             prefix_pad=prefix_pad, option_mask=option_mask,
                             task_type=task_type, ranks=ranks)


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_model(args):
    """args: a Namespace / dict with model + lora + head + pool fields."""
    g = lambda d, k, dv: d.get(k, dv) if isinstance(d, dict) else getattr(d, k, dv)
    num_layers = g(args, "num_layers", 20)
    attn_impl = g(args, "attn_implementation", "sdpa")
    random_init = g(args, "random_init", False)
    backbone = build_backbone(g(args, "backbone_name", "Qwen/Qwen3-0.6B"), num_layers,
                              attn_impl, random_init=random_init,
                              vocab_size=g(args, "vocab_size", 512))
    backbone = attach_lora(backbone, num_layers, g(args, "lora_r", 16), g(args, "lora_alpha", 32),
                           g(args, "lora_dropout", 0.0), g(args, "lora_last_k_layers", num_layers),
                           g(args, "lora_include_mlp", True))
    d_model = backbone.config.hidden_size
    head_name = g(args, "head", "v2")
    common = dict(num_task_types=NUM_TASK_TYPES)
    if head_name == "v1":
        head = DecisionHeadV1(hidden_dim=d_model, new_dim=g(args, "head_new_dim", 512),
                              num_layers=g(args, "head_layers", 2), **common)
    else:
        head = DecisionHeadV2(
            d_model=d_model, d=g(args, "head_new_dim", 512), h=g(args, "head_heads", 8),
            num_layers=g(args, "head_layers", 2), n_ctx=g(args, "n_ctx", 8),
            use_null=g(args, "use_null", False), use_ctx=g(args, "use_ctx", True),
            use_rank=g(args, "use_rank", True), dropout=g(args, "head_dropout", 0.0), **common)
    return DeciderV2(backbone, head, pool=g(args, "pool", POOL_BEV))


# --------------------------------------------------------------------------- checkpoint io
def _is_peft(m):
    return hasattr(m, "peft_config") and hasattr(m, "get_base_model") and type(m).__name__ == "PeftModel"


def _unwrap(backbone):
    """The Qwen3Model underneath a (possibly peft-wrapped) backbone."""
    if not _is_peft(backbone):
        return backbone
    b = backbone.get_base_model()
    while not isinstance(b, Qwen3Model) and hasattr(b, "model"):
        b = b.model
    return b


def save_checkpoint(network, path, meta, merge=False):
    """Write model.safetensors (backbone bf16 + fp32 head) and, when a LoRA is attached,
    lora.safetensors.

    merge=True folds the adapter into the backbone for a self-contained deployment artifact.  It
    runs on a deepcopy, so the training model is never mutated (an earlier version called
    merge_and_unload() in place, which silently stripped the adapter from the live model).
    """
    import copy
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    backbone = network.backbone
    sd = {}
    if _is_peft(backbone):
        lora = {k: v.detach().cpu().contiguous() for k, v in backbone.state_dict().items()
                if "lora_" in k}
        save_file(lora, os.path.join(path, "lora.safetensors"))
        source = copy.deepcopy(backbone).merge_and_unload() if merge else _unwrap(backbone)
    else:
        source = _unwrap(backbone)
    for k, v in source.state_dict().items():
        sd["backbone." + k] = v.detach().to(torch.bfloat16).contiguous()
    for k, v in network.head.state_dict().items():
        sd["head." + k] = v.detach().contiguous()
    save_file(sd, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


def load_checkpoint(network, path, strict_head=True):
    from safetensors.torch import load_file
    weights = load_file(os.path.join(path, "model.safetensors"))
    backbone = network.backbone

    lora_file = os.path.join(path, "lora.safetensors")
    if os.path.exists(lora_file):
        if not _is_peft(backbone):
            raise ValueError(f"{path} has a LoRA adapter but the rebuilt network has none "
                             "(lora_last_k_layers=0?)")
        from peft import set_peft_model_state_dict
        res = set_peft_model_state_dict(backbone, load_file(lora_file))
        bad = [k for k in res.unexpected_keys if "lora_" in k] + \
              [k for k in res.missing_keys if "lora_" in k]
        if bad:
            raise ValueError(f"LoRA weights do not match the config: {bad[:5]}")

    base = _unwrap(backbone)
    bsd = {k[len("backbone."):]: v.float() for k, v in weights.items() if k.startswith("backbone.")}
    emb = base.get_input_embeddings().weight
    if bsd["embed_tokens.weight"].shape[0] != emb.shape[0]:
        # mode="special" resized the embedding at train time; rebuild it the same way
        from serialize import extend_tokenizer
        from transformers import AutoTokenizer
        args = json.load(open(os.path.join(path, "meta.json")))["args"]
        tok = AutoTokenizer.from_pretrained(args["backbone_name"])
        extend_tokenizer(tok)
        with torch.no_grad():
            mean = emb.data.mean(dim=0)
            base.resize_token_embeddings(len(tok))
            base.get_input_embeddings().weight.data.copy_(mean.expand(len(tok), -1))
    missing, unexpected = base.load_state_dict(bsd, strict=False)
    unexpected = [k for k in unexpected if "rotary_emb" not in k and "inv_freq" not in k]
    if strict_head and unexpected:
        raise ValueError(f"unexpected backbone keys: {unexpected[:5]}")

    hsd = {k[len("head."):]: v for k, v in weights.items() if k.startswith("head.")}
    network.head.load_state_dict({k: v.float() for k, v in hsd.items()})
    return network


def load_reference_bev_decider(repo="avbiswas/bev-decider-0.4B", revision=None, device="cpu",
                               cache_dir=None, max_state_tokens=2048, max_choice_tokens=64):
    """Load the published avbiswas/bev-decider-0.4B checkpoint as a DeciderV2(head=v1).

    That checkpoint is exactly ChoiceHeadV1 on a merged 20-layer Qwen3-0.6B, so no peft adapter is
    needed and no bev-decider package import is required.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    path = snapshot_download(repo, revision=revision, cache_dir=cache_dir)
    cfg = json.loads(open(os.path.join(path, "config.json")).read())
    bb_cfg = Qwen3Config.from_pretrained(path, **json.loads(
        open(os.path.join(path, cfg["backbone_config"])).read()))
    backbone = Qwen3Model(bb_cfg)
    weights = load_file(os.path.join(path, "model.safetensors"))
    backbone.load_state_dict({k[len("backbone."):]: v.float() for k, v in weights.items()
                              if k.startswith("backbone.")})
    head = DecisionHeadV1(hidden_dim=backbone.config.hidden_size, **cfg["head"])
    head.load_state_dict({k[len("head."):]: v.float() for k, v in weights.items()
                          if k.startswith("head.")})
    net = DeciderV2(backbone.float(), head, pool=POOL_BEV).to(device).eval()
    meta = {**cfg, "resolved_max_state_tokens": max_state_tokens, "resolved_max_choice_tokens": max_choice_tokens}
    return net, meta, path