import math

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Model


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, new_dim):
        super().__init__()
        self.new_dim = new_dim
        self.q = nn.Linear(hidden_dim, new_dim)
        self.k = nn.Linear(hidden_dim, new_dim)
        self.v = nn.Linear(hidden_dim, new_dim)
        self.o = nn.Linear(new_dim, hidden_dim)

    def forward(self, x, padding_mask, kv=None):

        # x: [B, T, H], padding_mask: [B, T] (True = may be attended to)
        # kv: cross-attention source; padding_mask always describes the keys (kv, or x itself)
        src = x if kv is None else kv
        q = self.q(x)
        k = self.k(src)
        v = self.v(src)

        # compute attention weights, padded keys are never attended to
        scores = torch.einsum("bqd,bkd->bqk", [q, k])/math.sqrt(self.new_dim) # [B, T, S]
        scores = scores.masked_fill(~padding_mask[:, None, :], -float("inf"))
        weights = torch.softmax(scores, dim=-1)

        out = torch.einsum("bqk,bkd->bqd", [weights, v])

        return self.o(out)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, new_dim, ctx_queries=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, new_dim)
        # Optional cross-attention sub-layer to the ctx summary of the prefix (head.ctx_queries > 0)
        self.norm_ctx = nn.LayerNorm(hidden_dim) if ctx_queries else None
        self.ctx_cross = SelfAttention(hidden_dim, new_dim) if ctx_queries else None
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, new_dim),
            nn.GELU(),
            nn.Linear(new_dim, hidden_dim),
        )

    def forward(self, x, padding_mask, ctx=None):
        x = x + self.attn(self.norm1(x), padding_mask)
        if ctx is not None:
            ctx_mask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=ctx.device)
            x = x + self.ctx_cross(self.norm_ctx(x), ctx_mask, kv=ctx)
        x = x + self.mlp(self.norm2(x))
        return x


class ChoiceHead(nn.Module):
    """Runs self-attention over [task token, choice embeddings..., answer embedding] and scores each choice."""

    def __init__(self, hidden_dim, new_dim, num_layers, num_task_types, ctx_queries=0):
        super().__init__()
        self.new_dim = new_dim
        self.ctx_queries = ctx_queries
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        if ctx_queries:
            # Learned queries that summarize the prefix (question + state) hidden states
            self.ctx_q = nn.Parameter(torch.randn(ctx_queries, hidden_dim) * 0.02)
            self.ctx_norm = nn.LayerNorm(hidden_dim)
            self.prefix_norm = nn.LayerNorm(hidden_dim)
            self.ctx_attn = SelfAttention(hidden_dim, new_dim)
        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, new_dim, ctx_queries) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, new_dim)
        self.choice_proj = nn.Linear(hidden_dim, new_dim)

    def forward(self, choice_embeddings, answer_embedding, choice_mask, task_type, prefix_hidden=None, prefix_mask=None):
        # choice_embeddings: [B, C, H], answer_embedding: [B, H], choice_mask: [B, C], task_type: [B]
        # prefix_hidden / prefix_mask: backbone states of the tokens before the first option (ctx_queries only)
        B = choice_embeddings.shape[0]
        task_token = self.task_embedding(task_type).unsqueeze(1)

        ctx = None
        if self.ctx_queries:
            if prefix_hidden is None or prefix_mask is None:
                raise ValueError("head.ctx_queries > 0 needs prefix_len in the batch (prefix_hidden/prefix_mask)")
            queries = self.ctx_q.unsqueeze(0).expand(B, -1, -1)          # [B, n_ctx, H]
            ctx = self.ctx_attn(self.ctx_norm(queries), prefix_mask, kv=self.prefix_norm(prefix_hidden))

        x = torch.cat([task_token, choice_embeddings, answer_embedding.unsqueeze(1)], dim=1) # [B, C+2, H]
        x = self.input_norm(x)

        always_valid = torch.ones(B, 1, dtype=torch.bool, device=choice_mask.device)
        padding_mask = torch.cat([always_valid, choice_mask, always_valid], dim=1)
        for layer in self.layers:
            x = layer(x, padding_mask, ctx)
        x = self.final_norm(x)

        # Bilinear score between the answer slot and every choice slot -> works for any number of choices
        answer = self.answer_proj(x[:, -1])        # [B, D]
        choices = self.choice_proj(x[:, 1:-1])     # [B, C, D]
        logits = torch.einsum("bd,bcd->bc", [answer, choices])/math.sqrt(self.new_dim)
        return logits.masked_fill(~choice_mask, -float("inf"))


class BEVNetwork(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, position_ids, attention_mask, choice_read_idx, choice_mask, answer_end_idx, task_type, prefix_len=None, **kwargs):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False
        )
        hidden = output.last_hidden_state.float()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)

        choice_embeddings = hidden[batch_idx[:, None], choice_read_idx]   # [B, C, H]
        answer_embedding = hidden[batch_idx, answer_end_idx]              # [B, H]
        # Head runs in fp32 even under bf16 autocast: bf16 logits are too coarse for a stable loss
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            if self.head.ctx_queries:
                if prefix_len is None:
                    raise ValueError("head.ctx_queries > 0 needs prefix_len in the batch (from collate_fn)")
                prefix_mask = torch.arange(hidden.shape[1], device=hidden.device)[None] < prefix_len[:, None]
                return self.head(choice_embeddings, answer_embedding, choice_mask, task_type, hidden, prefix_mask)
            return self.head(choice_embeddings, answer_embedding, choice_mask, task_type)


def load_backbone(model_name, num_layers=None):
    # Only the first num_layers decoder layers are built and loaded; the rest are never executed
    overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
    return Qwen3Model.from_pretrained(model_name, dtype=torch.float32, **overrides)


LORA_TARGETS = {
    # lora.target: attn = q/k/v/o of the attention; attn_mlp = attention + gate/up/down of the MLP
    "attn": r"self_attn\.(q_proj|k_proj|v_proj|o_proj)",
    "attn_mlp": r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
}


def lora_target_regex(num_layers, last_k_layers, target="attn"):
    layer_ids = "|".join(str(i) for i in range(num_layers - last_k_layers, num_layers))
    if target not in LORA_TARGETS:
        raise ValueError(f"unknown lora.target {target!r}, expected one of {sorted(LORA_TARGETS)}")
    return rf".*layers\.({layer_ids})\.{LORA_TARGETS[target]}"


def build_network(model_name, num_layers, lora_r, lora_alpha, lora_dropout, lora_last_k_layers, head_config, lora_target="attn"):
    backbone = load_backbone(model_name, num_layers)
    if lora_last_k_layers == 0:  # frozen Qwen, only the head trains
        backbone.requires_grad_(False)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_regex(backbone.config.num_hidden_layers, lora_last_k_layers, lora_target),
        )
        backbone = get_peft_model(backbone, lora_config)
    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **head_config)
    return BEVNetwork(backbone, head)
