"""Serialization: questions -> tokens, plus the isolating attention mask and shared position ids.

Two formats:

  bev     exact bev-decider / bev-train format.  No new special tokens at all: the option
          delimiters are ordinary text ("<option>" -> ['<option', '>'], "</option>" ->
          ['</', 'option', '>']), and the option vector is read at the LAST CONTENT TOKEN of the
          option.  Prompt order is task hint, question, state (so every state token is read
          knowing what is asked), then the options, then "The answer is:".

  special v2 format.  Four added special tokens <state> <q> <opt> <opt_end>; the option vector is
          read either at the <opt_end> token ("opt_end" pooling) or as the mean over the option's
          tokens ("mean" pooling).

Everything below is plain integer bookkeeping.  The 4-D additive mask and the shared position ids
are what make the model option-order invariant: every option starts at the same position id and no
option token may attend to another option.
"""

import json

import torch

TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}
NUM_TASK_TYPES = len(TASK_TYPES)

SPECIAL_TOKENS = ["<state>", "<q>", "<opt>", "<opt_end>"]

TASK_PROMPTS = {
    "choice": "Choose the best option.",
    "score": "Rate it on the given scale.",
    "noul": "Answer yes or no.",
}
ANSWER_PROMPT = "The answer is:"

POOL_BEV = "bev"        # last content token of the option (bev-decider); requires mode "bev"
POOL_OPT_END = "opt_end"  # the <opt_end> token (requires mode "special")
POOL_MEAN = "mean"        # mean over the option's own tokens (requires mode "special")


# --------------------------------------------------------------------------- tokenizer plumbing
def extend_tokenizer(tokenizer):
    """Add the four special tokens if they are missing. Returns the number of tokens added."""
    missing = [t for t in SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return len(missing)


def resize_and_init_embeddings(model, tokenizer, verbose=True):
    """Resize the embedding matrix and init the new rows from the mean of the existing ones."""
    before = model.get_input_embeddings().weight.shape[0]
    target = max(before, len(tokenizer))
    if target == before:
        if verbose:
            print(f"embeddings: already {before} rows, nothing to do")
        return 0
    old = model.get_input_embeddings().weight.data
    mean = old[:before].mean(dim=0)
    model.resize_token_embeddings(target)
    with torch.no_grad():
        w = model.get_input_embeddings().weight.data
        w[before:target] = mean
        if getattr(model.config, "tie_word_embeddings", False):
            out = model.get_output_embeddings()
            if out is not None and out.weight.data.shape[0] == target:
                out.weight.data[before:target] = mean
    if verbose:
        print(f"embeddings: {before} -> {target} rows, new rows initialised from the embedding mean")
    return target - before


# --------------------------------------------------------------------------- question -> options
def as_text(x):
    """States / instructions / criteria may be strings or JSON values; JSON renders with indent=2."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=2)


def question_to_options(question, noul_use_criteria=False):
    """Jev-format question dict -> (option_texts, keys, label, ranks).

    `ranks[i]` is the ordinal position of option i on its scale, in [0, 1], and is 0 for choice and
    noul.  It is a property of the LEVEL, never of the slot: bev already writes the level index
    into the option text ("2: frustrated") precisely because options share position ids, so the
    ordinal information has to travel with the content.  Keeping ranks separate lets the head use
    it without having to re-parse the text, and it survives option shuffling.

    bev-decider's training code (bev-train/dataset.py) always emits the bare ["No", "Yes"] pair for
    noul, while its inference encoder (bev-decider/encode.py) emits "No: <desc>" / "Yes: <desc>"
    when criteria are present.  17,725 of the 208,677 noul questions in bev-decision('all') carry
    criteria, so that is a real train/serve skew.  Default to the training behaviour; the
    inference behaviour is available with noul_use_criteria=True.
    """
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        if not criteria:
            raise ValueError("a choice question needs criteria: {option_key: description}")
        keys = list(criteria)
        options = [f"{k}: {as_text(criteria[k])}" if criteria[k] else k for k in keys]
        label = keys.index(question["label"]) if question.get("label") is not None else None
        ranks = [0.0] * len(keys)
    elif kind == "score":
        if not criteria:
            raise ValueError("a score question needs criteria: a list of level descriptions")
        keys = [str(i) for i in range(len(criteria))]
        options = [f"{i}: {as_text(lv)}" for i, lv in enumerate(criteria)]
        label = question.get("label")
        K = len(criteria)
        ranks = [i / max(K - 1, 1) for i in range(K)]
    elif kind == "noul":
        keys = ["false", "true"]
        if criteria and noul_use_criteria:
            options = [f"No: {as_text(criteria['false'])}", f"Yes: {as_text(criteria['true'])}"]
        else:
            options = ["No", "Yes"]
        label = None if question.get("label") is None else int(question["label"])
        ranks = [0.0, 0.0]
    else:
        raise ValueError(f"unknown question type {kind!r}; expected choice, noul or score")
    return options, keys, label, ranks


# --------------------------------------------------------------------------- encoder
class Encoder:
    """Encodes (state, question) into token ids, shared position ids and an isolating 4-D mask.

    pool:
      "bev"     -> read index is the last content token of the option (bev-decider behaviour)
      "opt_end" -> read index is the <opt_end> token
      "mean"    -> read index is unused; the model averages the option's tokens (opt_segment)
    """

    def __init__(self, tokenizer, max_state_tokens=1024, max_choice_tokens=64,
                 mode="bev", pool=POOL_BEV, noul_use_criteria=False):
        self.tok = tokenizer
        self.max_state_tokens = max_state_tokens
        self.max_choice_tokens = max_choice_tokens
        self.mode = mode
        self.pool = pool
        self.noul_use_criteria = noul_use_criteria
        if mode == "special":
            extend_tokenizer(tokenizer)
        self.opt_begin = tokenizer.encode("<option>") if mode == "bev" else tokenizer.convert_tokens_to_ids("<opt>")
        self.opt_end = tokenizer.encode("</option>") if mode == "bev" else tokenizer.convert_tokens_to_ids("<opt_end>")
        self.answer = tokenizer.encode(ANSWER_PROMPT)
        if not isinstance(self.opt_begin, list):
            self.opt_begin = [self.opt_begin]
        if not isinstance(self.opt_end, list):
            self.opt_end = [self.opt_end]
        assert (pool == POOL_BEV) == (mode == "bev"), "pool 'bev' requires mode 'bev'; 'opt_end'/'mean' require 'special'"

    def truncate(self, text, max_tokens):
        ids = self.tok.encode(text)
        return text if len(ids) <= max_tokens else self.tok.decode(ids[:max_tokens])

    def prompt_ids(self, state, instructions, task_type):
        state = self.truncate(as_text(state), self.max_state_tokens)
        if self.mode == "bev":
            prompt = f"{TASK_PROMPTS[task_type]}\nQuestion: {as_text(instructions)}\n\n{state}\n"
            return self.tok.encode(prompt)
        sid = self.tok.convert_tokens_to_ids("<state>")
        qid = self.tok.convert_tokens_to_ids("<q>")
        text = f"\n<state>{state}</state>\n<q type=\"{task_type}\">{as_text(instructions)}</q>\n"
        return [sid] + self.tok.encode(text[1:]) + [qid]

    def encode(self, state, instructions, options, task_type, ranks=None):
        """-> dict of python lists / ints (no tensors: the collate builds everything vectorised)."""
        prefix = self.prompt_ids(state, instructions, task_type)
        P = len(prefix)
        ids = list(prefix)
        pos = list(range(P))
        spans = []          # (begin, end) inclusive-exclusive, in the packed sequence
        segs = []           # option token ids per option, for mean pooling
        for opt in options:
            body = self.truncate(opt, self.max_choice_tokens)
            toks = self.opt_begin + self.tok.encode(body)
            read = len(toks) + len(self.opt_end) - 1 if self.pool == POOL_OPT_END else len(toks) - 1
            spans.append((len(ids), len(ids) + len(toks) + len(self.opt_end), read))
            segs.append(list(range(len(ids), len(ids) + len(toks))))
            ids += toks + self.opt_end
            pos += list(range(P, P + len(toks) + len(self.opt_end)))
        ans_start = len(ids)
        ids += self.answer
        # The answer tokens start after the LONGEST option, so every option token is in their past.
        # max(pos) therefore depends only on the option LENGTHS, not on their order.
        pos += list(range(max(pos) + 1, max(pos) + 1 + len(self.answer)))
        L = len(ids)

        # Block ids drive the vectorised mask in collate(): 0 = prefix, 1..K = options, K+1 = answer.
        return {
            "input_ids": ids,
            "position_ids": pos,
            "prefix_len": P,
            "answer_start": ans_start,
            "L": L,
            "n_options": len(spans),
            "spans": spans,                      # (begin, end, read_offset_within_option)
            "option_token_segments": segs,       # for mean pooling
            "option_ranks": list(ranks) if ranks is not None else [0.0] * len(spans),
            "answer_end_idx": L - 1,
            "task_type": TASK_TYPES[task_type],
        }

    def encode_question(self, state, question):
        options, keys, _, ranks = question_to_options(question, self.noul_use_criteria)
        ex = self.encode(state, question["instructions"], options, question["type"], ranks=ranks)
        ex["keys"] = keys
        return ex


# --------------------------------------------------------------------------- vectorised collate
def build_batch(examples, pad_id, option_seg_value=-1):
    """Pack a list of Encoder.encode() dicts into tensors.

    The [B, 1, L, L] additive mask is built with broadcasting only -- no Python loop over blocks:

      block[t] = 0 for prefix tokens, i+1 for option i, n_options+1 for answer tokens, PAD for
      padding.  A query row q may attend to key k iff
        (block[q] == block[k] and k <= q)              within-block causal
     or (k < P and block[q] > 0)                       any option / answer token sees the prefix
     or (k <= q and block[q] == n_options + 1)          the answer sees every option too
      and, for padding rows, only itself.

    Shared position ids come straight from the encoder: every option starts at position P.
    """
    B = len(examples)
    L = max(ex["L"] for ex in examples)
    C = max(max(ex["n_options"], 1) for ex in examples)

    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    prefix_len = torch.zeros(B, dtype=torch.long)
    answer_block = torch.zeros(B, dtype=torch.long)
    valid_len = torch.zeros(B, dtype=torch.long)
    read_idx = torch.zeros((B, C), dtype=torch.long)
    option_mask = torch.zeros((B, C), dtype=torch.bool)
    ranks = torch.zeros((B, C), dtype=torch.float)
    option_seg = torch.full((B, L), option_seg_value, dtype=torch.long)
    answer_end = torch.zeros(B, dtype=torch.long)
    task_type = torch.zeros(B, dtype=torch.long)

    block = torch.full((B, L), -1, dtype=torch.long)   # -1 == padding

    for b, ex in enumerate(examples):
        n, P, K = ex["L"], ex["prefix_len"], ex["n_options"]
        input_ids[b, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        position_ids[b, :n] = torch.tensor(ex["position_ids"], dtype=torch.long)
        block[b, :P] = 0
        ex_ranks = ex.get("option_ranks") or [0.0] * K
        for i, (s, e, ro) in enumerate(ex["spans"]):
            block[b, s:e] = i + 1
            read_idx[b, i] = s + ro
            option_mask[b, i] = True
            ranks[b, i] = float(ex_ranks[i]) if i < len(ex_ranks) else 0.0
        block[b, ex["answer_start"]:n] = K + 1
        answer_end[b] = ex["answer_end_idx"]
        prefix_len[b] = P
        answer_block[b] = K + 1
        valid_len[b] = n
        task_type[b] = ex["task_type"]
        for i, seg in enumerate(ex["option_token_segments"]):
            option_seg[b, torch.tensor(seg, dtype=torch.long)] = i

    q = torch.arange(L).view(1, L, 1)
    k = torch.arange(L).view(1, 1, L)
    bq = block.unsqueeze(2)          # B, L, 1
    bk = block.unsqueeze(1)          # B, 1, L
    within = (bq == bk) & (k <= q)
    to_prefix = (k < prefix_len.view(B, 1, 1)) & (bq > 0)
    answer_sees_all = (bq == answer_block.view(B, 1, 1)) & (k <= q)

    may = within | to_prefix | answer_sees_all                       # B, L, L
    may &= (k < valid_len.view(B, 1, 1))                            # never attend to padding
    pad_query = q >= valid_len.view(B, 1, 1)                        # padded rows: self only
    may = torch.where(pad_query, k == q, may)
    # A padding row must still see its own key, but its key index is itself; guard the empty case.
    may = may | (pad_query & (k == q) & (k < L))

    attention_mask = torch.zeros(B, 1, L, L, dtype=torch.float32)
    attention_mask.masked_fill_(~may.unsqueeze(1), -float("inf"))

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "option_read_idx": read_idx,
        "option_mask": option_mask,
        "option_seg": option_seg,
        "answer_end_idx": answer_end,
        "task_type": task_type,
        "token_valid": torch.arange(L).view(1, L) < valid_len.view(B, 1),
        "prefix_len": prefix_len,
        "ranks": ranks,
    }


def pad_ranks(option_mask, ranks):
    """Zero the rank of padding slots.  Choice and noul already carry rank 0 for every option, which
    is what keeps the rank embedding from becoming a positional feature."""
    return ranks * option_mask.float()


def segment_mean(hidden, option_seg, option_mask, option_seg_value=-1):
    """Mean of `hidden` [B, L, H] over each option's tokens -> [B, C, H]."""
    B, L, H = hidden.shape
    C = option_mask.shape[1]
    valid = (option_seg != option_seg_value).unsqueeze(-1).float()          # B, L, 1
    seg = option_seg.clamp(min=0)
    sums = hidden.new_zeros(B, C, H)
    sums.scatter_add_(1, seg.unsqueeze(-1).expand(B, L, H), hidden * valid)
    counts = hidden.new_zeros(B, C, 1)
    counts.scatter_add_(1, seg.unsqueeze(-1), valid)
    return sums / counts.clamp(min=1.0)