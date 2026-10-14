"""Deterministic DERIVED pre-pass (Tier-2 flag `--use_derived 1`).

Appends machine-checked facts to the state before tokenisation so the model is not asked to do
date arithmetic or unit conversion.  Pure stdlib + `dateparser`/`pint` when available; if either is
missing the corresponding extractor is skipped and nothing else changes.

Only comparisons that are actually present in the text (or implied by the question) are emitted --
see ARCH_V2.md Tier 2.  No synthetic training data is generated here.
"""

import re
from datetime import datetime

try:
    from dateparser import parse as _date_parse
except Exception:  # pragma: no cover
    _date_parse = None

try:
    import pint as _pint
    _UREG = _pint.UnitRegistry()
except Exception:  # pragma: no cover
    _pint, _UREG = None, None

DERIVED_OPEN = "[DERIVED]"
DERIVED_CLOSE = "[/DERIVED]"

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}

DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
    re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I),
    re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.I),
]
MONEY = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)")
QTY = re.compile(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*([a-zA-Z]{1,12})\b")


def extract_dates(text):
    out = []
    for pat_i, pat in enumerate(DATE_PATTERNS):
        for m in pat.finditer(text):
            try:
                if pat_i == 0:
                    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elif pat_i == 1:
                    mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                elif pat_i == 2:
                    mo, d, y = MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
                else:
                    mo, y, d = MONTHS[m.group(1).lower()], int(m.group(2)), 1
                datetime(y, mo, d)
                out.append((m.group(0), datetime(y, mo, d)))
            except ValueError:
                continue
    return out


def normalize_dates(text):
    """Emit each distinct date as ISO plus, when two dates exist, their signed day delta."""
    dates = extract_dates(text)
    if not dates:
        return []
    lines = []
    seen = set()
    for raw, dt in dates:
        key = dt.date()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"date_normalized: {raw} -> {dt.date().isoformat()}")
    if len(seen) >= 2:
        ordered = sorted(seen)
        a, b = ordered[0], ordered[-1]
        delta = (b - a).days
        lines.append(f"date_delta_days: {a.isoformat()} -> {b.isoformat()} = {delta} "
                     f"({'after' if delta > 0 else 'before' if delta < 0 else 'same day'})")
    return lines


def normalize_money(text):
    vals = [float(v.replace(",", "")) for v in MONEY.findall(text)]
    if not vals:
        return []
    lines = [f"money_values: {', '.join(f'{v:.2f}' for v in sorted(set(vals)))}"]
    if len(set(vals)) >= 2:
        lines.append(f"money_sum: {sum(vals):.2f}   money_max: {max(vals):.2f}")
    return lines


def normalize_units(text):
    if _UREG is None:
        return []
    out = []
    for num, unit in QTY.findall(text):
        try:
            q = float(num.replace(",", "")) * _UREG(unit.lower())
        except Exception:
            continue
        try:
            out.append(f"unit_normalized: {num} {unit} = {q.to_base_units():~P}")
        except Exception:
            continue
    return out[:20]


def derived_block(state, instructions=""):
    """-> the [DERIVED]...[/DERIVED] string, or "" when nothing was extracted."""
    lines = []
    for fn in (normalize_dates, normalize_money, normalize_units):
        try:
            lines.extend(fn(state))
        except Exception:
            continue
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    return "\n" + DERIVED_OPEN + "\n" + "\n".join(lines) + "\n" + DERIVED_CLOSE + "\n"


def apply_derived(state, instructions="", enabled=True):
    if not enabled:
        return state
    block = derived_block(state, instructions)
    return state + block if block else state