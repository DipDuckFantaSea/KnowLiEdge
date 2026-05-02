from __future__ import annotations

from typing import List, Tuple


def restore_abstract_from_inverted_index(inv: object) -> str:
    """Restore OpenAlex abstract from ``abstract_inverted_index`` format.

    OpenAlex returns abstracts as an inverted index mapping word -> positions, e.g.:
    ``{"microwave": [0, 3], "circuit": [1], "design": [2]}``.

    Args:
        inv: The ``abstract_inverted_index`` payload from OpenAlex.

    Returns:
        Restored plain-text abstract, or empty string if unavailable/invalid.
    """
    if not isinstance(inv, dict) or not inv:
        return ""
    pairs: List[Tuple[int, str]] = []
    for w, pos in inv.items():
        if not isinstance(w, str) or not w.strip():
            continue
        if not isinstance(pos, list):
            continue
        for p in pos:
            try:
                i = int(p)
            except Exception:
                continue
            if i < 0:
                continue
            pairs.append((i, w))
    if not pairs:
        return ""
    pairs.sort(key=lambda x: x[0])
    max_i = pairs[-1][0]
    tokens: List[str] = [""] * (max_i + 1)
    for i, w in pairs:
        if 0 <= i < len(tokens) and not tokens[i]:
            tokens[i] = w
    return " ".join(t for t in tokens if t).strip()
