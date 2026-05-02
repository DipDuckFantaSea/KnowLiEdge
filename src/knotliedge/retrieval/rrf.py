from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RankedId:
    chunk_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class RrfMerged:
    chunk_id: str
    rrf_score: float
    fts_rank: Optional[int]
    vec_rank: Optional[int]
    fts_score: Optional[float]
    vec_score: Optional[float]


@dataclass(frozen=True)
class RrfMergedMulti:
    """RRF merge result over N ranked lists (same chunk may appear in multiple lists)."""

    chunk_id: str
    rrf_score: float
    ranks: Tuple[Optional[int], ...]
    scores: Tuple[Optional[float], ...]


def _rrf_add(*, k: int, rank: int) -> float:
    return 1.0 / (float(k) + float(rank))


def rrf_merge_rankings(
    rankings: Sequence[Sequence[RankedId]],
    *,
    rrf_k: int = 60,
    limit: int = 10,
) -> List[RrfMergedMulti]:
    """Reciprocal Rank Fusion over any number of ranked lists.

    Each ``RankedId`` list contributes ``1 / (k + rank)`` to the chunk's fused score.

    Args:
        rankings: Ordered retrievers (e.g. ``[fts, vec_kw1, vec_kw2, ...]``). Ranks start at 1.
        rrf_k: RRF constant ``k``.
        limit: Max chunks to return.

    Returns:
        Top chunks by fused ``rrf_score`` descending, with per-list rank/score tuples
        (``None`` where that list had no hit for the chunk).
    """

    if not rankings:
        return []
    n = len(rankings)
    k = max(1, int(rrf_k))
    rrf_scores: Dict[str, float] = {}
    ranks_acc: Dict[str, List[Optional[int]]] = {}
    scores_acc: Dict[str, List[Optional[float]]] = {}

    for li, items in enumerate(rankings):
        for item in items:
            cid = str(item.chunk_id)
            if not cid:
                continue
            r = max(1, int(item.rank))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_add(k=k, rank=r)
            if cid not in ranks_acc:
                ranks_acc[cid] = [None] * n
                scores_acc[cid] = [None] * n
            ranks_acc[cid][li] = r
            scores_acc[cid][li] = float(item.score)

    merged = [
        RrfMergedMulti(
            chunk_id=cid,
            rrf_score=float(s),
            ranks=tuple(ranks_acc[cid][i] for i in range(n)),
            scores=tuple(scores_acc[cid][i] for i in range(n)),
        )
        for cid, s in rrf_scores.items()
    ]
    merged.sort(key=lambda x: x.rrf_score, reverse=True)
    return merged[: max(1, int(limit))]


def rrf_merge(
    *,
    fts: Sequence[RankedId],
    vec: Sequence[RankedId],
    rrf_k: int = 60,
    limit: int = 10,
) -> List[RrfMerged]:
    """Reciprocal Rank Fusion (RRF) merge for two ranked lists.

    Args:
        fts: Ranked list from keyword search (rank starts from 1).
        vec: Ranked list from vector search (rank starts from 1).
        rrf_k: RRF constant k (larger reduces head dominance).
        limit: Return top-N merged results.

    Returns:
        Sorted merged list by rrf_score desc.
    """
    ms = rrf_merge_rankings([fts, vec], rrf_k=int(rrf_k), limit=int(limit))
    out: List[RrfMerged] = []
    for m in ms:
        fr = m.ranks[0] if len(m.ranks) > 0 else None
        vr = m.ranks[1] if len(m.ranks) > 1 else None
        fs = m.scores[0] if len(m.scores) > 0 else None
        vs = m.scores[1] if len(m.scores) > 1 else None
        out.append(
            RrfMerged(
                chunk_id=m.chunk_id,
                rrf_score=m.rrf_score,
                fts_rank=fr,
                vec_rank=vr,
                fts_score=fs,
                vec_score=vs,
            )
        )
    return out

