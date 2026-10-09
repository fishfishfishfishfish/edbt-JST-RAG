"""MRAG semantic + passage-level temporal coefficients: a dual-axis partitioned cascaded early-stopping reranker.

Building on semantic-temporal hybrid reranking (see
``tcrag.retrievers.metriever_semantic_temporal``), the expensive
**full semantic reranking** is transformed into a cascaded pipeline of
**T×C dual-axis partitioning → diagonal-level traversal → top-G zero-increment
patience-gated early stopping**. The design document for the earlier adjacent-level
extremum-separation gating version can be found at
``docs/mrag_semantic_temporal_phased-cascade-early-stopping-pipeline.md``.

Pipeline:

    BM25 initial recall (default max(top_k*10, 1000) candidates, in descending BM25 score order)
      → Temporal question parsing (years / time_relation_type / normalized_question)
      → Compute the true temporal_coeff for all candidates in one pass (regex+spline, CPU, millisecond-level)
      → Dual-axis partitioning:
          T axis: bucket by the true coeff -- the highest bucket coeff≥τ_high,
                the [τ_low,τ_high) interval is evenly split into t_bins equal-width buckets
                (``cascade_t_bins`` / ``CASCADE_T_BINS``,
                default 1; the higher the coeff, the smaller the bucket index), plus a fallback bucket coeff<τ_low;
                total number of buckets = t_bins+2; with t_bins=1 this is a fixed 3 buckets T1/T2/T3
          C axis: evenly split by BM25 rank (number of buckets controlled by ``cascade_c_bins`` /
                ``CASCADE_C_BINS``, default 3: C1/C2/C3)
          Each candidate falls into the (t_bins+2)×c_bins grid;
          the grid degenerates to 1×c_bins when there is no temporal information
      → Organize cells into levels along anti-diagonals (empty levels are collapsed):
          With the defaults t_bins=1 and c_bins=3 the historical 6-level table is used:
          L1=T1∩C1
          L2=T1∩C2 ∪ T2∩C1
          L3=T2∩C2
          L4=T1∩C3 ∪ T3∩C1
          L5=T2∩C3 ∪ T3∩C2
          L6=T3∩C3 (fallback level, guaranteed to be visited)
          For other bucket combinations, generic anti-diagonal grouping s=t+c is used,
          with (t_bins+2)+c_bins-1 levels
      → Hybrid scoring level by level with a sliding window (rank_by_semantic + coeff combination);
        the first batch merges L1∪L2 into a single reranking call; afterwards only each new level is scored
      → Top-G zero-increment patience gating: maintain the global top-G of the scored pool
        (``CASCADE_GATE_TOPK``, default 10, independent of the returned top_k slice);
        a new level with no candidate entering the global top-G counts as one zero-increment;
        M consecutive levels (``CASCADE_PATIENCE``, default 2) of zero-increment stop early
      → Take the scored pool in stable descending final-score order and return the top_k

Key properties:

- **Scoring formula unchanged**: ``final = hybrid_base*semantic
  + (1-hybrid_base)*semantic*coeff``; year extraction/filtering/fallback conventions fully reuse the
  standard MRAG leaf functions (``get_spline_function`` / ``get_temporal_coeffs``);
- **Determinism**: partitioning is based on the true coeff values and BM25 ranks; gating only watches whether the top-G membership
  changes, with no random sampling;
- **Scale-invariant**: gating does not depend on absolute score values/units (logit shifts, coeff weighting,
  or switching models do not affect the criterion); single-level outliers are absorbed by patience M;
- **Zero loss in the worst case**: when the last fallback level is reached all candidates have been scored, yielding results identical to the full
  baseline;
- **Degradation**: when the question has no temporal information the grid degenerates to 1×c_bins; when the reranker fails to load it
  automatically falls back to the full semantic reranking path (equivalent to metriever_semantic_temporal);
  ``CASCADE_EARLY_STOP=false`` also explicitly disables the cascade and runs the baseline.

Switching in the benchmark YAML:

.. code-block:: yaml

    systems:
      jstrag:
        retriever_factory:
          "tcrag.retrievers.jstretriever:create_jst_retriever"
"""
from __future__ import annotations

import bisect
import math
import os
from datetime import datetime
from typing import Any

from tcrag.logging_config import get_logger
from tcrag.retrievers.metriever import (
    MRAGRetriever,
    Passage,
    TemporalInfo,
    get_spline_function,
    get_temporal_coeffs,
)

logger = get_logger(__name__)


class JSTRetriever(MRAGRetriever):
    """Semantic-temporal hybrid reranker with cascaded early stopping via dual-axis partitioning
    (T coeff × C BM25 rank).

    Constructor parameters are inherited from :class:`MRAGRetriever`, with the following additions:
      - ``cascade_early_stop``: master switch for cascaded early stopping (the full baseline path is used when off);
      - ``cascade_t_high`` / ``cascade_t_low``: fixed threshold boundaries for T-axis partitioning;
      - ``cascade_t_bins``: number of evenly sized partitions over the
        ``[cascade_t_low, cascade_t_high)`` interval (default 1). Total T-axis
        buckets = ``t_bins + 2``: the highest bucket ``coeff ≥ t_high``,
        ``t_bins`` equal-width middle buckets, and the fallback bucket
        ``coeff < t_low``; when set to 1 it degenerates to a fixed 3 buckets (T1/T2/T3);
      - ``cascade_c_bins``: number of C-axis (BM25 rank) partition buckets, default 3, must be ≥ 1;
      - ``cascade_gate_topk``: gating frontier depth G (default 10): maintains the global
        top-G of scored candidates; a new level with no candidate entering the global top-G counts as one zero-increment;
      - ``cascade_patience``: patience level count M (default 2): M consecutive levels contributing
        zero increments to the global top-G trigger early stopping.

    The cross_encoder (e.g. ms-marco-MiniLM) outputs raw logits that may be negative.
    This variant performs no normalization and does not discard candidates by sign;
    ``_score_passages`` applies the coeff combination formula directly to all candidates
    before pooling, and the top-G zero-increment gating decides when to stop scoring
    subsequent levels.
    """

    def __init__(
        self,
        *,
        cascade_early_stop: bool = True,
        cascade_t_high: float = 0.9,
        cascade_t_low: float = 0.6,
        cascade_t_bins: int = 1,
        cascade_c_bins: int = 3,
        cascade_gate_topk: int = 10,
        cascade_patience: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= cascade_t_low <= cascade_t_high <= 1.0:
            raise ValueError(
                "Must satisfy 0 <= cascade_t_low <= cascade_t_high <= 1, got "
                f"low={cascade_t_low}, high={cascade_t_high}"
            )
        if cascade_t_bins < 1:
            raise ValueError(
                f"cascade_t_bins must be >= 1, got {cascade_t_bins}"
            )
        if cascade_c_bins < 1:
            raise ValueError(
                f"cascade_c_bins must be >= 1, got {cascade_c_bins}"
            )
        if cascade_gate_topk < 1:
            raise ValueError(
                f"cascade_gate_topk must be >= 1, got {cascade_gate_topk}"
            )
        if cascade_patience < 1:
            raise ValueError(
                f"cascade_patience must be >= 1, got {cascade_patience}"
            )
        self._cascade_enabled = cascade_early_stop
        self._t_high = cascade_t_high
        self._t_low = cascade_t_low
        self._t_bins = cascade_t_bins
        # Evenly split [t_low, t_high) into t_bins equal-width buckets;
        # when t_low == t_high the width is 0 and the middle-bucket intervals
        # are empty (no candidate actually falls into them)
        self._t_bin_width = (
            cascade_t_high - cascade_t_low
        ) / cascade_t_bins
        self._c_bins = cascade_c_bins
        self._gate_topk = cascade_gate_topk
        self._gate_patience = cascade_patience

    # ───────────────────────────────────────────────────────────────────────
    # Shared leaves: coeff computation and hybrid combination
    # ───────────────────────────────────────────────────────────────────────

    def _compute_passage_coeffs(
        self,
        passages: list[Passage],
        temporal_info: TemporalInfo,
    ) -> dict[str, float] | None:
        """Compute the true temporal_coeff for all candidates and return an ``id → coeff`` mapping.

        Uses exactly the same spline / year extraction / eligibility filtering /
        0.5 fallback conventions as the full baseline's
        ``_rank_passages_by_temporal``. Returns None when the question has no
        temporal information, is of the ``other`` type, or ``hybrid_score`` is
        off (the caller then degenerates the grid to 1×3 and skips temporal weighting).
        """
        years = temporal_info.years
        time_relation_type = temporal_info.time_relation_type

        if not (
            len(years) > 0
            and time_relation_type != "other"
            and self._hybrid_score
        ):
            return None

        spline = get_spline_function(
            time_relation_type, temporal_info.implicit_condition, years,
        )
        if self._snt_with_title:
            passage_tuples = [
                (ctx.id, ctx.title + " " + ctx.text, 0.0) for ctx in passages
            ]
        else:
            passage_tuples = [(ctx.id, ctx.text, 0.0) for ctx in passages]

        coeffs = get_temporal_coeffs(
            years,
            passage_tuples,
            time_relation_type,
            temporal_info.implicit_condition,
            spline,
        )
        return {ctx.id: coeff for ctx, coeff in zip(passages, coeffs)}

    def _combine_final_score(
        self, semantic_score: float, coeff: float | None,
    ) -> float:
        """Hybrid combination formula; when ``coeff=None`` (no temporal information) final is just the semantic score.

        Equivalent form ``final = s * w``, where
        ``w = hybrid_base + (1-hybrid_base)*coeff``. The caller only passes
        candidates with a **positive semantic score** (``s > 0``, already
        filtered in ``_score_passages``), so the smaller the coeff the stronger
        the penalty, with no inverse-scaling problem on negative scores.
        """
        if coeff is None:
            return semantic_score
        return (
            self._hybrid_base * semantic_score
            + (1.0 - self._hybrid_base) * semantic_score * coeff
        )

    # ───────────────────────────────────────────────────────────────────────
    # Dual-axis partitioning and diagonal level construction
    # ───────────────────────────────────────────────────────────────────────

    def _coeff_t_bin(self, coeff: float, t_total: int) -> int:
        """Map a single candidate's true coeff to a T-axis bucket index.

        - ``coeff ≥ t_high`` → 0 (highest bucket);
        - ``coeff < t_low`` → ``t_total-1`` (fallback bucket);
        - The rest fall into one of the ``t_bins`` equal-width buckets within
          ``[t_low, t_high)``: ``t = 1 + floor((coeff-t_low)/width)``; the
          higher the coeff, the smaller the index, and floating-point error is
          absorbed with epsilon + clamp (index ∈ [1, t_bins]);
          bucket boundaries (e.g. t_low+j*width) follow the left-closed/right-open
          convention and are assigned to the adjacent bucket with the higher
          coeff (smaller index); epsilon only eliminates binary floating-point representation error;
        - When ``t_low == t_high`` the middle interval is empty, so the middle
          branch should theoretically never be reached; defensively assign to
          bucket 1 (the bucket contains no other candidates and does not affect level results).
        """
        if coeff >= self._t_high:
            return 0
        if coeff < self._t_low:
            return t_total - 1
        if self._t_bin_width <= 0.0:
            return 1
        j = int((coeff - self._t_low) / self._t_bin_width + 1e-9)
        j = min(max(j, 0), self._t_bins - 1)
        return 1 + j

    def _build_diagonal_layers(
        self,
        candidates: list[Passage],
        coeff_map: dict[str, float] | None,
    ) -> list[list[Passage]]:
        """Partition candidates by the T×C dual axes into a ``t_total``×``c_bins`` grid, then organize them into
        diagonal levels.

        - T axis (``cascade_t_bins`` constructor parameter /
          ``CASCADE_T_BINS`` environment variable, default 1):
          total buckets ``t_total = t_bins + 2`` -- the highest bucket
          ``coeff ≥ t_high`` (t=0), ``t_bins`` equal-width buckets within the
          ``[t_low, t_high)`` interval (t=1..t_bins; the higher the coeff, the
          smaller the index), and the fallback bucket ``coeff < t_low``
          (t=t_total-1); with ``t_bins=1`` it degenerates to a fixed 3 buckets T1/T2/T3.
          When ``coeff_map`` is None the T axis degenerates to a single row (a 1×``c_bins`` grid);
        - C axis: candidates remain in descending BM25 order and are evenly
          split into ``c_bins`` buckets by rank (``cascade_c_bins`` constructor
          parameter / ``CASCADE_C_BINS`` environment variable);
        - Levels are ordered by the sum of the T/C indices (anti-diagonal) from
          small to large; cells with the same index sum are merged into one
          level; empty levels are collapsed and discarded, returning only
          non-empty levels (the relative BM25 order is preserved within each level).

        Special case: ``t_bins == 1 and c_bins == 3`` (the default 3×3) retains
        the historical 6-level table (the central cell (1,1), whose anti-diagonal
        sum is 2, forms its own level before the two corner cells
        (0,2)/(2,0)), ensuring full comparability with existing experimental results;
        other bucket counts use generic anti-diagonal grouping (number of levels = t_total+c_bins-1).
        """
        temporal_on = coeff_map is not None
        n = len(candidates)
        c_bins = self._c_bins
        t_total = self._t_bins + 2 if temporal_on else 1

        # C-axis equal-split boundaries: the exclusive right boundary of the
        # k-th bucket is ceil((k+1)*n/c_bins)
        c_ends = [
            ((k + 1) * n + c_bins - 1) // c_bins
            for k in range(c_bins - 1)
        ]

        # grid[t][c], t ∈ {0,..,t_total-1}, c ∈ {0,..,c_bins-1}
        grid: list[list[list[Passage]]] = [
            [[] for _ in range(c_bins)] for _ in range(t_total)
        ]

        for idx, ctx in enumerate(candidates):
            # bisect_right: after how many right boundaries idx falls, i.e. its bucket index
            c_idx = bisect.bisect_right(c_ends, idx)
            if temporal_on:
                t_idx = self._coeff_t_bin(coeff_map[ctx.id], t_total)
            else:
                t_idx = 0
            grid[t_idx][c_idx].append(ctx)

        if temporal_on and self._t_bins == 1 and c_bins == 3:
            # Historical 6-level table (3×3 specific): the central cell (1,1)
            # precedes the two corner cells on the same anti-diagonal
            layer_cells: list[list[tuple[int, int]]] = [
                [(0, 0)],                                 # L1: T1∩C1
                [(0, 1), (1, 0)],                         # L2
                [(1, 1)],                                 # L3
                [(0, 2), (2, 0)],                         # L4
                [(1, 2), (2, 1)],                         # L5
                [(2, 2)],                                 # L6: T3∩C3 fallback
            ]
        elif temporal_on:
            # Generic anti-diagonal grouping: cells with the same s = t + c form one level
            layer_cells = []
            for s in range(t_total + c_bins - 1):
                cells = [
                    (t, s - t)
                    for t in range(t_total)
                    if 0 <= s - t < c_bins
                ]
                layer_cells.append(cells)
        else:
            layer_cells = [[(0, c)] for c in range(c_bins)]

        layers: list[list[Passage]] = []
        for cells in layer_cells:
            layer: list[Passage] = []
            for t, c in cells:
                layer.extend(grid[t][c])
            if layer:
                layers.append(layer)
        return layers

    # ───────────────────────────────────────────────────────────────────────
    # Intra-level scoring
    # ───────────────────────────────────────────────────────────────────────

    def _score_passages(
        self,
        passages: list[Passage],
        normalized_question: str,
        coeff_map: dict[str, float] | None,
    ) -> list[Passage]:
        """Issue one ``rank_by_semantic`` call for a group of candidates (batched together within the group).

        Flow:
          1. The reranker scores them (the raw semantic score is written back to ``ctx.score``);
          2. All candidates are retained in the pool and the hybrid combination
             formula is applied (when ``coeff_map=None`` final is just the
             semantic score), without discarding any by score sign;
          3. Return in descending final-score order.

        The number of returned candidates matches the input; whether subsequent
        levels continue to be scored is decided by the top-G zero-increment gate
        and is unrelated to this function.
        """
        scored = self.rank_by_semantic(
            passages, normalized_question, normalized=True,
        )

        if coeff_map is not None:
            for ctx in scored:
                semantic_score = ctx.score
                ctx.score = self._combine_final_score(
                    semantic_score, coeff_map[ctx.id],
                )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    @staticmethod
    def _topg_snapshot(
        pool: list[Passage],
        gate_k: int,
    ) -> list[tuple[str, float]]:
        """Snapshot of the global top-``gate_k`` of the scored pool at this moment: ``[(id, score), ...]``.

        Returns all candidates in the pool (in descending order) when the pool
        contains fewer than ``gate_k`` candidates; returns [] for an empty pool.
        """
        ranked = sorted(pool, key=lambda x: x.score, reverse=True)
        cutoff = gate_k if len(pool) >= gate_k else len(pool)
        return [(ctx.id, ctx.score) for ctx in ranked[:cutoff]]

    @classmethod
    def _count_layer_in_topk(
        cls,
        pool: list[Passage],
        kept: list[Passage],
        gate_k: int,
    ) -> tuple[int, list[tuple[str, float]]]:
        """Count how many candidates of ``kept`` (a newly scored level) enter the global pool-wide top-``gate_k``.

        Returns ``(new_in_count, top_g_snapshot)``; when the pool contains fewer
        than ``gate_k`` candidates the frontier takes all pool candidates (any
        candidate of the new level counts as newly entered, equivalent to an
        automatically relaxed count gate, avoiding a small pool being misjudged
        as zero-increment). When ``kept`` is empty (an empty level) the new-entry
        count is 0, but the current top-G snapshot is still returned (identical
        to the previous level, used for logging frontier tracking).
        """
        if not pool:
            return 0, []
        snapshot = cls._topg_snapshot(pool, gate_k)
        top_ids = {cid for cid, _ in snapshot}
        new_in = sum(1 for ctx in kept if ctx.id in top_ids) if kept else 0
        return new_in, snapshot

    @staticmethod
    def _layer_score_quantiles(
        passages: list[Passage],
    ) -> tuple[float, float, float, float, float]:
        """(min, p5, p50, p95, max) of the intra-level final scores.

        Uses the nearest-rank convention: after ascending sorting, the p-th
        percentile is ``sorted[ceil(p*n)-1]``; a pure-Python implementation
        with at most N samples per level, so the cost is negligible.
        All five statistics are identical when n=1.
        """
        scores = sorted(ctx.score for ctx in passages)
        n = len(scores)

        def rank(p: float) -> float:
            idx = min(n - 1, math.ceil(p * n) - 1)
            return scores[max(0, idx)]

        return scores[0], rank(0.05), rank(0.50), rank(0.95), scores[-1]

    # ───────────────────────────────────────────────────────────────────────
    # Full baseline path (cascade off / model unavailable / fallback control)
    # ───────────────────────────────────────────────────────────────────────

    def _retrieve_full(
        self,
        initial_candidates: list[Passage],
        normalized_question: str,
        temporal_info: TemporalInfo,
        top_k: int,
    ) -> list[Passage]:
        """Full baseline: semantic reranking of all candidates at once → all retained → temporal weighting.

        Uses the same ``_score_passages`` as the cascade path (retain all +
        hybrid combination + descending order), ensuring the scoring conventions
        of the two paths are identical.
        """
        coeff_map = self._compute_passage_coeffs(initial_candidates, temporal_info)
        ranked = self._score_passages(
            initial_candidates, normalized_question, coeff_map,
        )
        return ranked[:top_k]

    # ───────────────────────────────────────────────────────────────────────
    # Cascaded early-stopping main pipeline
    # ───────────────────────────────────────────────────────────────────────

    def _retrieve_cascade(
        self,
        initial_candidates: list[Passage],
        normalized_question: str,
        temporal_info: TemporalInfo,
        top_k: int,
    ) -> list[Passage]:
        """T×C dual-axis partitioning → diagonal-level sliding window → top-G zero-increment patience gating → top_k.

        Gating (scheme A): maintain the global top-G (``cascade_gate_topk``) of
        scored candidates; after each level is scored, count ``new_in``, the
        number of candidates from that level entering the global top-G; when
        ``new_in=0`` the zero-increment count is increased by 1, otherwise
        reset; M consecutive levels (``cascade_patience``) of zero-increment
        trigger early stopping. L1 establishes the initial frontier and does not
        participate in the zero-increment count.

        The criterion depends only on "whether the top-G membership changes" and
        is independent of score scale (logit shifts / coeff weighting); single-level
        outliers are absorbed by the patience level count M; the gating depth G is
        independent of the final returned slice count ``top_k``.
        """
        total = len(initial_candidates)
        gate_k = self._gate_topk
        patience = self._gate_patience

        # Full coeff computation upfront (pure CPU; shared by T partitioning and intra-level scoring)
        coeff_map = self._compute_passage_coeffs(
            initial_candidates, temporal_info,
        )

        layers = self._build_diagonal_layers(initial_candidates, coeff_map)
        layer_sizes = [len(layer) for layer in layers]

        pool: list[Passage] = []
        # Visited levels → candidates after hybrid (an empty list means the level is empty after scoring)
        kept_by_layer: dict[int, list[Passage]] = {}
        # Per-level count of candidates entering the global top-G / consecutive zero-increment streak after that level is scored
        new_in_by_layer: dict[int, int] = {}
        streak_by_layer: dict[int, int] = {}
        # Global top-G frontier snapshot after each level is scored [(id, final_score), ...] (descending)
        topg_by_layer: dict[int, list[tuple[str, float]]] = {}

        def score_layer(idx: int) -> list[Passage]:
            kept = self._score_passages(
                layers[idx], normalized_question, coeff_map,
            )
            kept_by_layer[idx] = kept
            pool.extend(kept)
            return kept

        def account(
            idx: int,
            *,
            establish: bool = False,
            snapshot_pool: list[Passage] | None = None,
        ) -> tuple[int, int]:
            """Compute the top-G new-entry count after merging one level and update the consecutive zero-increment streak.

            ``snapshot_pool`` is used for the case where the first batch
            L1∪L2 shares one scoring call: when recording the L1 row, only L1
            candidates are allowed into the snapshot, which is semantically
            equivalent to "after L1 is scored"; the gating state (streak) still
            follows the real full pool, and the establish layer does not participate in gating.
            """
            kept = kept_by_layer[idx]
            view_pool = snapshot_pool if snapshot_pool is not None else pool
            new_in, snapshot = self._count_layer_in_topk(
                view_pool, kept, gate_k,
            )
            new_in_by_layer[idx] = new_in
            topg_by_layer[idx] = snapshot
            if establish:
                # L1 establishes the frontier; the streak is fixed at 0 and does not participate in gating
                streak = 0
            else:
                prev = streak_by_layer.get(idx - 1, 0)
                streak = 0 if new_in > 0 else prev + 1
            streak_by_layer[idx] = streak
            return new_in, streak

        # Degenerate case: only one non-empty level after collapsing; return once it is scored
        if len(layers) == 1:
            score_layer(0)
            account(0, establish=True)
            self._log_cascade_stats(
                total, layer_sizes, kept_by_layer,
                new_in_by_layer, streak_by_layer, topg_by_layer,
                gate_k=gate_k, patience=patience,
                model_scored=layer_sizes[0],
                stopped_layer=0, early_stopped=False,
                temporal_on=coeff_map is not None,
            )
            pool.sort(key=lambda x: x.score, reverse=True)
            return pool[:top_k]

        # First batch: merge L1∪L2 into one reranking call (to preserve batch utilization), then split back by level identity
        first_ids = {ctx.id for ctx in layers[0]}
        first_scored = self._score_passages(
            layers[0] + layers[1], normalized_question, coeff_map,
        )
        front = [ctx for ctx in first_scored if ctx.id in first_ids]
        back = [ctx for ctx in first_scored if ctx.id not in first_ids]
        kept_by_layer[0] = front
        kept_by_layer[1] = back
        pool.extend(first_scored)
        model_scored = layer_sizes[0] + layer_sizes[1]

        # L1 establishes the frontier (the snapshot contains only L1, semantically
        # the top-G "after L1 is scored"); from L2 on, patience accumulates
        # against the real full pool
        account(0, establish=True, snapshot_pool=front)
        _, streak = account(1)

        stopped_layer = -1  # -1 means the last fallback level was reached
        if streak >= patience:
            stopped_layer = 1
        else:
            i = 2
            while i < len(layers):
                score_layer(i)
                model_scored += layer_sizes[i]
                _, streak = account(i)
                if streak >= patience:
                    stopped_layer = i
                    break
                i += 1

        self._log_cascade_stats(
            total, layer_sizes, kept_by_layer,
            new_in_by_layer, streak_by_layer, topg_by_layer,
            gate_k=gate_k, patience=patience,
            model_scored=model_scored,
            stopped_layer=stopped_layer,
            early_stopped=stopped_layer >= 0,
            temporal_on=coeff_map is not None,
        )

        # Sort the scored pool in stable descending order (ties keep the within-batch order) and take top_k
        pool.sort(key=lambda x: x.score, reverse=True)
        return pool[:top_k]

    def _log_cascade_stats(
        self,
        total: int,
        layer_sizes: list[int],
        kept_by_layer: dict[int, list[Passage]],
        new_in_by_layer: dict[int, int],
        streak_by_layer: dict[int, int],
        topg_by_layer: dict[int, list[tuple[str, float]]],
        *,
        gate_k: int,
        patience: int,
        model_scored: int,
        stopped_layer: int,
        early_stopped: bool,
        temporal_on: bool,
    ) -> None:
        kept_total = sum(len(k) for k in kept_by_layer.values())
        ratio = model_scored / total if total else 0.0
        logger.debug(
            "级联早停%s: total=%d, layers=%s, 重排调用候选=%d(%.1f%%), "
            "入池候选=%d, 停止层级=%s, gate=top%d/patience%d, temporal_axis=%s",
            "命中" if early_stopped else "兜底",
            total, layer_sizes, model_scored, ratio * 100,
            kept_total,
            f"L{stopped_layer + 1}" if stopped_layer >= 0 else "<last>",
            gate_k, patience,
            ("on(%dx%d)" % (self._t_bins + 2, self._c_bins))
            if temporal_on
            else "off(1x%d)" % self._c_bins,
        )
        # Two lines per level:
        #   ① final-score quantiles + gating signals (new_in, the new-entry
        #      count into the global top-G, and streak, the consecutive
        #      zero-increment count; streak >= patience is the gate-hit point);
        #   ② global top-G frontier snapshot after that level is scored
        #      (id=final score, descending), making it easy to see when the
        #      frontier members stop changing.
        for idx, size in enumerate(layer_sizes):
            tag = f"L{idx + 1}"
            if idx not in kept_by_layer:
                logger.debug(f"  {tag} n={size:<4d} <未打分(早停)>")
                continue
            if not kept_by_layer[idx]:
                line = f"  {tag} n={size:<4d} kept=0  <打分后为空>"
                logger.debug(line)
            else:
                kept = kept_by_layer[idx]
                lo, p5, p50, p95, hi = self._layer_score_quantiles(kept)
                new_in = new_in_by_layer.get(idx, 0)
                streak = streak_by_layer.get(idx, 0)
                hit = " <门控命中>" if streak >= patience and idx > 0 else ""
                logger.debug(
                    f"  {tag} n={size:<4d} kept={len(kept):<4d} "
                    f"min={lo:.4f} p5={p5:.4f} p50={p50:.4f} "
                    f"p95={p95:.4f} max={hi:.4f}"
                    f"  |  top{gate_k}新入={new_in}, 零新增streak={streak}{hit}"
                )
            snapshot = topg_by_layer.get(idx)
            if snapshot is not None:
                members = " ".join(
                    f"{cid}={sc:.4f}" for cid, sc in snapshot
                )
                logger.debug(
                    f"    {tag}后 top{gate_k}({len(snapshot)}/{gate_k}): {members}"
                )

    # ───────────────────────────────────────────────────────────────────────
    # Public entry point
    # ───────────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        time_relation: str = "",
        candidates: list[Passage] | None = None,
        top_k: int = 10,
    ) -> list[Passage]:
        """Dual-axis partitioned cascaded early-stopping retrieval pipeline (standalone mode).

        Steps:
            0. Initial BM25 recall (reusing the parent class; candidates in descending BM25 score order)
            1. Preprocess temporal information and take normalized_question
            2. Compute the true temporal_coeff for all candidates
            3. Dual-axis partitioning by T (coeff threshold) × C (BM25 rank), constructing diagonal levels
            4. Sliding-window level-by-level hybrid scoring with top-G zero-increment patience-gated early stopping
            5. Sort the scored pool by final score in descending order and take top_k

        Falls back to the full baseline path when the cascade is off or the
        reranker is unavailable.
        """
        # Step 0: initial retrieval
        initial_candidates = await self._initial_retrieval(query, candidates)
        if not initial_candidates:
            return []

        # Step 1: temporal normalization (same convention as standard MRAG Step 4)
        temporal_info = self.parse_temporal_question(query, time_relation)
        normalized_question = temporal_info.normalized_question or query

        # When the model is unavailable there is nothing for the cascade to
        # save, so go directly through the full path (on model failure
        # rank_by_semantic returns the BM25 order as-is internally, behaving consistently with the baseline)
        use_cascade = self._cascade_enabled and self._load_reranker() is not None
        if not use_cascade:
            return self._retrieve_full(
                initial_candidates, normalized_question, temporal_info, top_k,
            )

        return self._retrieve_cascade(
            initial_candidates, normalized_question, temporal_info, top_k,
        )

    async def aretrieve(
        self,
        query: str,
        *,
        query_time: datetime | None = None,
        view: str = "active",
        top_k: int = 10,
        fusion: str = "rrf",
        filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """NuggetIndex integration interface; the signature matches the parent class and it dispatches to the cascaded early-stopping pipeline.

        The initial recall candidate count matches the standard version
        (``max(top_k*10, 1000)``).
        """
        candidates = await self._initial_retrieval(
            query, top_k=max(top_k * 10, 1000))
        if not candidates:
            return []

        ranked_passages = await self.retrieve(
            query, candidates=candidates, top_k=top_k,
        )
        return await self._to_retrieval_results(ranked_passages, top_k)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no")


def create_jst_retriever(
    store: Any, *, llm: Any | None = None,
) -> JSTRetriever:
    """NuggetIndex integration factory function (dual-axis partitioned cascaded early-stopping version).

    The signature is compatible with
    :func:`tcrag.retrievers.metriever.create_mrag_retriever` and can directly
    replace the ``retriever_factory`` in the benchmark YAML.

    Args:
        store: NuggetStore instance (nuggetindex integration mode).
        llm: Retained only for compatibility with the benchmark's
            partial(llm=...) calling convention; this pipeline has no LLM stage
            and any passed-in instance will not be used.

    Effective environment variables:
      - ``BM25_INDEX_PATH``: Path to the Pyserini Lucene index
        (the store backend BM25 is used when unset);
      - ``RERANKER_MODEL``: HF name of the semantic reranking model
        (default ``nvidia/NV-Embed-v2``);
      - ``RERANKER_TYPE``: cross_encoder | bge | nv_embed | sfr | jina
        (default ``nv_embed``);
      - ``HYBRID_BASE``: Minimum retained fraction of the semantic score in the hybrid formula (default 0.0);
      - ``SNT_WITH_TITLE``: Whether to attach the title when extracting passage years (default true);
      - ``CASCADE_EARLY_STOP``: Master switch for cascaded early stopping (default true;
        when false this variant is equivalent to the full semantic_temporal baseline);
      - ``CASCADE_T_HIGH`` / ``CASCADE_T_LOW``:
        Fixed T-axis thresholds (default 0.9 / 0.6);
      - ``CASCADE_T_BINS``: Number of evenly sized partitions over the
        ``[t_low, t_high)`` interval (default 1). Total T-axis buckets =
        t_bins+2 (highest bucket + t_bins equal-width buckets + fallback bucket);
        when 1 it degenerates to a fixed 3 buckets T1/T2/T3; the historical
        6-level diagonal table is retained only when ``t_bins=1 and c_bins=3``;
        other combinations are automatically grouped along anti-diagonals, with
        (t_bins+2)+c_bins-1 levels;
      - ``CASCADE_C_BINS``: Number of C-axis (BM25 rank) partition buckets (default 3;
        when 3 and t_bins=1 the historical 6-level diagonal table is retained;
        other values are automatically grouped along anti-diagonals, with
        (t_bins+2)+c_bins-1 levels);
      - ``CASCADE_GATE_TOPK``: Gating frontier depth G (default 10), independent
        of the final returned slice count top_k (the latter is amplified with
        redundancy to top_k*3 during nuggetindex aggregation and must not be
        used as the gating depth);
      - ``CASCADE_PATIENCE``: Patience level count M (default 2); early stopping
        triggers when M consecutive levels have no candidate entering the global top-G.

    Semantic score handling: after reranking all candidates are retained in the
    pool and combined by coeff (no normalization is performed).
    """
    bm25_index = os.getenv("BM25_INDEX_PATH")
    reranker_model = os.getenv("RERANKER_MODEL", "nvidia/NV-Embed-v2")
    reranker_type = os.getenv("RERANKER_TYPE", "nv_embed")
    hybrid_base = float(os.getenv("HYBRID_BASE", "0.0"))
    snt_with_title = _env_flag("SNT_WITH_TITLE", True)
    cascade_enabled = _env_flag("CASCADE_EARLY_STOP", True)
    cascade_t_high = float(os.getenv("CASCADE_T_HIGH", "0.9"))
    cascade_t_low = float(os.getenv("CASCADE_T_LOW", "0.6"))
    cascade_t_bins = int(os.getenv("CASCADE_T_BINS", "1"))
    cascade_c_bins = int(os.getenv("CASCADE_C_BINS", "3"))
    cascade_gate_topk = int(os.getenv("CASCADE_GATE_TOPK", "10"))
    cascade_patience = int(os.getenv("CASCADE_PATIENCE", "2"))

    logger.info(
        "Creating JSTRetriever (T coeff × C BM25 dual-axis partition "
        "cascaded early stopping, top-G zero-increment patience gating): bm25_index=%s, reranker=%s(%s), "
        "hybrid_base=%.2f, snt_with_title=%s, cascade=%s, T[low=%.2f, high=%.2f], "
        "t_bins=%d(total %d bins), c_bins=%d, gate_topk=%d, patience=%d",
        bm25_index or "<store-backend>", reranker_model or "<none>", reranker_type,
        hybrid_base, snt_with_title, cascade_enabled,
        cascade_t_low, cascade_t_high, cascade_t_bins, cascade_t_bins + 2,
        cascade_c_bins,
        cascade_gate_topk, cascade_patience,
    )

    return JSTRetriever(
        bm25_index_path=bm25_index,
        reranker_model_name=reranker_model,
        reranker_type=reranker_type,
        llm=llm,
        hybrid_base=hybrid_base,
        snt_with_title=snt_with_title,
        store=store,
        cascade_early_stop=cascade_enabled,
        cascade_t_high=cascade_t_high,
        cascade_t_low=cascade_t_low,
        cascade_t_bins=cascade_t_bins,
        cascade_c_bins=cascade_c_bins,
        cascade_gate_topk=cascade_gate_topk,
        cascade_patience=cascade_patience,
    )
