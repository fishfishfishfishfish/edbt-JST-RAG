"""MRAG 语义 + passage 级时间系数:双轴分区级联早停重排检索器。

在语义+时间混合重排(见
``tcrag.retrievers.metriever_semantic_temporal``)的基础上,把昂贵的
**全量语义重排**改造为 **T×C 双轴分区 → 对角层级遍历 → top-G 零新增
耐心门控早停** 的级联管道,早期版本的相邻层极值分离门控设计文档见
``docs/mrag_semantic_temporal_分阶段级联早停检索流程.md``。

管道:

    BM25 初始召回(默认 max(top_k*10, 1000) 候选,按 BM25 分降序)
      → 时间问题解析(years / time_relation_type / normalized_question)
      → 对全部候选一次性计算真实 temporal_coeff(正则+spline,CPU 毫秒级)
      → 双轴分区:
          T 轴:按真实 coeff 分桶 —— 最高桶 coeff≥τ_high、
                [τ_low,τ_high) 区间均匀划分为 t_bins 个等宽桶
                (``cascade_t_bins`` / ``MRAG_CASCADE_T_BINS``,
                默认 1,coeff 越高桶编号越小)、兜底桶 coeff<τ_low;
                总桶数 = t_bins+2,t_bins=1 时即固定 3 桶 T1/T2/T3
          C 轴:按 BM25 名次等分(桶数由 ``cascade_c_bins`` /
                ``MRAG_CASCADE_C_BINS`` 控制,默认 3 分 C1/C2/C3)
          每个候选落入 (t_bins+2)×c_bins 网格;
          无时间信息时网格退化为 1×c_bins
      → 按反对角线把格子组织成层级(空层折叠):
          默认 t_bins=1、c_bins=3 时沿用历史 6 层表:
          L1=T1∩C1
          L2=T1∩C2 ∪ T2∩C1
          L3=T2∩C2
          L4=T1∩C3 ∪ T3∩C1
          L5=T2∩C3 ∪ T3∩C2
          L6=T3∩C3(兜底层,保证被访问)
          其余桶数组合按 s=t+c 通用反对角线分组,
          层数 = (t_bins+2)+c_bins-1
      → 滑动窗口逐层 hybrid 打分(rank_by_semantic + coeff 组合),
        首批 L1∪L2 合并一次重排调用,之后每次只打新一层
      → top-G 零新增耐心门控:维护已打分池的全局 top-G
        (``MRAG_CASCADE_GATE_TOPK``,默认 10,与返回切片 top_k 独立);
        新层无候选进入全局 top-G 记一次零新增,连续 M 层
        (``MRAG_CASCADE_PATIENCE``,默认 2)零新增即提前停止
      → 已打分池按 final 稳定降序取 top_k

关键性质:

- **打分公式不变**:``final = hybrid_base*semantic
  + (1-hybrid_base)*semantic*coeff``,年份提取/筛选/兜底口径完全复用
  标准 MRAG 叶子函数(``get_spline_function`` / ``get_temporal_coeffs``);
- **确定性**:分区依据 coeff 真实值与 BM25 名次,门控只看 top-G 成员
  是否变化,无随机采样;
- **尺度无关**:门控不依赖分数绝对值/量纲(logit 平移、coeff 加权、
  换模型均不影响判据),单层离群点由耐心 M 吸收;
- **最坏零损失**:走到最后一层兜底时全部候选均被打分,结果与全量
  baseline 一致;
- **降级**:问题无时间信息时网格退化为 1×c_bins;重排模型加载失败时
  自动回退到全量语义重排路径(等价 metriever_semantic_temporal);
  ``MRAG_CASCADE_EARLY_STOP=false`` 也可显式关闭级联走 baseline。

在 benchmark YAML 中切换:

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
    """双轴分区(T coeff × C BM25 名次)级联早停版语义时间混合重排器。

    构造参数继承自 :class:`MRAGRetriever`,新增:
      - ``cascade_early_stop``:级联早停总开关(关闭走全量 baseline 路径);
      - ``cascade_t_high`` / ``cascade_t_low``:T 轴固定阈值分区边界;
      - ``cascade_t_bins``:``[cascade_t_low, cascade_t_high)`` 区间的均匀
        分区数(默认 1)。T 轴总桶数 = ``t_bins + 2``:最高桶
        ``coeff ≥ t_high``、中间 ``t_bins`` 个等宽桶、兜底桶
        ``coeff < t_low``;取 1 时退化为固定 3 桶(T1/T2/T3);
      - ``cascade_c_bins``:C 轴(BM25 名次)分区桶数,默认 3,必须 ≥ 1;
      - ``cascade_gate_topk``:门控前沿深度 G(默认 10):维护已打分候选的
        全局 top-G,当某新层没有任何候选进入全局 top-G 时记一次零新增;
      - ``cascade_patience``:耐心层数 M(默认 2):连续 M 个层对全局 top-G
        零新增即早停。

    cross_encoder(如 ms-marco-MiniLM)输出的是可负的原始 logit。本变体
    不做归一化、不按正负剔除候选,``_score_passages`` 对全部候选直接套
    coeff 组合公式后入池,由 top-G 零新增门控决定何时停止后续层打分。
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
        # [t_low, t_high) 均匀划分为 t_bins 个等宽桶;
        # t_low == t_high 时宽度为 0,中间桶区间为空(实际无候选落入)
        self._t_bin_width = (
            cascade_t_high - cascade_t_low
        ) / cascade_t_bins
        self._c_bins = cascade_c_bins
        self._gate_topk = cascade_gate_topk
        self._gate_patience = cascade_patience

    # ───────────────────────────────────────────────────────────────────────
    # 共享叶子:coeff 计算与 hybrid 组合
    # ───────────────────────────────────────────────────────────────────────

    def _compute_passage_coeffs(
        self,
        passages: list[Passage],
        temporal_info: TemporalInfo,
    ) -> dict[str, float] | None:
        """对全部候选计算真实 temporal_coeff,返回 ``id → coeff`` 映射。

        与全量 baseline 的 ``_rank_passages_by_temporal`` 使用完全相同的
        spline / 年份提取 / 合规筛选 / 0.5 兜底口径。问题无时间信息、
        ``other`` 类型或关闭 ``hybrid_score`` 时返回 None(调用方据此
        把网格退化为 1×3,不做时间加权)。
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
        """hybrid 组合公式;``coeff=None``(无时间信息)时 final 即语义分。

        等价写法 ``final = s * w``,其中
        ``w = hybrid_base + (1-hybrid_base)*coeff``。调用方只传入
        **正语义分**候选(``s > 0``,已在 ``_score_passages`` 过滤),
        因此 coeff 越小惩罚越强,不存在负分上的反向缩放问题。
        """
        if coeff is None:
            return semantic_score
        return (
            self._hybrid_base * semantic_score
            + (1.0 - self._hybrid_base) * semantic_score * coeff
        )

    # ───────────────────────────────────────────────────────────────────────
    # 双轴分区与对角层级构造
    # ───────────────────────────────────────────────────────────────────────

    def _coeff_t_bin(self, coeff: float, t_total: int) -> int:
        """把单个候选的真实 coeff 映射到 T 轴桶编号。

        - ``coeff ≥ t_high`` → 0(最高桶);
        - ``coeff < t_low`` → ``t_total-1``(兜底桶);
        - 其余落入 ``[t_low, t_high)`` 内 ``t_bins`` 个等宽桶之一:
          ``t = 1 + floor((coeff-t_low)/width)``,coeff 越高编号越小,
          浮点误差用 epsilon + clamp 吸收(编号 ∈ [1, t_bins]);
          桶边界(如 t_low+j*width)按左闭右开口径归入 coeff 更高
          (编号更小)的相邻桶,epsilon 仅消除二进制浮点表示误差;
        - ``t_low == t_high`` 时中间区间为空,理论上不会走到中间分支,
          防御性归入第 1 桶(该桶无其他候选,不影响层级结果)。
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
        """把候选按 T×C 双轴分入 ``t_total``×``c_bins`` 网格,再组织为
        对角层级。

        - T 轴(``cascade_t_bins`` 构造参数 /
          ``MRAG_CASCADE_T_BINS`` 环境变量,默认 1):
          总桶数 ``t_total = t_bins + 2`` —— 最高桶 ``coeff ≥ t_high``
          (t=0)、``[t_low, t_high)`` 区间内 ``t_bins`` 个等宽桶
          (t=1..t_bins,coeff 越高编号越小)、兜底桶 ``coeff < t_low``
          (t=t_total-1);``t_bins=1`` 时退化为固定 3 桶 T1/T2/T3。
          ``coeff_map`` 为 None 时 T 轴退化为单行(网格 1×``c_bins``);
        - C 轴:候选保持 BM25 降序,按名次等分为 ``c_bins`` 桶
          (``cascade_c_bins`` 构造参数 / ``MRAG_CASCADE_C_BINS`` 环境变量);
        - 层级顺序按 T/C 编号之和(反对角线)从小到大,编号和相同的格子
          合并为一层;空层折叠剔除,仅返回非空层(每层保留 BM25 相对序)。

        特例:``t_bins == 1 且 c_bins == 3``(默认 3×3)时沿用历史
        6 层表(把反对角线和为 2 的中心格 (1,1) 提前于两角格
        (0,2)/(2,0) 单独成层),保证与既有实验结果完全可比;
        其余桶数走通用反对角线分组(层数 = t_total+c_bins-1)。
        """
        temporal_on = coeff_map is not None
        n = len(candidates)
        c_bins = self._c_bins
        t_total = self._t_bins + 2 if temporal_on else 1

        # C 轴等分边界:第 k 桶右边界(不含)为 ceil((k+1)*n/c_bins)
        c_ends = [
            ((k + 1) * n + c_bins - 1) // c_bins
            for k in range(c_bins - 1)
        ]

        # grid[t][c],t ∈ {0,..,t_total-1},c ∈ {0,..,c_bins-1}
        grid: list[list[list[Passage]]] = [
            [[] for _ in range(c_bins)] for _ in range(t_total)
        ]

        for idx, ctx in enumerate(candidates):
            # bisect_right:idx 落在第几个右边界之后,即所属桶编号
            c_idx = bisect.bisect_right(c_ends, idx)
            if temporal_on:
                t_idx = self._coeff_t_bin(coeff_map[ctx.id], t_total)
            else:
                t_idx = 0
            grid[t_idx][c_idx].append(ctx)

        if temporal_on and self._t_bins == 1 and c_bins == 3:
            # 历史 6 层表(3×3 专用):中心格 (1,1) 早于同对角线两角格
            layer_cells: list[list[tuple[int, int]]] = [
                [(0, 0)],                                 # L1: T1∩C1
                [(0, 1), (1, 0)],                         # L2
                [(1, 1)],                                 # L3
                [(0, 2), (2, 0)],                         # L4
                [(1, 2), (2, 1)],                         # L5
                [(2, 2)],                                 # L6: T3∩C3 兜底
            ]
        elif temporal_on:
            # 通用反对角线分组:s = t + c 相同的格子为一层
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
    # 层内打分
    # ───────────────────────────────────────────────────────────────────────

    def _score_passages(
        self,
        passages: list[Passage],
        normalized_question: str,
        coeff_map: dict[str, float] | None,
    ) -> list[Passage]:
        """对一组候选发起一次 ``rank_by_semantic`` 调用(组内合并 batch)。

        流程:
          1. 重排模型打分(原始语义分写回 ``ctx.score``);
          2. 全部候选保留入池,套 hybrid 组合公式(``coeff_map=None``
             时 final 即语义分),不按分数正负剔除;
          3. 按 final 降序返回。

        返回候选数与入参一致;是否继续打后续层由 top-G 零新增门控决定,
        与本函数无关。
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
        """该时刻已打分池的全局 top-``gate_k`` 快照 ``[(id, score), ...]``。

        池内候选不足 ``gate_k`` 时返回池内全部候选(降序);空池返回 []。
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
        """统计 ``kept``(新打分一层)有多少候选进入全池全局 top-``gate_k``。

        返回 ``(new_in_count, top_g_snapshot)``;池内候选不足 ``gate_k``
        时前沿取池内全部候选(新层任何候选都算新入,等价于数量门控自动
        放宽,避免小池被误判为零新增)。``kept`` 为空(空层)时新入数为 0,
        但仍返回当前 top-G 快照(与上一层相同,用于日志追踪前沿)。
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
        """层内 final 分的 (min, p5, p50, p95, max)。

        采用最近秩(nearest-rank)口径:升序排序后 p 分位取
        ``sorted[ceil(p*n)-1]``;纯 Python 实现,单层样本量 ≤ N,开销可忽略。
        n=1 时五个统计量相同。
        """
        scores = sorted(ctx.score for ctx in passages)
        n = len(scores)

        def rank(p: float) -> float:
            idx = min(n - 1, math.ceil(p * n) - 1)
            return scores[max(0, idx)]

        return scores[0], rank(0.05), rank(0.50), rank(0.95), scores[-1]

    # ───────────────────────────────────────────────────────────────────────
    # baseline 全量路径(级联关闭 / 模型不可用 / 兜底对照)
    # ───────────────────────────────────────────────────────────────────────

    def _retrieve_full(
        self,
        initial_candidates: list[Passage],
        normalized_question: str,
        temporal_info: TemporalInfo,
        top_k: int,
    ) -> list[Passage]:
        """全量 baseline:全部候选一次语义重排 → 全部保留 → 时间加权。

        与级联路径使用同一个 ``_score_passages``(全量保留 + hybrid
        组合 + 降序),保证两种路径打分口径完全一致。
        """
        coeff_map = self._compute_passage_coeffs(initial_candidates, temporal_info)
        ranked = self._score_passages(
            initial_candidates, normalized_question, coeff_map,
        )
        return ranked[:top_k]

    # ───────────────────────────────────────────────────────────────────────
    # 级联早停主管道
    # ───────────────────────────────────────────────────────────────────────

    def _retrieve_cascade(
        self,
        initial_candidates: list[Passage],
        normalized_question: str,
        temporal_info: TemporalInfo,
        top_k: int,
    ) -> list[Passage]:
        """T×C 双轴分区 → 对角层级滑动窗口 → top-G 零新增耐心门控 → top_k。

        门控(方案 A):维护已打分候选的全局 top-G(``cascade_gate_topk``),
        每打完一层统计该层进入全局 top-G 的候选数 ``new_in``;``new_in=0``
        时零新增计数 +1,否则清零;连续 M 层(``cascade_patience``)零新增
        即早停。L1 用于建立初始前沿,不参与零新增计数。

        判据只依赖「top-G 成员是否变化」,与分数尺度(logit 平移/coeff
        加权)无关,且对单层离群点由耐心层数 M 吸收;门控深度 G 与最终
        返回切片数 ``top_k`` 相互独立。
        """
        total = len(initial_candidates)
        gate_k = self._gate_topk
        patience = self._gate_patience

        # 全量 coeff 前置(纯 CPU,供 T 分区与层内打分共用)
        coeff_map = self._compute_passage_coeffs(
            initial_candidates, temporal_info,
        )

        layers = self._build_diagonal_layers(initial_candidates, coeff_map)
        layer_sizes = [len(layer) for layer in layers]

        pool: list[Passage] = []
        # 已访问层 → hybrid 后的候选(空列表表示该层打分后为空)
        kept_by_layer: dict[int, list[Passage]] = {}
        # 每层进入全局 top-G 的候选数 / 该层打完后的连续零新增 streak
        new_in_by_layer: dict[int, int] = {}
        streak_by_layer: dict[int, int] = {}
        # 每层打完后全局 top-G 前沿快照 [(id, final_score), ...](降序)
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
            """合并一层后计算 top-G 新入数并更新连续零新增 streak。

            ``snapshot_pool`` 用于首批 L1∪L2 共用一次打分的情形:记录
            L1 行时只让 L1 候选进入快照,语义上等价于「打完 L1 后」;
            门控状态(streak)仍以真实全池为准,establish 层不参与门控。
            """
            kept = kept_by_layer[idx]
            view_pool = snapshot_pool if snapshot_pool is not None else pool
            new_in, snapshot = self._count_layer_in_topk(
                view_pool, kept, gate_k,
            )
            new_in_by_layer[idx] = new_in
            topg_by_layer[idx] = snapshot
            if establish:
                # L1 建立前沿,streak 固定为 0,不参与门控
                streak = 0
            else:
                prev = streak_by_layer.get(idx - 1, 0)
                streak = 0 if new_in > 0 else prev + 1
            streak_by_layer[idx] = streak
            return new_in, streak

        # 退化情形:折叠后只有一个非空层,打完即返回
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

        # 首批:L1∪L2 合并为一次重排调用(保 batch 利用率),再按层身份拆开
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

        # L1 建立前沿(快照只含 L1,语义上是「打完 L1 后」的 top-G);
        # L2 起按真实全池累计耐心
        account(0, establish=True, snapshot_pool=front)
        _, streak = account(1)

        stopped_layer = -1  # -1 表示走到最后一层兜底
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

        # 已打分池稳定降序(同分保持打分批次内顺序),取 top_k
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
        # 每层两行:
        #   ① final 分分位数 + 门控信号(进入全局 top-G 的新入数 new_in、
        #      连续零新增 streak;streak >= patience 即门控命中点);
        #   ② 打完该层后全局 top-G 前沿快照(id=final 分,降序),
        #      直观看前沿成员何时停止变化。
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
    # 对外入口
    # ───────────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        time_relation: str = "",
        candidates: list[Passage] | None = None,
        top_k: int = 10,
    ) -> list[Passage]:
        """双轴分区级联早停检索管道(独立模式)。

        Steps:
            0. 初始 BM25 召回(复用父类,候选按 BM25 分降序)
            1. 时间信息预处理,取 normalized_question
            2. 全量计算真实 temporal_coeff
            3. T(coeff 阈值)×C(BM25 名次)双轴分区,构造对角层级
            4. 滑动窗口逐层 hybrid 打分,top-G 零新增耐心门控早停
            5. 已打分池按最终分降序取 top_k

        级联关闭或重排模型不可用时,回退全量 baseline 路径。
        """
        # Step 0: 初始检索
        initial_candidates = await self._initial_retrieval(query, candidates)
        if not initial_candidates:
            return []

        # Step 1: 时间归一化(与标准 MRAG Step4 同口径)
        temporal_info = self.parse_temporal_question(query, time_relation)
        normalized_question = temporal_info.normalized_question or query

        # 模型不可用时级联没有节省对象,直接走全量路径(模型失败时
        # rank_by_semantic 内部原样返回 BM25 序,行为与 baseline 一致)
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
        """NuggetIndex 集成接口,签名与父类一致,调度到级联早停管道。

        初始召回候选数与标准版一致(``max(top_k*10, 1000)``)。
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
    """NuggetIndex 集成工厂函数(双轴分区级联早停版)。

    签名与 :func:`tcrag.retrievers.metriever.create_mrag_retriever`
    兼容,可直接替换 benchmark YAML 中的 ``retriever_factory``。

    Args:
        store: NuggetStore 实例(nuggetindex 集成模式)。
        llm: 仅为与 benchmark 的 partial(llm=...) 调用约定兼容而保留;
            本管道无 LLM 环节,传入的实例不会被使用。

    生效的环境变量:
      - ``MRAG_BM25_INDEX_PATH``:Pyserini Lucene 索引路径
        (不设则用 store 后端 BM25);
      - ``MRAG_RERANKER_MODEL``:语义重排模型 HF 名称
        (默认 ``nvidia/NV-Embed-v2``);
      - ``MRAG_RERANKER_TYPE``:cross_encoder | bge | nv_embed | sfr | jina
        (默认 ``nv_embed``);
      - ``MRAG_HYBRID_BASE``:混合公式中语义分数的最低保留比例(默认 0.0);
      - ``MRAG_SNT_WITH_TITLE``:提取 passage 年份时是否附加标题(默认 true);
      - ``MRAG_CASCADE_EARLY_STOP``:级联早停总开关(默认 true;
        设为 false 时本变体等价于全量 semantic_temporal baseline);
      - ``MRAG_CASCADE_T_HIGH`` / ``MRAG_CASCADE_T_LOW``:
        T 轴固定阈值(默认 0.9 / 0.6);
      - ``MRAG_CASCADE_T_BINS``:``[t_low, t_high)`` 区间的均匀分区数
        (默认 1)。T 轴总桶数 = t_bins+2(最高桶 + t_bins 个等宽桶 +
        兜底桶),取 1 时退化为固定 3 桶 T1/T2/T3;仅在
        ``t_bins=1 且 c_bins=3`` 时沿用历史 6 层对角表,其他组合按
        反对角线自动分组,层数为 (t_bins+2)+c_bins-1;
      - ``MRAG_CASCADE_C_BINS``:C 轴(BM25 名次)分区桶数(默认 3;
        取 3 且 t_bins=1 时沿用历史 6 层对角表,其他值按反对角线
        自动分组,层数为 (t_bins+2)+c_bins-1);
      - ``MRAG_CASCADE_GATE_TOPK``:门控前沿深度 G(默认 10),与最终
        返回切片数 top_k 独立(后者经 nuggetindex 聚合冗余放大为
        top_k*3,不应用作门控深度);
      - ``MRAG_CASCADE_PATIENCE``:耐心层数 M(默认 2),连续 M 层无
        候选进入全局 top-G 即早停。

    语义分处理:重排后全部候选保留入池并按 coeff 组合(不做归一化)。
    """
    bm25_index = os.getenv("MRAG_BM25_INDEX_PATH")
    reranker_model = os.getenv("MRAG_RERANKER_MODEL", "nvidia/NV-Embed-v2")
    reranker_type = os.getenv("MRAG_RERANKER_TYPE", "nv_embed")
    hybrid_base = float(os.getenv("MRAG_HYBRID_BASE", "0.0"))
    snt_with_title = _env_flag("MRAG_SNT_WITH_TITLE", True)
    cascade_enabled = _env_flag("MRAG_CASCADE_EARLY_STOP", True)
    cascade_t_high = float(os.getenv("MRAG_CASCADE_T_HIGH", "0.9"))
    cascade_t_low = float(os.getenv("MRAG_CASCADE_T_LOW", "0.6"))
    cascade_t_bins = int(os.getenv("MRAG_CASCADE_T_BINS", "1"))
    cascade_c_bins = int(os.getenv("MRAG_CASCADE_C_BINS", "3"))
    cascade_gate_topk = int(os.getenv("MRAG_CASCADE_GATE_TOPK", "10"))
    cascade_patience = int(os.getenv("MRAG_CASCADE_PATIENCE", "2"))

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
