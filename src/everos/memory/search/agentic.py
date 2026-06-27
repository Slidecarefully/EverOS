"""Episode AGENTIC cluster-path orchestration — 1:1 with everalgo benchmark.

Implements the cluster main path from ``benchmarks/common/stages/search.py``
(``enable_cluster_retrieval=True``):

    fact-MaxSim (dense + sparse)
        -> ahybrid_retrieve  (hybrid_full)
        -> acluster_retrieve (cluster_scoped, base=hybrid_full)
        -> aagentic_retrieve (base=cluster_scoped, round2=hybrid_full)

Hyperparameters match benchmark ``config.py`` defaults and are frozen as
module-level constants — no env/TOML knobs at this layer.

id contract: candidates flowing through the pipeline carry ``id=memcell_id``
(fact.parent_id chain). The final shaping step remaps to ``id=episode_id``
via ``metadata["episode_id"]`` before calling ``shape_episode_from_candidate``.
"""

# 这个模块实现的是 SearchManager 在 ``method=AGENTIC`` 且检索对象为 episode
# 时真正调用的“集群主路径”。它不是单纯做一次向量检索，而是把事实级检索、
# episode 父级回捞、稀疏/稠密融合、cluster 扩展、LLM agentic 二轮检索和
# 最终 DTO shaping 串成一条流水线。
#
# 阅读顺序可以抓住两个贯穿全文件的约定：
#
# 1. 中间阶段使用 ``memcell_id`` 作为 Candidate.id。
#    这是因为 atomic_fact 的 ``metadata["parent_id"]`` 指向 memcell，
#    MaxSim 也是先从 fact 命中再按 memcell 聚合。
#
# 2. 真正返回给 API 的 episode id 会暂存在 ``metadata["episode_id"]``。
#    到最后 ``_shape_results`` 才把 Candidate.id 从 memcell_id 换回
#    LanceDB episode row id，再交给已有 shaper 生成 ``SearchEpisodeItem``。

from __future__ import annotations

import datetime as _dt
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from everalgo.rank.agentic import aagentic_retrieve
from everalgo.rank.cluster import acluster_retrieve
from everalgo.rank.hybrid import ahybrid_retrieve
from everalgo.rank.maxsim import amaxsim_retrieve
from everalgo.types import Candidate

from everos.component.utils.datetime import from_timestamp, to_timestamp_ms
from everos.infra.persistence.sqlite import cluster_repo
from everos.memory.search.callbacks import build_rerank_fn
from everos.memory.search.shaper import shape_episode_from_candidate

from .dto import SearchEpisodeItem

if TYPE_CHECKING:
    from everalgo.clustering import Cluster
    from everalgo.llm.protocols import LLMClient

    from everos.component.rerank import RerankProvider
    from everos.memory.search.recall.atomic_fact import AtomicFactRecaller
    from everos.memory.search.recall.episode import EpisodeRecaller

# ── Benchmark hyperparameters (config.py defaults) ──────────────────────────
# 下面这组常量把 benchmark 里的检索参数固定在模块层：
# dense/sparse 各自取多少候选、RRF 的 k 值、cluster 扩展规模、
# agentic 第一轮/第二轮规模、多查询数量等，都在这里一眼可见。
# 这样做的含义是：这个文件追求与 benchmark 路径 1:1 对齐，不把这些值暴露成
# 运行期配置项，避免线上路径和实验路径悄悄漂移。
_DENSE_CANDIDATES: int = 50
_SPARSE_CANDIDATES: int = 50
_HYBRID_RRF_K: int = 40
_CLUSTER_BASE_CANDIDATES: int = 100
_CLUSTER_TOP_K: int = 10
_ROUND1_TOP_N: int = 50
_ROUND1_RERANK_TOP_N: int = 10
_ROUND2_CAP: int = 40
_MULTI_QUERY_COUNT: int = 3
_REFINEMENT_STRATEGY: str = "multi_query"

# Child-pool sizing — mirrors SearchManager._MAXSIM_FACT_MULTIPLIER / _CAP.
# AGENTIC 路径虽然最终返回 episode，但召回的起点仍然是 atomic fact。
# fact 数量通常远多于 episode，所以这里按 ``k * multiplier`` 放大 child pool，
# 再用 cap 防止一次 ANN/BM25 扫描过大。后续 MaxSim 会把这些 fact 聚合回
# 父 memcell，因此 child pool 足够宽，才能避免过早漏掉相关 episode。
_FACT_CHILD_MULTIPLIER: int = 20
_FACT_CHILD_CAP: int = 2000

# Qwen3-Reranker task instruction for the search scene (benchmark
# ``config.reranker_instruction``). Steers the cross-encoder toward fact /
# entity / detail relevance rather than topical similarity.
# 这个 instruction 会被传给 cross-encoder rerank callback。
# 它明确告诉 reranker：这里看重的是“能直接回答问题的事实、实体、细节”，
# 而不是泛泛的主题相似度。这和 atomic fact 驱动的检索目标是一致的。
_RERANK_INSTRUCTION: str = (
    "Determine if the passage contains specific facts, entities "
    "(names, dates, locations), or details that directly answer the question."
)


async def search_episodes_agentic(
    query: str,
    *,
    owner_id: str,
    where: str,
    episode_recaller: EpisodeRecaller,
    atomic_fact_recaller: AtomicFactRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    reranker: RerankProvider,
    llm: LLMClient,
    top_k: int,
) -> list[SearchEpisodeItem]:
    """Episode AGENTIC search via cluster-scoped MaxSim — 1:1 with benchmark.

    Args:
        query: User search query.
        owner_id: Owner whose memories are searched.
        where: Pre-compiled LanceDB filter string (owner + any request filters).
        episode_recaller: Episode-table sparse + dense + fetch callbacks.
        atomic_fact_recaller: AtomicFact-table sparse + dense callbacks.
        embed_query_fn: Async ``(str) -> list[float]`` query embedder.
        reranker: Cross-encoder rerank provider.
        llm: LLM client for sufficiency check + multi-query generation.
        top_k: Maximum episodes to return (maps to ``top_n`` in aagentic_retrieve).

    Returns:
        Ranked list of at most ``top_k`` ``SearchEpisodeItem`` objects.
        Empty when no clusters exist or retrieval returns nothing.
    """

    # 整个函数先不直接执行检索，而是逐层定义一组异步闭包。
    # 这些闭包会被 everalgo 的高阶检索算子调用，形成如下依赖关系：
    #
    # ``_fact_dense`` / ``_fact_sparse``
    #   -> ``amaxsim_retrieve`` 生成 ``_dense`` / ``_sparse`` 的 episode 候选
    #   -> ``ahybrid_retrieve`` 融合为 ``hybrid_full``
    #   -> ``acluster_retrieve`` 基于 cluster 做范围扩展
    #   -> ``aagentic_retrieve`` 做 sufficiency 判断与第二轮 refinement
    #   -> ``_shape_results`` 把 memcell-keyed Candidate 还原成 episode DTO。
    #
    # 这种写法让 I/O 依赖都留在 everos 的 recaller 里，
    # everalgo 只拿到“如何召回/如何取父文档”的 callback。

    # 1. Fact-level child retrieve closures (dense + sparse via atomic_fact table).
    # 第一层只负责从 atomic_fact 表里找“子文档”。
    # dense 路径需要先把 query embed 成向量；sparse 路径直接用原始 query。
    # 两条路径返回的 Candidate 仍然是 fact 级别的命中，后面才通过 parent_fetch
    # 回到 episode/memcell 粒度。
    async def _fact_dense(q: str, k: int) -> list[Candidate]:
        # dense fact recall 的 k 表示父级候选目标规模，但真正扫描 fact 时要放大。
        # 如果只取 k 个 fact，多个 fact 可能都属于同一个 memcell，
        # MaxSim 聚合后会导致父 episode 覆盖面太窄。
        vec = await embed_query_fn(q)
        if not vec:
            # 没有向量就不能做 dense recall；直接返回空列表，让上层融合自然退化。
            return []
        child_limit = min(k * _FACT_CHILD_MULTIPLIER, _FACT_CHILD_CAP)
        return await atomic_fact_recaller.dense_recall(vec, where, limit=child_limit)

    async def _fact_sparse(q: str, k: int) -> list[Candidate]:
        # sparse fact recall 与 dense 使用相同的 child pool 放大策略。
        # 这样后续 hybrid 融合时，BM25 和向量路径在候选覆盖面上大体对齐。
        child_limit = min(k * _FACT_CHILD_MULTIPLIER, _FACT_CHILD_CAP)
        return await atomic_fact_recaller.sparse_recall(q, where, limit=child_limit)

    # 2. parent_fetch: maps memcell_ids -> Candidate(id=memcell_id) for the amaxsim
    #    score lookup. Stores the real LanceDB episode id in metadata["episode_id"]
    #    for final shaping.
    # MaxSim 的核心是“子文档命中后，按 parent 聚合分数”。
    # atomic_fact 的 parent 是 memcell_id，所以 amaxsim_retrieve 会把一批
    # memcell_id 交给 ``_parent_fetch``。这里再从 episode 表批量取回父 episode。
    async def _parent_fetch(memcell_ids: list[str]) -> list[Candidate]:
        ep_cands = await episode_recaller.fetch_by_parent_ids(memcell_ids, where)
        result: list[Candidate] = []
        for c in ep_cands:
            mc_id = c.metadata.get("parent_id")
            if not isinstance(mc_id, str):
                # 没有合法 parent_id 的 episode 无法参与 memcell-keyed 的 MaxSim 流程。
                continue
            result.append(
                Candidate(
                    # 中间 Candidate.id 必须换成 memcell_id。
                    # 这是为了让 amaxsim_retrieve 能把 fact.parent_id 聚合出来的分数
                    # 正确挂到父 Candidate 上。
                    id=mc_id,
                    score=0.0,
                    source=c.source,
                    # 真实 episode LanceDB id 不能丢，否则最后无法 shape。
                    # 同时把 metadata 转成 everalgo prompt 需要的字段形态。
                    metadata=_to_everalgo_doc_metadata(
                        {**c.metadata, "episode_id": c.id}
                    ),
                )
            )
        return result

    # 3. MaxSim RetrieveFns: fact vectors/BM25 -> max-pool by memcell -> candidates.
    # 这里把 fact-level recall 包装成 episode-level retrieve function。
    # 调用者看到的是 Candidate 列表，但它们的分数来自“最匹配的 atomic fact”，
    # 而不是 episode 自身的平均向量或全文 BM25。
    async def _dense(q: str, k: int) -> list[Candidate]:
        return await amaxsim_retrieve(
            q,
            child_retrieve=_fact_dense,
            parent_fetch=_parent_fetch,
            top_n=k,
            child_candidates=min(k * _FACT_CHILD_MULTIPLIER, _FACT_CHILD_CAP),
        )

    async def _sparse(q: str, k: int) -> list[Candidate]:
        return await amaxsim_retrieve(
            q,
            child_retrieve=_fact_sparse,
            parent_fetch=_parent_fetch,
            top_n=k,
            child_candidates=min(k * _FACT_CHILD_MULTIPLIER, _FACT_CHILD_CAP),
        )

    # 4. hybrid_full: RRF fusion of dense + sparse MaxSim.
    # dense/sparse 现在都已经是“fact -> memcell -> episode”的 MaxSim 结果。
    # ``hybrid_full`` 再用 RRF 融合两条路径：向量负责语义召回，BM25 负责关键词、
    # 名称、日期等精确匹配。这个函数后面会被复用两次：
    # 一次作为 cluster 检索的 base，一次作为 agentic 第二轮检索的 round2_retrieve。
    async def hybrid_full(q: str, k: int) -> list[Candidate]:
        return await ahybrid_retrieve(
            q,
            dense_retrieve=_dense,
            sparse_retrieve=_sparse,
            top_n=k,
            dense_candidates=_DENSE_CANDIDATES,
            sparse_candidates=_SPARSE_CANDIDATES,
            rrf_k=_HYBRID_RRF_K,
        )

    # 5. Load cluster snapshot + full-corpus all_docs (memcell-keyed).
    #    Reshape metadata to the everalgo doc contract so the sufficiency /
    #    multi-query LLM prompt (rendered by ``_format_docs``) sees the episode
    #    body and a ms-epoch date instead of the memcell id.
    # 走到这里才读取 cluster 快照和 owner 全量 episode 文档。
    # cluster 检索需要知道“哪些 memcell 属于同一语义簇”，也需要 all_docs 作为
    # cluster 成员展开时的文档池。注意 all_docs 也要转成 everalgo 能渲染 prompt
    # 的 metadata 形态，否则 LLM 看到的可能只是 id，而不是 episode 正文。
    clusters: list[Cluster] = await cluster_repo.list_for_owner(owner_id, "user_memory")
    raw_all_docs = await episode_recaller.fetch_all_for_owner(where)
    all_docs: list[Candidate] = [
        c.model_copy(update={"metadata": _to_everalgo_doc_metadata(c.metadata)})
        for c in raw_all_docs
    ]

    # 6. cluster_scoped: narrows hybrid_full to top-K cluster member expansions.
    # cluster_scoped 先用 hybrid_full 找一批高置信种子，再根据 cluster 结构扩展
    # 同簇文档，并限制到最相关的 cluster_top_k。这样第一轮不是在全库裸检索结果上
    # 直接交给 agentic，而是先把候选收束到“相关语义簇”附近。
    async def cluster_scoped(q: str, _k: int) -> list[Candidate]:
        # ``_k`` 被保留在签名里，是为了符合 everalgo RetrieveFn 协议；
        # 这里实际规模由 benchmark 常量控制，保证与 cluster 主路径一致。
        return await acluster_retrieve(
            q,
            base_retrieve=hybrid_full,
            base_candidates=_CLUSTER_BASE_CANDIDATES,
            clusters=clusters,
            all_docs=all_docs,
            cluster_top_k=_CLUSTER_TOP_K,
        )

    # 7. Cross-encoder rerank fn (2-arg RerankFn, no internal truncation).
    # agentic 第一轮候选出来后，需要 cross-encoder 对 query-document 做更精细的相关性判断。
    # ``build_rerank_fn`` 把 everos 的 reranker provider 包装成 everalgo 需要的二参函数，
    # 并指定从 metadata["episode"] 取文本；这个字段不能在 metadata bridge 中被破坏。
    rerank_fn = build_rerank_fn(
        reranker, text_field="episode", instruction=_RERANK_INSTRUCTION
    )

    # 8. aagentic_retrieve — benchmark cluster main path.
    # 这里是真正启动 agentic 检索：
    #
    # - 第一轮：用 cluster_scoped 取候选，并用 rerank_fn 缩小到高质量片段。
    # - sufficiency check：LLM 判断第一轮结果是否足够回答 query。
    # - 若不足：按 refinement_strategy 生成多查询，走 round2_retrieve=hybrid_full
    #   做第二轮补充召回。
    # - 最后：多路结果用 RRF 合并，并截断为 top_k。
    #
    # 返回值里的 decision 目前没有向外暴露，所以用 ``_decision`` 接住，
    # 但保留它可以方便未来记录“是否触发第二轮/为什么触发”的诊断信息。
    candidates, _decision = await aagentic_retrieve(
        query,
        base_retrieve=cluster_scoped,
        round2_retrieve=hybrid_full,
        llm=llm,
        rerank_fn=rerank_fn,
        round2_cap=_ROUND2_CAP,
        top_n=top_k,
        round1_top_n=_ROUND1_TOP_N,
        round1_rerank_top_n=_ROUND1_RERANK_TOP_N,
        refinement_strategy=_REFINEMENT_STRATEGY,
        multi_query_count=_MULTI_QUERY_COUNT,
        rrf_k=_HYBRID_RRF_K,
    )

    # 9. Shape: remap id from memcell_id -> episode_id, then build DTO.
    # 到这里，检索排序已经完成，但 Candidate 仍然遵守中间阶段的 memcell-keyed 约定。
    # API 不能返回 memcell_id，所以最后一步统一还原 id 和 timestamp，再交给通用 shaper。
    return _shape_results(candidates)


def _to_everalgo_doc_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Bridge recall metadata to the everalgo ``_format_docs`` doc contract.

    ``aagentic_retrieve`` renders Round-1 candidates into the sufficiency /
    multi-query LLM prompt via ``everalgo.rank.agentic._format_docs``, which
    reads the doc body from ``metadata["content"] | metadata["text"] | id`` and
    the date from a ms-epoch ``metadata["timestamp"]``. everos episode rows
    carry the body in ``episode`` (str) and the time in ``timestamp`` (datetime);
    without this bridge the prompt degrades to the memcell id as the body and a
    "N/A" date. ``episode`` is left untouched so the reranker and shaper -- both
    expecting a plain string -- keep working. ``_restore_shaper_metadata``
    reverts the timestamp before DTO shaping.
    """
    # 这个 helper 解决的是“同一份 metadata 要喂给三类消费者”的问题：
    #
    # 1. everalgo 的 LLM prompt formatter 需要 ``text`` 和毫秒时间戳。
    # 2. everos 的 reranker wrapper 仍然从 ``episode`` 字段取正文。
    # 3. 最后的 DTO shaper 要求 ``timestamp`` 是 datetime。
    #
    # 因此这里只做增量桥接：复制 metadata，补充 ``text``，
    # 临时把 datetime timestamp 转成 ms epoch，但不删除原本的 ``episode``。
    bridged = dict(metadata)
    episode = metadata.get("episode")
    if isinstance(episode, str):
        # 让 everalgo 的 ``_format_docs`` 在 prompt 中展示 episode 正文，
        # 而不是退化到 Candidate.id（也就是 memcell_id）。
        bridged["text"] = episode
    timestamp = metadata.get("timestamp")
    if isinstance(timestamp, _dt.datetime):
        # everalgo prompt 侧按毫秒时间戳渲染日期；这里做一次格式适配。
        bridged["timestamp"] = to_timestamp_ms(timestamp)
    return bridged


def _restore_shaper_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Revert the ms-epoch ``timestamp`` injected for everalgo back to datetime.

    ``shape_episode_from_candidate`` requires a ``datetime`` timestamp and drops
    the row otherwise; the agentic pipeline carried it as ms-epoch for the LLM
    prompt. The extra ``text`` key is ignored by the shaper and left in place.
    """
    # 这是 ``_to_everalgo_doc_metadata`` 的收口步骤。
    # agentic 检索过程中 timestamp 被改成了数字，方便 LLM prompt 渲染；
    # 但 SearchEpisodeItem 的 shaper 期望 datetime。若不还原，相关候选会被丢弃。
    timestamp = metadata.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        # 如果 metadata 没被桥接过，或者 timestamp 已经是 datetime，
        # 就保持原样，避免无意义复制。
        return metadata
    reverted = dict(metadata)
    reverted["timestamp"] = from_timestamp(timestamp)
    return reverted


def _shape_results(candidates: list[Candidate]) -> list[SearchEpisodeItem]:
    """Remap candidate id from memcell_id -> episode_id; build the DTO list."""
    # 前面所有检索算子为了配合 fact.parent_id 和 MaxSim 聚合，都把 Candidate.id
    # 维持成 memcell_id。这个函数是边界转换点：从内部检索 id 合约切回
    # API/DTO 需要的 episode row id。
    result: list[SearchEpisodeItem] = []
    for c in candidates:
        ep_id = c.metadata.get("episode_id")
        if not isinstance(ep_id, str):
            # 没有 episode_id 的候选无法映射到真实 episode 行。
            # 这种候选可能来自异常 metadata 或非 episode 文档，直接跳过比返回错 id 更安全。
            continue
        ep_cand = Candidate(
            id=ep_id,
            score=c.score,
            source=c.source,
            metadata=_restore_shaper_metadata(c.metadata),
        )
        item = shape_episode_from_candidate(ep_cand)
        if item is not None:
            result.append(item)
    return result
