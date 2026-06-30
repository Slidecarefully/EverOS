"""Hierarchical episode retrieval — two-path recall fused with per-fact eviction.

Episode HYBRID search path: combines episode-level hybrid recall (Layer 1)
with fact-driven MaxSim re-scoring (Layer 2), merges via RRF (Layer 3), then
runs a single-pass eviction where a fact that outscores its parent episode
enters top-N in place of the episode (Layer 4).

Uses everalgo operators as pure algorithm primitives; all I/O is injected
via recaller callbacks.  No changes to the everalgo library are required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.rank import amaxsim_retrieve
from everalgo.rank.fusion import rrf
from everalgo.types import Candidate, FactCandidate, ScoredItem

from everos.core.observability.logging import get_logger

from .dto import SearchEpisodeItem
from .shaper import reshape_hybrid_output

# 这里的导入分成两类：
# 1. 运行期真正会用到的算法对象，例如 rrf、amaxsim_retrieve、Candidate。
# 2. 仅用于类型标注的 recaller 类型，放在 TYPE_CHECKING 里避免运行期循环依赖。
# 这个文件本身不直接碰存储层；它只规定“检索流程怎么编排”，具体 I/O 通过 recaller 注入。
if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.memory.search.recall.atomic_fact import AtomicFactRecaller
    from everos.memory.search.recall.episode import EpisodeRecaller

logger = get_logger(__name__)


async def hierarchy_retrieve_episodes(
    query: str,
    *,
    sparse: list[Candidate],
    dense: list[Candidate],
    query_vector: list[float],
    fact_recaller: AtomicFactRecaller,
    episode_recaller: EpisodeRecaller,
    where: str,
    top_k: int,
    fact_child_candidates: int = 200,
) -> list[SearchEpisodeItem]:
    """Run the four-layer hierarchical episode retrieval pipeline.

    Layer 1: RRF fusion over pre-recalled sparse + dense episode candidates.
    Layer 2: MaxSim re-score via atomic-fact child retrieval (fact cosine ANN
             → group by parent memcell → episode re-score by best fact).
    Layer 3: RRF merge of Layer-1 and Layer-2 results, sliced to top_k.
    Layer 4: Pre-fetch facts for merged episodes, then single-pass eviction
             (fact outscoring its parent episode enters top-N instead).

    Args:
        query: Raw query string passed to amaxsim_retrieve.
        sparse: BM25 episode candidates from the caller's recall phase.
        dense: Vector ANN episode candidates from the caller's recall phase.
        query_vector: Pre-computed query embedding; reused for fact ANN recall
            and per-fact scoring in facts_for_episodes.
        fact_recaller: AtomicFactRecaller instance for child retrieval and
            facts_for_episodes.
        episode_recaller: EpisodeRecaller instance for MaxSim parent fetch.
        where: LanceDB filter clause (owner scope, tenant, etc.).
        top_k: Maximum number of items in the final merged slice before eviction.
        fact_child_candidates: How many atomic-fact ANN candidates to pull in
            Layer 2. Default 200.

    Returns:
        Shaped SearchEpisodeItem list (episodes with nested atomic_facts),
        sorted by score descending.
    """
    # 入口函数接收的 sparse / dense 不是在这里查出来的，而是 SearchManager
    # 之前已经按同一个 where 条件召回好的 episode 候选。
    # 因此本函数的职责不是“从零搜索”，而是把两条 episode 路径和一条 fact 路径
    # 组合成最终 SearchEpisodeItem：先在 episode 粒度找粗召回，再用 atomic fact
    # 修正长 episode 被均值向量稀释的问题，最后把更相关的 fact 提到前面。

    # Layer 1 — episode RRF fusion
    # 第一层只处理 episode 级别的候选：
    # sparse 代表关键词/BM25 命中，dense 代表向量语义命中。
    # RRF 的价值在于不强依赖两路分数是否同标尺，而是按各自排序位置做融合。
    # 这里得到的是“传统 HYBRID episode 召回”的基础结果。
    layer1_episodes = rrf(sparse, dense)

    # Layer 2 — MaxSim re-score via atomic-fact child retrieval
    # 第二层换一个视角：不直接拿 episode 向量匹配 query，而是先搜更细粒度的 atomic facts。
    # 如果某个 fact 很匹配 query，就把它的 parent memcell 对应的 episode 拉回来，
    # 并用“该 episode 下最匹配 fact 的分数”重新给 episode 打分。
    # 这补足了长 episode 的缺陷：一个 episode 里可能只有一小段高度相关，
    # 但整体 episode embedding 会被其他内容稀释。
    layer2_episodes = await _maxsim_episode_rescore(
        query=query,
        query_vector=query_vector,
        fact_recaller=fact_recaller,
        episode_recaller=episode_recaller,
        where=where,
        child_candidates=fact_child_candidates,
    )

    # Layer 3 — RRF merge of episode-level results, slice to top_k
    # 第三层把“episode 级混合召回结果”和“fact 驱动回捞的 episode 结果”再融合一次。
    # 两者都已经是 episode Candidate，所以这里仍然可以用 RRF 做排序融合。
    # 切到 top_k 是为了把后续 facts_for_episodes 的批量取数范围控制住：
    # 先确定最有希望进入最终答案的 episode，再只为这些 episode 拉取子事实。
    merged = rrf(layer1_episodes, layer2_episodes)[:top_k]

    # 没有候选时提前返回，避免继续查 facts_for_episodes。
    # 这个日志记录的是“融合层没有任何 episode 可继续加工”，通常意味着上游 sparse/dense
    # 和 MaxSim 路径都没有有效命中，或 where 过滤过窄。
    if not merged:
        logger.info("hierarchy_retrieve_empty_merge", top_k=top_k)
        return []

# Layer 4 的目的不是扩大 episode 候选集，而是给已经进入 top_k 的 episode
# 找出最能解释 query 命中的 atomic fact。
#
# 注意：merged episode.score 已经是 RRF rank score，而 best_fact.score 是
# query-vector 与 fact-vector 的 cosine relevance score，二者不是完全同一标尺。
# 因此这个判断更像“如果存在足够强的 fact 证据，就让 fact 作为该 episode 的
# 命中证据进入 scored_items”，而不是严格比较 episode 与 fact 的同尺度相关性。

根据代码，Layer 4 比较的是：

best_fact.score     = facts_for_episodes 里根据 query_vector 算出的 cosine similarity score
episode.score       = Layer 3 RRF score

而 RRF score 通常很小，比如默认 k=60：

rank 1: 1 / 61 ≈ 0.0164
rank 2: 1 / 62 ≈ 0.0161
两路都 rank 1: ≈ 0.0328

但 fact cosine score 可能是：

0.4 / 0.6 / 0.8

所以：

best_fact.score > episode.score

在有 query_vector 且 facts 存在时，大概率会成立。

这意味着 Layer 4 的实际行为更像：

只要这个 top_k episode 下存在一个 query-relevant fact，就优先用 fact 作为命中证据。

#
# 最终 reshape_hybrid_output 仍会返回 SearchEpisodeItem；
# fact 不会顶层裸露，只会嵌套到 parent episode.atomic_facts 里。
    
    # Layer 4a — pre-fetch facts for merged episodes
    # 第四层开始进入“episode 与 fact 的替换关系”：
    # 先把 merged 中每个 episode 的 LanceDB id 映射到它所属的 memcell_id，
    # 因为 atomic_facts 表里的 parent_id 指向 memcell，而不是 episode row id。
    ep_to_memcell = _build_ep_to_memcell(merged)

    # 这里一次性把 merged episodes 的相关 facts 拉回来，而不是在 eviction 循环里逐个查。
    # per_episode 使用 max(top_k * 2, 20)，是为了给每个候选 episode 多取一些事实，
    # 让“最佳 fact 是否超过 parent episode”这个判断有足够样本。
    # query_vector 会被复用来计算 fact 与 query 的相似度，避免重复 embed query。
    episode_to_facts = await fact_recaller.facts_for_episodes(
        ep_to_memcell,
        where,
        per_episode=max(top_k * 2, 20),
        query_vector=query_vector,
    )

    # Layer 4b — single-pass eviction
    # eviction 不是重新排序全量 fact，而是在已经排序好的 merged episode 列表上顺序扫描：
    # 对每个 episode，只看它最相关的一个 fact。
    # 如果 best_fact 分数高过 parent episode，就让 fact 占据这个位置；
    # 否则 episode 自己保留位置。这样既能突出局部高相关事实，也不会让一个 episode
    # 下面的多个 facts 把结果页刷屏。
    scored_items = _hierarchy_eviction_pass(merged, episode_to_facts)

    # Build episode pool for orphan fact parent lookup.
    # Include layer2_episodes so episodes surfaced only via MaxSim path
    # (not in the original sparse/dense recall) can still serve as parent.
    # reshape_hybrid_output 负责把 ScoredItem 变成 SearchEpisodeItem。
    # 如果 eviction 后输出的是 atomic_fact，它需要找到 parent episode 来承载这个 fact：
    # 最终 API 仍然返回 episode item，只是 atomic_facts 字段里会带上胜出的 fact。
    # episode_pool 因此必须覆盖三类来源：
    # - sparse：关键词召回来的 episode；
    # - dense：向量召回来的 episode；
    # - layer2_episodes：只通过 atomic fact MaxSim 回捞出来的 episode。
    episode_pool = {c.id: c for c in (*sparse, *dense, *layer2_episodes)}

    # 到这里，算法层输出的是混合 ScoredItem；shaper 层再负责 DTO 形状。
    # 这个边界很重要：hierarchy 只管检索与排序语义，不关心 API 字段如何拼装。
    return reshape_hybrid_output(scored_items, episode_pool=episode_pool)


def _hierarchy_eviction_pass(
    merged: list[Candidate],
    episode_to_facts: dict[str, list[FactCandidate]],
) -> list[ScoredItem]:
    """Single-pass eviction: fact outscoring its parent episode enters top-N.

    For each episode in merged order: if its best matching atomic fact scores
    higher than the episode itself, emit the fact as a ScoredItem
    (item_type='atomic_fact') and mark the episode as an orphan parent.
    Otherwise emit the episode directly as item_type='episode'.

    Args:
        merged: RRF-merged episode candidates, ordered by descending score.
        episode_to_facts: Map from episode_id to its pre-fetched FactCandidates,
            sorted by cosine similarity descending.

    Returns:
        Mixed list of ScoredItem instances (episodes and atomic_facts) ready
        for reshape_hybrid_output.
    """
    # out 保存的是最终交给 shaper 的“扁平混合结果”。
    # 它不是最终 API 响应；它只表达每个位置应该由 episode 本身占据，
    # 还是由该 episode 下的某个 atomic_fact 占据。
    out: list[ScoredItem] = []

    # 这里保持 merged 的原始顺序逐个处理，避免 Layer 4 变成一次新的全局重排。
    # 也就是说，Layer 1-3 已经决定了 parent episode 的大致位置；
    # Layer 4 只允许该位置被这个 episode 自己最强的 child fact 替换。
    for episode in merged:
        # facts_for_episodes 约定每个 episode 对应的 facts 已按 query 相似度降序排列。
        # 所以这里只取 facts[0]，代表这个 parent 下最有资格挑战 parent score 的事实。
        facts = episode_to_facts.get(episode.id, [])
        best_fact = facts[0] if facts else None

        # 判断标准非常克制：只有 child fact 的分数严格高于 parent episode 分数，
        # fact 才能进入 out。相等时仍保留 episode，避免过度偏向碎片化事实。
        if best_fact is not None and best_fact.score > episode.score:
            # Fact wins: emit fact; episode becomes orphan parent
            # 这里输出 item_type="atomic_fact"，并把 parent_episode_id 写成当前 episode.id。
            # 后续 reshape_hybrid_output 会用 parent_episode_id 回到 episode_pool 中找父 episode，
            # 再把这个 fact 填到父 episode 的 atomic_facts 字段里。
            # “orphan parent”不是说父 episode 丢失，而是说父 episode 没有作为独立 ScoredItem
            # 出现在当前位置；它转而作为 fact 的承载容器存在。
            out.append(
                ScoredItem(
                    id=best_fact.id,
                    score=best_fact.score,
                    item_type="atomic_fact",
                    metadata=best_fact.metadata,
                    parent_episode_id=episode.id,
                )
            )

            # debug 日志记录一次替换决策，便于排查为什么最终结果里某个 fact 顶替了 episode。
            logger.debug(
                "hierarchy_eviction_fact_wins",
                episode_id=episode.id,
                fact_id=best_fact.id,
                fact_score=best_fact.score,
                episode_score=episode.score,
            )
        else:
            # Episode wins: emit episode with its metadata intact
            # 没有 fact，或 best_fact 不比 parent episode 更相关时，保留 episode 本身。
            # metadata 用 dict(...) 复制一份，避免后续 shaper 或调用链意外修改原 Candidate metadata。
            out.append(
                ScoredItem(
                    id=episode.id,
                    score=episode.score,
                    item_type="episode",
                    metadata=dict(episode.metadata),
                    parent_episode_id=None,
                )
            )

    return out


# ── Internal helpers ─────────────────────────────────────────────────────


async def _maxsim_episode_rescore(
    *,
    query: str,
    query_vector: list[float],
    fact_recaller: AtomicFactRecaller,
    episode_recaller: EpisodeRecaller,
    where: str,
    child_candidates: int,
) -> list[Candidate]:
    """Run amaxsim_retrieve to produce MaxSim-rescored episode candidates.

    Atomic facts serve as child documents (their metadata["parent_id"] is
    the memcell_id). Episodes are fetched as parents via
    episode_recaller.fetch_by_parent_ids.

    ``amaxsim_retrieve`` calls ``child_retrieve`` exactly once with the
    original query string. We reuse the pre-computed ``query_vector`` to
    avoid a redundant embed call.

    Args:
        query: Raw query string (passed verbatim to amaxsim_retrieve).
        query_vector: Pre-computed query embedding; used directly for child
            ANN recall, bypassing a second embed call.
        fact_recaller: Provides the child ANN retrieval function.
        episode_recaller: Provides the parent fetch function.
        where: LanceDB filter clause.
        child_candidates: Number of atomic-fact candidates to pull per call.

    Returns:
        Episode candidates re-scored by their best matching atomic fact.
    """
    # 这个 helper 把 MaxSim 的“算法协议”适配到当前 memory 存储模型：
    # - child 是 atomic_fact，通过 dense_recall 在 fact 表里按 query_vector 搜；
    # - parent 是 episode，通过 atomic_fact.metadata["parent_id"] 指向的 memcell_id 批量取回；
    # - amaxsim_retrieve 负责“child 命中 → parent 聚合 → parent 重新计分”的纯算法部分。
    #
    # 因为上层已经提前计算了 query_vector，这里不再调用 embedding provider。
    # 这样 Layer 1 的 dense episode 召回、Layer 2 的 fact ANN、Layer 4 的 fact scoring
    # 可以共享同一个 query embedding，减少延迟，也避免同一 query 多次 embedding 产生细微差异。

    async def child_retrieve(_q: str, n: int) -> Sequence[Candidate]:
        # amaxsim_retrieve calls this exactly once with the original query string.
        # Reuse the pre-computed query_vector instead of re-embedding.
        # _q 被保留在签名里，是为了满足 amaxsim_retrieve 的回调接口；
        # 实际检索时使用的是已经注入进来的 query_vector。
        # n 由 amaxsim_retrieve 根据 child_candidates 传入，用来限制 atomic fact ANN 召回规模。
        return await fact_recaller.dense_recall(query_vector, where, limit=n)

    async def parent_fetch(memcell_ids: list[str]) -> list[Candidate]:
        # child_retrieve 返回的是 fact Candidate；MaxSim 算法会从 fact metadata 中收集 parent memcell id。
        # parent_fetch 再用这些 memcell ids 批量取回 episode Candidate。
        # where 会再次应用在 episode 查询上，防止 child 表与 episode 表之间因为租户/owner 过滤不一致
        # 而把不该返回的 parent episode 带出来。
        return await episode_recaller.fetch_by_parent_ids(memcell_ids, where)

    # top_n 固定为 50，表示 MaxSim 路径本身最多产出 50 个 episode 候选；
    # 真正最终返回多少仍由上层 Layer 3 的 top_k 控制。
    # child_candidates 控制 fact ANN 的入口宽度：入口越宽，越可能覆盖更多 parent memcell，
    # 但也会增加一次 fact 表向量扫描和后续 parent 聚合成本。
    return await amaxsim_retrieve(
        query,
        child_retrieve=child_retrieve,
        parent_fetch=parent_fetch,
        top_n=50,
        child_candidates=child_candidates,
    )


def _build_ep_to_memcell(episodes: list[Candidate]) -> dict[str, str]:
    """Extract episode_id → memcell_id mapping from episode candidates.

    Episodes store their source memcell id in metadata["parent_id"].
    Entries missing or having a non-string parent_id are silently skipped
    (they will receive no facts during Layer 4).

    Args:
        episodes: Merged episode candidate list.

    Returns:
        Dict mapping episode LanceDB id to memcell id.
    """
    # facts_for_episodes 的输入不是 episode id 列表，而是 episode_id → memcell_id。
    # 原因是 episode 和 atomic_fact 是两张不同粒度的表：
    # - episode Candidate 的 id 是 episode 这一行自己的 LanceDB id；
    # - atomic_fact.metadata["parent_id"] 指向的是 memcell_id；
    # - episode.metadata["parent_id"] 同样保存 memcell_id，作为两者之间的连接键。
    result: dict[str, str] = {}
    for ep in episodes:
        mc_id = ep.metadata.get("parent_id")
        # 缺少 parent_id 的 episode 不参与 Layer 4 的 fact 预取。
        # 这里选择跳过而不是报错，是为了让检索在部分脏数据/旧数据上仍能返回 episode 级结果。
        if isinstance(mc_id, str) and mc_id:
            result[ep.id] = mc_id
    return result
