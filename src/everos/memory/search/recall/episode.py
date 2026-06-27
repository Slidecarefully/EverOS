"""Episode recaller — BM25 over ``episode_tokens`` + cosine ANN."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everalgo.types import Candidate

from everos.infra.persistence.lancedb import Episode, get_table

from .base import (
    RecallerDeps,
    build_or_query,
    cosine_score_from_distance,
    row_to_candidate,
)


def _q(value: str) -> str:
    # 这里不是做业务过滤，而是给后面手写 LanceDB SQL where 子句时使用。
    # parent_id 会被拼进 ``IN ('...')``，所以单引号必须转义成 SQL 字符串里的两个单引号，
    # 避免 parent_id 本身带引号时破坏查询语法。
    return value.replace("'", "''")


class EpisodeRecaller:
    """BM25 + vector recall over the LanceDB ``episode`` table."""

    # 这三个类属性把“当前 recaller 负责哪类 memory、给 everalgo 看成哪种 memory、
    # 以及文本正文在哪个字段里”固定下来。
    #
    # 上层 SearchManager / ranker / shaper 不需要知道 LanceDB 表的细节，
    # 只要依赖这些稳定约定，就能把 episode 召回结果接进不同检索管线：
    # - ``kind`` 标识这是 episode recaller；
    # - ``everalgo_memory_type`` 给 everalgo rank/fusion 判断记忆类型；
    # - ``text_field`` 告诉 rerank 或格式化逻辑从 metadata 的哪个字段取正文。
    kind: ClassVar[str] = "episode"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "episode"

    def __init__(self, deps: RecallerDeps) -> None:
        # Recaller 本身不持有 tokenizer / embedding / DB 连接等全局对象，
        # 而是通过 deps 注入需要的轻量依赖。
        # 在这个类里，最直接使用的是 tokenizer：sparse_recall 会用它把 query
        # 编译成 LanceDB 的 BM25 BooleanQuery。
        self._deps = deps

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> list[Candidate]:
        """BM25 recall via OR-mode BooleanQuery.

        Each tokenised term becomes a ``SHOULD`` clause so a single
        IDF≈0 token (typically the partition owner's own name on
        owner-scoped corpora) cannot poison the entire query.
        Mirrors enterprise's ``bool.should + minimum_should_match=1``
        ES design.
        """

        # sparse_recall 是 episode 的关键词召回入口。
        #
        # 上层已经把 owner/app/project/session 等过滤条件编译进 where；
        # 这里只负责把自然语言 query 转成 BM25 可以理解的 OR-mode 查询。
        # OR-mode 的意义是：只要命中部分 token 就可以进入候选池，
        # 避免某个无区分度 token 把整个查询拖垮。
        bq = build_or_query(self._deps.tokenizer, query, column=Episode.BM25_FIELDS[0])

        # 如果 query 经过 tokenizer 后没有任何可用于 BM25 的 token，
        # 就没有必要访问 LanceDB；直接返回空候选，让上层决定是否还有 dense
        # 或其他路径可以补上结果。
        if bq is None:
            return []

        # 真正的 I/O 从这里开始。
        #
        # get_table 延迟取得 episode 表；随后按“文本近邻 + where 分区过滤 + limit”
        # 取回原始 row。这里返回的 row 仍是存储层形态，后面必须转换为 Candidate，
        # 才能进入 SearchManager、everalgo 或 shaper 的统一数据流。
        table = await get_table(Episode.TABLE_NAME, Episode)
        rows = (
            await table.query().nearest_to_text(bq).where(where).limit(limit).to_list()
        )

        # LanceDB/BM25 的原始分数字段是 ``_score``。
        # row_to_candidate 会把 row 里的 id、metadata 等包装成 everalgo 通用的 Candidate；
        # source 标成 keyword，方便后续 fusion / debug 区分它来自 sparse 路径。
        return [
            row_to_candidate(r, source="keyword", score=float(r.get("_score", 0.0)))
            for r in rows
        ]

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> list[Candidate]:
        # dense_recall 是 episode 的向量召回入口。
        #
        # 和 sparse_recall 不同，这里假设上层已经完成 query embedding，
        # 只把向量传进来；这样 SearchManager 可以复用同一个 query_vector，
        # 避免 hierarchy / MaxSim / facts_for_episodes 等路径重复 embed。
        if not vector:
            return []

        # 这里同样只做 episode 表内的一次 ANN 查询：
        # - nearest_to(list(vector)) 指定查询向量；
        # - distance_type("cosine") 指定余弦距离；
        # - where 保证只搜当前 owner / app / project / filters 范围内的数据；
        # - limit 控制召回池大小，通常由 SearchManager 根据 top_k 放大得到。
        table = await get_table(Episode.TABLE_NAME, Episode)
        rows = (
            await table.query()
            .nearest_to(list(vector))
            .distance_type("cosine")
            .where(where)
            .limit(limit)
            .to_list()
        )

        # LanceDB 返回的是 cosine distance，不是业务上更直观的 similarity score。
        # cosine_score_from_distance 把距离转成“越大越相关”的分数，
        # 这样 sparse、dense、MaxSim、RRF 等后续逻辑都可以按统一方向排序。
        # source 标成 vector，表示该 Candidate 来自 dense ANN 召回。
        return [
            row_to_candidate(
                r,
                source="vector",
                score=cosine_score_from_distance(r.get("_distance")),
            )
            for r in rows
        ]

    async def fetch_by_parent_ids(
        self, parent_ids: Sequence[str], where: str
    ) -> list[Candidate]:
        """Batch-fetch episodes whose ``parent_id`` (memcell id) is in the set.

        One LanceDB scan per call (``WHERE parent_id IN (...)``) — used by
        the MaxSim-style vector strategy that first ranks memcells via
        ``atomic_fact`` cosine and then reverse-resolves the episode.
        ``score`` on the returned candidates is left at ``0.0``; the
        caller re-attaches the upstream max-pool score before sorting.
        """

        # 这个方法不是按 query 搜 episode，而是按 memcell id 反查 episode。
        #
        # 它服务于 fact-driven / MaxSim 路径：
        # 1. atomic_fact 表先根据 query 找到最相关的事实；
        # 2. 每个 fact 通过 metadata["parent_id"] 指回 memcell；
        # 3. 上层按 memcell max-pool 得到一批 parent_ids；
        # 4. 这里再把这些 memcell 对应的 episode 行取回来。
        #
        # 所以 parent_ids 为空时，说明上游 fact 路径没有找到可回溯的 memcell，
        # 直接返回空列表即可。
        if not parent_ids:
            return []

        table = await get_table(Episode.TABLE_NAME, Episode)

        # parent_ids 会拼到 SQL ``IN`` 子句里，因此每个 id 都先经过 _q 转义。
        # full_where 在调用方给出的 owner/filter 条件外再叠加 parent_id 限定：
        # 即使上游传入的 parent_ids 来自某个召回结果，这里仍重新应用 where，
        # 防止跨 owner / app / project 的 episode 被误取回来。
        quoted = ", ".join(f"'{_q(p)}'" for p in parent_ids)
        full_where = f"({where}) AND (parent_id IN ({quoted}))"

        # limit 使用 len(parent_ids)，因为理论上一批 memcell id 最多对应同等数量的 episode。
        # 返回分数保持 0.0 是有意为之：这个函数只负责“取回父 episode 的内容和 metadata”，
        # 真正的相关性分数来自上游 fact max-pool，调用方会在拿到结果后重新赋分。
        rows = await table.query().where(full_where).limit(len(parent_ids)).to_list()
        return [row_to_candidate(r, source="vector", score=0.0) for r in rows]

    async def fetch_all_for_owner(self, where: str) -> list[Candidate]:
        """Flat scan — all episodes for this owner, keyed by memcell id.

        Returns every episode row as a ``Candidate`` with ``id = parent_id``
        (the memcell id) so ``acluster_retrieve`` membership matching against
        ``cluster.members`` (also memcell ids) works without extra mapping.
        The real LanceDB episode id travels in ``metadata["episode_id"]`` so
        the agentic orchestrator can restore canonical episode identity after
        ``aagentic_retrieve`` returns.

        No ``limit`` is applied — the full owner partition is required for
        cluster membership matching (``acluster_retrieve`` needs ``all_docs``
        to cover every member of every cluster).
        """

        # 这个方法专门服务 agentic / cluster 检索路径，而不是普通 top_k 搜索。
        #
        # cluster_retrieve 需要拿到 owner 范围内的全量 docs，
        # 再根据 cluster.members 做成员扩展和范围收窄。
        # cluster.members 使用的是 memcell id；而 episode 表自己的 LanceDB id
        # 是 episode row id。为了让 all_docs 能直接和 cluster.members 对齐，
        # 这里会把 Candidate.id 改成 parent_id，也就是 memcell id。
        table = await get_table(Episode.TABLE_NAME, Episode)
        rows = await table.query().where(where).to_list()

        result: list[Candidate] = []
        for r in rows:
            # parent_id 是 episode 所属 memcell 的 id。
            # 如果这一行缺少 parent_id，或者 parent_id 不是有效字符串，
            # 它就无法参与 cluster membership 匹配，只能跳过。
            mc_id = r.get("parent_id")
            if not isinstance(mc_id, str) or not mc_id:
                continue

            # 先用 row_to_candidate 按标准方式抽取 metadata。
            # 这一步得到的 base.id 仍然是 episode LanceDB id，
            # 后面会把它塞进 metadata["episode_id"]，避免在改写 Candidate.id
            # 为 memcell id 后丢失真实 episode 身份。
            base = row_to_candidate(r, source="vector", score=0.0)

            # 输出给 acluster_retrieve / aagentic_retrieve 的 Candidate：
            # - id 使用 memcell id，便于和 cluster.members 对齐；
            # - metadata["episode_id"] 保留真实 episode id，便于最终 shape DTO 时恢复；
            # - score 设为 0.0，因为这里是全量语料池，不是一次相关性召回。
            result.append(
                Candidate(
                    id=mc_id,
                    score=0.0,
                    source="vector",
                    metadata={**base.metadata, "episode_id": base.id},
                )
            )
        return result
