"""SearchManager — top-level orchestrator for ``POST /api/v1/memory/search``.

Hard partition by ``owner_type``:

* ``user``  → ``episodes`` (+ ``profiles`` when ``include_profile=true``)
* ``agent`` → ``agent_cases`` + ``agent_skills``

Per kind, :func:`memory.search.adapter.resolve_pipeline` decides whether
the path is "single-route recall, no fusion" (``KEYWORD`` / ``VECTOR``)
or "sparse + dense → everalgo.rank" (``HYBRID`` / ``AGENTIC``). Component
guards (embedding / cross-encoder / LLM) raise early when a method is
selected without its prerequisites.

``HYBRID`` defaults to **no LLM rerank** — the response comes back
straight after the four-layer hierarchy pipeline (RRF → MaxSim →
RRF merge → single-pass fact eviction). ``enable_llm_rerank`` is
**ignored** for the hierarchy path. ``AGENTIC`` keeps its own
internal cross-encoder rerank loop; the flag is ignored there.

``SearchEpisodeItem.atomic_facts`` is populated **only** when the HYBRID
pipeline runs over episodes. The other methods leave it empty: there is
no query-relevance score we can assign to a fact pulled by parent_id
alone, and emitting ``score=0.0`` facts would muddy the contract.

The manager never writes to storage; it only reads LanceDB + markdown.
"""

# 下面开始进入实现层。上面的模块 docstring 已经说明了整体契约；这里的补充注释按代码执行顺序解释各部分如何协作。
# 这些注释尽量放在逻辑节点前，而不是逐行翻译语法，便于跟着一次 search 请求从入口走到返回。
from __future__ import annotations

# 异步并发是这段代码的主线之一：很多召回、profile、buffer 查询互不依赖，可以用 gather 同时跑。
import asyncio
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

# everalgo 负责候选融合与重排；SearchManager 只准备 RankInput、选择 config，并把结果再适配回 DTO。
from everalgo.rank import DEFAULT_RANK_CONFIG, RankConfig, arank
from everalgo.types import Candidate, RankInput

# everos 内部依赖覆盖配置、时区、日志/追踪和未处理消息缓存，都是编排层需要但不应自己实现的能力。
from everos.component.utils.datetime import to_display_tz
from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import gen_request_id
from everos.infra.persistence.sqlite import (
    UnprocessedBuffer,
    unprocessed_buffer_repo,
)

# 同包模块被拆成几类：pipeline 路由、agentic/hybrid 具体检索、DTO 定义、过滤器编译、候选结果 shaping。
from .adapter import resolve_pipeline
from .agentic import search_episodes_agentic
from .agentic_agent import search_agent_cases_agentic, search_agent_skills_agentic
from .dto import (
    FilterNode,
    SearchAgentCaseItem,
    SearchAgentSkillItem,
    SearchData,
    SearchEpisodeItem,
    SearchMethod,
    SearchProfileItem,
    SearchRequest,
    SearchResponse,
    UnprocessedMessageDTO,
)
from .filters import compile_filters
from .hierarchy import hierarchy_retrieve_episodes
from .shaper import (
    shape_agent_case_from_candidate,
    shape_agent_skill_from_candidate,
    shape_episode_from_candidate,
)
from .skill_hybrid import search_agent_skills_hybrid

# 类型检查专用导入放在 TYPE_CHECKING 下，运行时不加载这些协议/recaller 类型，避免不必要的依赖和循环导入。
if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

    from everos.component.embedding import EmbeddingProvider
    from everos.component.rerank import RerankProvider
    from everos.component.tokenizer import Tokenizer

    from .recall import (
        AgentCaseRecaller,
        AgentSkillRecaller,
        AtomicFactRecaller,
        EpisodeRecaller,
        ProfileRecaller,
    )

# 模块级 logger 只负责观测性，不参与搜索流程本身。真正的请求级关联 id 在 search() 里生成。
logger = get_logger(__name__)

# 下面这些常量决定“取多少候选”和“低分候选何时被挡掉”。它们不是业务返回字段，但会显著影响召回质量与成本。
# Recall pool sizing — matches the legacy enterprise constants
# ``DEFAULT_RECALL_MULTIPLIER`` / ``DEFAULT_TOPK_LIMIT``.
# Multiplier kicks in for ``top_k > 0``; ``top_k = -1`` (unlimited) is capped
# at the fixed top-k limit (100) rather than ``100 * multiplier`` — that way
# the recall pool never balloons to 500 in unlimited mode.
_DEFAULT_RECALL_MULTIPLIER = 2
_DEFAULT_TOP_K_CAP = 100

# Agent cases / skills carry heavy per-row payloads (``approach``,
# full ``content``); cap unlimited mode at 10 to keep rerank context
# bounded. Positive ``top_k`` from the caller bypasses this.
_AGENT_TOP_K_CAP = 10

# Vector ``radius`` (cosine similarity threshold) default for **unlimited
# mode only**. In ``top_k > 0`` mode we trust the truncation cap to ditch
# low-quality tail; in ``top_k = -1`` mode we would otherwise return up to
# 100 candidates with no quality floor, so we layer a default 0.5
# similarity threshold the way enterprise does (enterprise uses 0.6 — we
# pick 0.5 slightly looser because LanceDB cosine vs Milvus cosine score
# distributions can drift a bit on the same model).
_DEFAULT_UNLIMITED_RADIUS = 0.5

# ``maxsim_atomic`` recall pool sizing — atomic facts are ~28× denser than
# episodes (one memcell → 1 episode + ~28 atomic facts), so the fact pool
# is sized as ``top_k_episode * 20`` to consistently cover enough distinct
# parent memcells before the max-pool reduction. Capped to keep the ANN
# scan bounded on very large top_k requests.
_MAXSIM_FACT_MULTIPLIER = 20
_MAXSIM_FACT_POOL_CAP = 2000

# Mirror of ``service._boundary._TRACK``. The unprocessed buffer is a single
# shared track because boundary detection is single-pass — switching mode
# requires a fresh process. Hard-coded here (instead of importing) to keep
# the memory layer free of service-layer imports per the DDD direction rule.
_UNPROCESSED_TRACK = "memorize"


# SearchManager 是一个薄编排层：它不直接知道 LanceDB/markdown 的细节，只协调各种 recaller/provider 完成一次搜索。
# 代码整体按 owner_type 先硬分区，再按 method/kind 选择召回与融合路径。
class SearchManager:
    """Orchestrates per-kind recall, fusion, and shape into the public DTO."""

    def __init__(
        self,
        *,
        episode_recaller: EpisodeRecaller,
        atomic_fact_recaller: AtomicFactRecaller,
        agent_case_recaller: AgentCaseRecaller,
        agent_skill_recaller: AgentSkillRecaller,
        profile_recaller: ProfileRecaller,
        embedding: EmbeddingProvider | None,
        reranker: RerankProvider | None,
        llm_client: LLMClient | None,
        search_tokenizer: Tokenizer | None = None,
    ) -> None:
# 这些 recaller 对应不同可搜索对象：episode、atomic_fact、agent_case、agent_skill、profile。
# manager 保存它们的引用，但不创建它们，方便在应用启动或测试里注入不同实现。
        self._ep = episode_recaller
        self._fact = atomic_fact_recaller
        self._case = agent_case_recaller
        self._skill = agent_skill_recaller
        self._profile = profile_recaller
# 这三类组件是“能力开关”：embedding 支撑向量召回，reranker 支撑交叉编码器重排，LLM 支撑 LLM rerank/agentic。
# 它们允许为 None，但真正进入某条需要它们的路径前，_validate_components 会提前拦截。
        self._embedding = embedding
        self._reranker = reranker
        self._llm = llm_client
        self._search_tokenizer = search_tokenizer

    # ── Public entry ────────────────────────────────────────────────

# search() 是外部请求进入本模块的唯一主流程：生成 request_id、编译过滤器、校验组件、分区检索、组装响应。
    async def search(self, req: SearchRequest) -> SearchResponse:
# request_id 先生成，后续即使分支并发执行，也可以在观测系统里把一次请求串起来。
        request_id = gen_request_id()
        # Compile filters first: a malformed `filters` payload is a user
        # input error (422) and should surface before the server-side
        # component guard (500). The two steps are independent.
# where 是所有底层召回共享的过滤条件；先统一编译，后续 episode/case/skill/buffer 不再各自理解原始 filters。
        where = compile_filters(
            req.filters,
            owner_id=req.owner_id,
            owner_type=req.owner_type,
            app_id=req.app_id,
            project_id=req.project_id,
        )
# 过滤器语义确认无误后，再检查 method 需要的组件是否齐全；这样用户输入错误优先暴露为 422，而不是被 500 掩盖。
        self._validate_components(req)

# owner_type 是最高优先级分流：user 只查用户记忆相关对象；agent 只查 agent 的 case/skill。
# 这种硬分区避免后续融合时混入不同语义空间的候选。
        if req.owner_type == "user":
# user 分支里，episode 搜索、profile 读取、未处理消息读取互不依赖，所以并发执行以降低整体延迟。
            episodes, profiles, unprocessed = await asyncio.gather(
                self._search_episodes(req, where),
                self._fetch_profile(req),
                self._load_unprocessed(req),
            )
# DTO 只填充 user 场景应该返回的字段；agent_cases/agent_skills 在这里保持默认空值。
            data = SearchData(
                episodes=episodes,
                profiles=profiles,
                unprocessed_messages=unprocessed,
            )
        else:  # "agent"
# agent 分支把 cases 与 skills 看成一个组合任务，因为 HYBRID + LLM rerank 时 skill 可能要继承 case 分数。
            (cases, skills), unprocessed = await asyncio.gather(
                self._search_cases_and_skills(req, where),
                self._load_unprocessed(req),
            )
# agent 场景组装 agent_cases/agent_skills，同时也带上同 session 的未处理消息，让调用方看到“已落库”和“处理中”的完整视图。
            data = SearchData(
                agent_cases=cases,
                agent_skills=skills,
                unprocessed_messages=unprocessed,
            )

# 这里返回的是纯响应对象；本 manager 的设计约束是只读，不在 search 流程里写存储。
        return SearchResponse(request_id=request_id, data=data)

    # ── Unprocessed buffer ──────────────────────────────────────────

    async def _load_unprocessed(
        self, req: SearchRequest
    ) -> list[UnprocessedMessageDTO]:
        """Load in-flight buffer rows for ``filters.session_id`` (if present).

        Returns ``[]`` unless ``filters`` carries a top-level ``session_id``
        eq scalar — buffer rows have no ``user_id`` / ``agent_id`` attribution
        (boundary detection runs before owner inference), so session is the
        only meaningful query dimension.
        """
# 未处理 buffer 只能在 session 被明确钉住时读取；否则无法安全判断哪些 in-flight 消息属于当前搜索上下文。
        session_id = _extract_top_level_session_id(req.filters)
        if session_id is None:
            return []
# repo 查询同时带 track、app_id、project_id，避免同一 session 在不同应用/项目或不同处理轨道间串数据。
        rows = await unprocessed_buffer_repo.list_for_track(
            session_id,
            _UNPROCESSED_TRACK,
            app_id=req.app_id,
            project_id=req.project_id,
        )
# 存储层 row 在这里统一转换成对外 DTO，隐藏 JSON 字段和数据库模型细节。
        return [_unprocessed_buffer_to_dto(r) for r in rows]

    # ── Agent partition ─────────────────────────────────────────────

    async def _search_cases_and_skills(
        self, req: SearchRequest, where: str
    ) -> tuple[list[SearchAgentCaseItem], list[SearchAgentSkillItem]]:
        """Cases + skills, serial when bridging.

        HYBRID + LLM rerank runs serially: reranked cases feed the
        skill bridge. Every other method runs the two kinds in parallel
        with no bridge — the bridge only pays off after rerank has
        produced high-quality case scores to inherit.
        """
# 只有 HYBRID 且启用 LLM rerank 时，case→skill bridge 才有可靠分数来源，因此必须先串行拿到 cases。
        if _effective_llm_rerank(req):
            cases = await self._search_agent_cases(req, where)
# bridge 只需要 case 的 id 和 rerank 后分数；metadata 在 bridge 入口不参与计算，所以用空 dict 降低耦合。
            bridge_cases = [
                Candidate(id=c.id, score=c.score, source="vector", metadata={})
                for c in cases
            ]
# skills 检索收到 bridge_cases 后，会把“由高相关 case 反查到的 skill”并入 dense 候选池。
            skills = await self._search_agent_skills(
                req, where, bridge_cases=bridge_cases
            )
            return cases, skills

# 非 bridge 路径中，cases 和 skills 完全独立，直接并发执行；这也是大多数请求的低延迟路径。
        cases, skills = await asyncio.gather(
            self._search_agent_cases(req, where),
            self._search_agent_skills(req, where),
        )
        return cases, skills

    # ── Episodes ────────────────────────────────────────────────────

    async def _search_episodes(
        self, req: SearchRequest, where: str
    ) -> list[SearchEpisodeItem]:
# episode 的 AGENTIC 路径完全交给 agentic 实现：它会自己处理多轮/多查询/充分性判断等复杂逻辑。
        if req.method == SearchMethod.AGENTIC:
            return await search_episodes_agentic(
                req.query,
                owner_id=req.owner_id,
                where=where,
                episode_recaller=self._ep,
                atomic_fact_recaller=self._fact,
                embed_query_fn=self._embedding.embed,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
            )

# 非 AGENTIC 的 episode 搜索先问 adapter：当前 method 在 episode 这种 kind 上究竟是单路召回，还是需要融合。
        fusion_mode, _ = resolve_pipeline(req.method, "episode")
# LLM rerank 是否真的生效被集中封装，避免每个分支各自解释 enable_llm_rerank 的语义。
        enable_rerank = _effective_llm_rerank(req)
# top_k 先标准化：调用方传 -1 表示 unlimited，但内部必须落到一个明确上限，避免扫描和返回失控。
        top_k = self._top_k(req.top_k)

        # ── KEYWORD / VECTOR: single-route recall ──
# fusion_mode 为 None 表示 KEYWORD/VECTOR 这类单路召回：只拿一条候选流，不做 sparse+dense 融合。
        if fusion_mode is None:
# KEYWORD 不需要 embedding，直接走稀疏召回；这也是缺少 embedding provider 时唯一合法的搜索方法。
            if req.method == SearchMethod.KEYWORD:
                cands = await self._ep.sparse_recall(
                    req.query, where, limit=self._recall_limit(req.top_k)
                )
# VECTOR 在 episode 上可以被配置成 maxsim_atomic：先匹配更细粒度的 atomic facts，再回填到 episode。
            elif load_settings().search.vector_strategy == "maxsim_atomic":
                cands = await self._maxsim_atomic_recall(req, where, top_k)
            else:
# 标准向量路径先嵌入 query，再查 episode 向量索引；radius 只在向量候选上生效，用来砍掉低相似度尾部。
                vector = await self._embed_query(req.query)
                cands = await self._ep.dense_recall(
                    vector, where, limit=self._recall_limit(req.top_k)
                )
                cands = self._apply_radius(cands, _effective_radius(req))
            # ``atomic_facts`` stays empty: facts come back only when the HYBRID
            # pipeline surfaces them with a score (see ``reshape_hybrid_output``).
            # Single-route recall has no per-fact score against the query, so
            # we do not back-fill — that would emit ``score=0.0`` facts whose
            # semantics are ambiguous.
# 单路召回结束后只做 shape 和截断：Candidate 是内部表示，SearchEpisodeItem 才是 API 契约。
            return [
                ep
                for ep in (shape_episode_from_candidate(c) for c in cands[:top_k])
                if ep is not None
            ]

        # ── HYBRID: parallel sparse + dense recall ──
# HYBRID 路径先统一拿 sparse、dense 和 query_vector；后面的 hierarchy/rrf/lr 都复用这一组输入。
        sparse, dense, query_vector = await self._recall_sparse_dense(
            self._ep, req, where, top_k
        )

# hierarchy 是 episode HYBRID 的主路径：它会结合 episode 与 atomic facts 的层级关系做更细的去重和事实选择。
        if fusion_mode == "hierarchy":
            return await hierarchy_retrieve_episodes(
                req.query,
                sparse=sparse,
                dense=dense,
                query_vector=query_vector,
                fact_recaller=self._fact,
                episode_recaller=self._ep,
                where=where,
                top_k=top_k,
            )

        # rrf / lr: standard everalgo fusion path (fallback).
# 如果 adapter 返回的是 rrf/lr 等普通融合模式，就走 everalgo 的通用 rank 管线作为 fallback。
        output = await arank(
            RankInput(
                query=req.query,
                memory_type=self._ep.everalgo_memory_type,  # type: ignore[arg-type]
                sparse_candidates=sparse,
                dense_candidates=dense,
                top_k=top_k,
                radius=_effective_radius(req),
            ),
            config=RankConfig(fusion_mode=fusion_mode)
            if fusion_mode != "rrf"
            else DEFAULT_RANK_CONFIG,
            llm=self._llm,
            enable_rerank=enable_rerank,
            rerank_top_k=top_k,
        )
# arank 输出的是 everalgo 的 ScoredItem；这里转回 Candidate，复用已有的 shaper，避免 DTO 适配逻辑分叉。
        ep_candidates = (_scored_as_candidate(s) for s in output.items)
        return [
            ep
            for ep in (shape_episode_from_candidate(c) for c in ep_candidates)
            if ep is not None
        ]

    # ── Agent cases ─────────────────────────────────────────────────

    async def _search_agent_cases(
        self, req: SearchRequest, where: str
    ) -> list[SearchAgentCaseItem]:
# agent_case 的结构与 episode 类似：AGENTIC 先短路给专用实现，普通路径再进入 adapter 决策。
        if req.method == SearchMethod.AGENTIC:
            return await search_agent_cases_agentic(
                req.query,
                where=where,
                case_recaller=self._case,
                embed_query_fn=self._embedding.embed,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
            )
# case 的 pipeline 按 kind 单独解析，因为同一个 SearchMethod 在不同 kind 上可能有不同实现策略。
        fusion_mode, _ = resolve_pipeline(req.method, "agent_case")
        enable_rerank = _effective_llm_rerank(req)
# agent case 的单条 payload 比 episode 更重，所以 unlimited 模式使用更小的 cap 控制上下文和 rerank 成本。
        top_k = self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP)

# 单路 case 召回复用通用 helper：KEYWORD 走 sparse，VECTOR 走 dense，并统一应用 radius。
        if fusion_mode is None:
            cands = await self._single_route_recall(
                self._case, req, where, top_k, cap=_AGENT_TOP_K_CAP
            )
            shaped = (shape_agent_case_from_candidate(c) for c in cands[:top_k])
            return [item for item in shaped if item is not None]

# HYBRID case 与普通 episode fallback 一样，先并发拿 sparse+dense，再交给 everalgo 做融合/可选 LLM rerank。
        sparse, dense, _ = await self._recall_sparse_dense(
            self._case, req, where, top_k, cap=_AGENT_TOP_K_CAP
        )
        output = await arank(
            RankInput(
                query=req.query,
                memory_type=self._case.everalgo_memory_type,  # type: ignore[arg-type]
                sparse_candidates=sparse,
                dense_candidates=dense,
                top_k=top_k,
                radius=_effective_radius(req),
            ),
            config=RankConfig(fusion_mode=fusion_mode)
            if fusion_mode != "rrf"
            else DEFAULT_RANK_CONFIG,
            llm=self._llm,
            enable_rerank=enable_rerank,
            rerank_top_k=top_k,
        )
# case 的返回阶段同样是“内部候选 → DTO”，None 会被过滤，防止坏 metadata 泄漏到 API 响应。
        case_candidates = (_scored_as_candidate(s) for s in output.items)
        shaped = (shape_agent_case_from_candidate(c) for c in case_candidates)
        return [item for item in shaped if item is not None]

    # ── Agent skills ────────────────────────────────────────────────

    async def _search_agent_skills(
        self,
        req: SearchRequest,
        where: str,
        *,
        bridge_cases: list[Candidate] | None = None,
    ) -> list[SearchAgentSkillItem]:
        """Rank agent skills. ``bridge_cases`` (reranked case id+score) is
        supplied only on HYBRID + LLM-rerank to feed the case→skill bridge;
        ``None`` everywhere else.
        """
# skill 的 AGENTIC 路径也独立处理；只有非 AGENTIC 才继续考虑 single-route、hybrid、bridge 等逻辑。
        if req.method == SearchMethod.AGENTIC:
            return await search_agent_skills_agentic(
                req.query,
                where=where,
                skill_recaller=self._skill,
                embed_query_fn=self._embedding.embed,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP),
            )
# skill 的 pipeline 也按 kind 解析；这里特别重要，因为 agent_skill HYBRID 默认不是普通 arank，而是 skill 专用 cross-encoder lane。
        fusion_mode, _ = resolve_pipeline(req.method, "agent_skill")
        top_k = self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP)

# KEYWORD/VECTOR 单路 skill 召回不需要 case bridge，直接查 skill 自己的索引并 shape。
        if fusion_mode is None:
            cands = await self._single_route_recall(
                self._skill, req, where, top_k, cap=_AGENT_TOP_K_CAP
            )
            shaped = (shape_agent_skill_from_candidate(c) for c in cands[:top_k])
            return [item for item in shaped if item is not None]

# HYBRID skill 先拿直接召回候选；如果上游提供了 reranked cases，下一步会补充“由 case 反推出来的 skill”。
        sparse, dense, _ = await self._recall_sparse_dense(
            self._skill, req, where, top_k, cap=_AGENT_TOP_K_CAP
        )

        # Case→skill bridge: union skills surfaced via lineage cases into
        # the dense pool with their max-pooled source-case score.
# bridged skills 被并入 dense 池，而不是另开第三路；这样下游融合仍然只面对 sparse/dense 两条通用输入。
        bridged = await self._case_bridged_skills(bridge_cases, where, top_k)
        dense = _merge_by_id_max(dense, bridged)

        # Lane selection lives here so ``skill_hybrid`` stays single-purpose
        # (cross-encoder) and symmetry with the case path is preserved.
# 如果启用了 LLM rerank，skill 走通用 arank 的 LLM lane；这里要求 case bridge 的分数也来自 LLM rerank，保持量纲一致。
        if _effective_llm_rerank(req):
            # LLM lane: generic ``arank`` dispatches by ``memory_type="skill"``
            # to the skill facade (adds the skill-only 0.4 relevance gate).
            # Config is ``rrf`` — ``skill_hybrid`` is an everos routing
            # label, not an everalgo fusion mode.
# 注意这里 config 用 DEFAULT_RANK_CONFIG：skill_hybrid 是 everos 的路由标签，不是 everalgo 的 fusion_mode。
            output = await arank(
                RankInput(
                    query=req.query,
                    memory_type=self._skill.everalgo_memory_type,  # type: ignore[arg-type]
                    sparse_candidates=sparse,
                    dense_candidates=dense,
                    top_k=top_k,
                    radius=_effective_radius(req),
                ),
                config=DEFAULT_RANK_CONFIG,
                llm=self._llm,
                enable_rerank=True,
                rerank_top_k=top_k,
            )
# LLM lane 结束后仍然通过 shaper 统一成 SearchAgentSkillItem，保持两条 skill lane 的输出契约一致。
            skill_candidates = (_scored_as_candidate(s) for s in output.items)
            shaped = (shape_agent_skill_from_candidate(c) for c in skill_candidates)
            return [item for item in shaped if item is not None]

        # Cross-encoder lane (default): rrf + skill-shaped cross-encoder rerank.
# 默认 skill HYBRID 走 cross-encoder lane：先 rrf 合并，再用 skill 形态的交叉编码器做精排。
        return await search_agent_skills_hybrid(
            req.query,
            sparse=sparse,
            dense=dense,
            reranker=self._reranker,  # type: ignore[arg-type]
            top_k=top_k,
        )

    # ── Profile ─────────────────────────────────────────────────────

# profile 是 user 搜索的附加信息，不参与 episode 排名；只有调用方显式 include_profile 才读取。
    async def _fetch_profile(self, req: SearchRequest) -> list[SearchProfileItem]:
# agent 请求即使误传 include_profile，也不会读 profile，保持 owner_type 分区边界。
        if not req.include_profile or req.owner_type != "user":
            return []
# profile 直接按 owner_id 拉取，不走 query 相关性，因为它表示用户画像而不是检索候选。
        return await self._profile.fetch(req.owner_id)

    # ── Recall helpers ──────────────────────────────────────────────

    async def _single_route_recall(
        self,
        recaller: EpisodeRecaller | AgentCaseRecaller | AgentSkillRecaller,
        req: SearchRequest,
        where: str,
        top_k: int,
        *,
        cap: int = _DEFAULT_TOP_K_CAP,
    ) -> list[Candidate]:
# 这个 helper 抽象出 case/skill/episode 共有的单路召回逻辑，让各 kind 的主流程只关心“何时调用”。
        if req.method == SearchMethod.KEYWORD:
            return await recaller.sparse_recall(
                req.query, where, limit=self._recall_limit(req.top_k, cap=cap)
            )
# 非 KEYWORD 的单路路径都需要 query embedding；组件缺失在入口已校验，这里只负责执行。
        vector = await self._embed_query(req.query)
        cands = await recaller.dense_recall(
            vector, where, limit=self._recall_limit(req.top_k, cap=cap)
        )
# radius 只过滤 dense 召回结果；如果没有半径限制，候选按原顺序原样返回。
        return self._apply_radius(cands, _effective_radius(req))

    async def _recall_sparse_dense(
        self,
        recaller: EpisodeRecaller | AgentCaseRecaller | AgentSkillRecaller,
        req: SearchRequest,
        where: str,
        top_k: int,
        *,
        cap: int = _DEFAULT_TOP_K_CAP,
    ) -> tuple[list[Candidate], list[Candidate], list[float]]:
        """Fan out keyword + vector recall in parallel.

        The third return is the query embedding itself — the HYBRID
        pipeline passes it into ``facts_for_episodes`` so per-fact
        cosine scoring reuses the same vector instead of re-embedding
        the query. Returns
        ``[]`` for ``vector`` when no embedding provider is configured.
        """
# HYBRID 先嵌入一次 query，并把向量保留下来给后续 fact scoring 复用，避免重复 embedding。
        vector = await self._embed_query(req.query)
# sparse 和 dense 使用同一 recall limit，保证两路候选池规模可比，方便后续 RRF/LR/层级融合。
        limit = self._recall_limit(req.top_k, cap=cap)
# 稀疏召回和向量召回互不依赖，用 gather 并发；没有 embedding 时 dense 分支返回空列表。
        sparse, dense = await asyncio.gather(
            recaller.sparse_recall(req.query, where, limit=limit),
            recaller.dense_recall(vector, where, limit=limit)
            if vector
            else _empty_candidates(),
        )
# dense 先过 radius，再进入融合，避免低质量向量候选在 RRF 等融合算法里拿到额外机会。
        dense = self._apply_radius(dense, _effective_radius(req))
        return sparse, dense, vector

    async def _maxsim_atomic_recall(
        self, req: SearchRequest, where: str, top_k: int
    ) -> list[Candidate]:
        """MaxSim-style: ANN atomic_facts → max-pool by memcell → batch fetch episodes.

        Trades one extra LanceDB ANN scan (over the ~28× denser
        ``atomic_fact`` table) for finer-grained semantic match — long
        episodes whose single mean-pooled vector dilutes a specific topic
        recover via the matching atomic fact's own embedding. Mirrors
        the EverOS MaxSim retrieval pattern.
        """
# MaxSim atomic 先查更细粒度的 fact；如果无法生成 query 向量，就没有可执行的向量召回。
        vector = await self._embed_query(req.query)
        if not vector:
            return []
# fact 候选池按 episode top_k 放大，因为一个 episode 会拆出多个 fact；放大后才能覆盖足够多的 parent memcell。
        fact_limit = min(top_k * _MAXSIM_FACT_MULTIPLIER, _MAXSIM_FACT_POOL_CAP)
        fact_cands = await self._fact.dense_recall(vector, where, limit=fact_limit)
        # Max-pool fact scores by their parent memcell. ``atomic_fact``
        # rows always carry ``parent_id = memcell_id`` (cascade contract).
# 这里的核心是 max-pool：同一个 memcell 下只保留最相关 fact 的分数，作为该 episode 的代表分数。
        mc_score: dict[str, float] = {}
        for fc in fact_cands:
            mc = fc.metadata.get("parent_id")
            if not isinstance(mc, str) or not mc:
                continue
            if fc.score > mc_score.get(mc, -1.0):
                mc_score[mc] = fc.score
# 如果没有任何 fact 能映射回 parent memcell，就没有安全的 episode 可返回。
        if not mc_score:
            return []
# 先按 max-pooled fact 分数选出 top memcell，再去批量取 episode，避免对所有 fact parent 做昂贵回查。
        ranked = sorted(mc_score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        top_mc_ids = [mc for mc, _ in ranked]
        score_by_mc = dict(ranked)
        # One LanceDB scan: ``WHERE parent_id IN (...)``. The episode
        # ``where`` re-applies the partition filter so episodes whose
        # owner partition no longer matches the request are dropped.
        ep_cands = await self._ep.fetch_by_parent_ids(top_mc_ids, where)
# fetch 回来的 episode 自身分数不再可信为排序依据；这里用对应 memcell 的 fact max 分重新打分。
        rescored: list[Candidate] = []
        for c in ep_cands:
            mc = c.metadata.get("parent_id")
            s = score_by_mc.get(mc, 0.0) if isinstance(mc, str) else 0.0
            rescored.append(
                Candidate(id=c.id, score=s, source="vector", metadata=c.metadata)
            )
# 回填后按新的 MaxSim 分数排序，再套用 radius，保证输出顺序体现 fact 级匹配强度。
        rescored.sort(key=lambda c: c.score, reverse=True)
        return self._apply_radius(rescored, _effective_radius(req))

    async def _case_bridged_skills(
        self,
        bridge_cases: list[Candidate] | None,
        where: str,
        top_k: int,
    ) -> list[Candidate]:
        """Reverse-resolve lineage skills and max-pool their source-case
        scores.

        Reuses ``bridge_cases`` (already-reranked id+score) so ``agent_case``
        is never scanned twice. Scores must be LLM-rerank relevance in
        ``[0, 1]`` to stay comparable with the direct dense pool — never
        feed BM25 / fusion scores in here. Mirrors the ``maxsim_atomic``
        fact→episode pooling. Empty input ⇒ no bridge.
        """
# bridge 输入为空时直接退出；这样普通 HYBRID skill 不会额外扫描 case→skill lineage。
        if not bridge_cases:
            return []
# case_score 保存 reranked case 的相关性分数，后面每个 skill 会继承其来源 case 中最高的那个分数。
        case_score = {c.id: c.score for c in bridge_cases}
        # Bound the reverse fetch by the matched-case count; one case can map
        # to several skills, so allow a small fan-out per case.
# 反查 skill 时允许每个 case 有小幅 fan-out；既覆盖多技能来源，又避免 lineage 查询无限放大。
        skill_cands = await self._skill.fetch_by_case_ids(
            list(case_score), where, limit=max(top_k, len(case_score) * 4)
        )
        bridged: list[Candidate] = []
# 对每个 skill，读取 source_case_ids，找到命中的来源 case，并把最高 case 分数作为 bridged skill 分数。
        for sc in skill_cands:
            raw_ids = sc.metadata.get("source_case_ids")
            src_ids = raw_ids if isinstance(raw_ids, list) else []
            best = max(
                (case_score[cid] for cid in src_ids if cid in case_score),
                default=0.0,
            )
            bridged.append(
                Candidate(id=sc.id, score=best, source="vector", metadata=sc.metadata)
            )
# 返回的 bridged candidates 会在调用方与 direct dense candidates 做 id 去重和 max-score 合并。
        return bridged

# embedding 的空值处理集中在这里：调用方只需要判断返回向量是否为空，不直接触碰 provider。
    async def _embed_query(self, query: str) -> list[float]:
        if self._embedding is None:
            return []
        return await self._embedding.embed(query)

    # ── Limits / filters ────────────────────────────────────────────

    @staticmethod
# _top_k 处理“返回多少”的语义：-1 是调用方的 unlimited 表示，内部转换成可控 cap。
    def _top_k(top_k: int, *, cap: int = _DEFAULT_TOP_K_CAP) -> int:
        """Resolve ``-1`` to ``cap``; pass others through unchanged.

        ``cap`` defaults to the episode/atomic_fact upper bound; agent
        cases / skills pass :data:`_AGENT_TOP_K_CAP` so an unbounded
        request still returns a tight, rerank-friendly result set.
        """
        return cap if top_k == -1 else top_k

    @staticmethod
# _recall_limit 处理“先召回多少再排序”的语义：通常比最终 top_k 更大，给融合/重排留候选余量。
    def _recall_limit(top_k_request: int, *, cap: int = _DEFAULT_TOP_K_CAP) -> int:
        """Effective recall pool size — branches on the *raw* request value.

        Mirrors enterprise:

        - ``top_k == -1`` (unlimited)  → fixed ``cap``
        - ``top_k > 0``                → ``top_k * DEFAULT_RECALL_MULTIPLIER``

        ``cap`` aligns the unlimited-mode pool with each kind's
        :meth:`_top_k` ceiling (e.g. agent kinds use the tighter
        :data:`_AGENT_TOP_K_CAP`).
        """
        if top_k_request == -1:
            return cap
        return max(
            top_k_request * _DEFAULT_RECALL_MULTIPLIER, _DEFAULT_RECALL_MULTIPLIER
        )

    @staticmethod
# 半径过滤是一个很窄的工具函数，但集中放这里能保证 episode/case/skill 对阈值的解释一致。
    def _apply_radius(cands: list[Candidate], radius: float | None) -> list[Candidate]:
        if radius is None:
            return cands
        return [c for c in cands if c.score >= radius]

    # ── Component guards ────────────────────────────────────────────

# 组件校验把“某条路线需要什么能力”集中声明在入口附近，避免在深层调用里才 AttributeError。
    def _validate_components(self, req: SearchRequest) -> None:
        """Fail fast when the chosen method needs components that are missing."""
        method = req.method
# 只要不是 KEYWORD，就一定需要 embedding；这是向量召回、HYBRID dense 路、AGENTIC 查询扩展的共同前提。
        needs_embedding = method != SearchMethod.KEYWORD
        if needs_embedding and self._embedding is None:
            raise RuntimeError(
                f"method={method.value!r} requires an embedding provider; "
                "configure [embedding] in settings"
            )
        # LLM is only mandatory when the caller explicitly opts into
        # Phase-5 rerank on HYBRID, or always for AGENTIC (sufficiency
        # check + multi-query generation).
        if (
            method == SearchMethod.HYBRID
            and req.enable_llm_rerank
            and self._llm is None
        ):
            raise RuntimeError(
                "method='hybrid' with enable_llm_rerank=true needs an LLM; "
                "configure [llm] in settings or drop the flag"
            )
        # agent_skill HYBRID without LLM rerank reaches the cross-encoder
        # lane; without the reranker it would AttributeError deep in the
        # callback. Episode / agent_case HYBRID don't need it.
        if (
            method == SearchMethod.HYBRID
            and req.owner_type == "agent"
            and not req.enable_llm_rerank
            and self._reranker is None
        ):
            raise RuntimeError(
                "owner_type='agent' with method='hybrid' requires a rerank "
                "provider (skill cross-encoder lane); configure [rerank] in "
                "settings, or set enable_llm_rerank=true to use the LLM lane"
            )
# AGENTIC 始终需要 reranker 和 LLM，因为它不是单纯召回，而是带多步判断和重排的智能搜索流程。
        if method == SearchMethod.AGENTIC:
            if self._reranker is None:
                raise RuntimeError(
                    "method='agentic' requires a rerank provider; "
                    "configure [rerank] in settings"
                )
            if self._llm is None:
                raise RuntimeError(
                    "method='agentic' requires an LLM client; "
                    "configure [llm] in settings"
                )


# 下面是模块级小工具：它们不依赖 SearchManager 实例，只封装跨分支共享的适配和策略判断。
def _scored_as_candidate(scored) -> Candidate:  # type: ignore[no-untyped-def]
    """Adapt a single-type ``ScoredItem`` back to a ``Candidate``.

    Adapts ``ScoredItem`` back to ``Candidate`` so the existing
    Candidate-based shapers apply.
    """
# Candidate 是本模块 shaper 的统一输入格式；即使上游类型不同，也先对齐到这层再继续。
    return Candidate(
        id=scored.id,
        score=scored.score,
        source="other",
        metadata=dict(scored.metadata),
    )


# LLM rerank 的有效条件集中写成函数，避免 HYBRID/AGENTIC 对 enable_llm_rerank 的特殊规则散落各处。
def _effective_llm_rerank(req: SearchRequest) -> bool:
    """LLM Phase-5 rerank only kicks in for ``HYBRID`` and only when the
    caller opts in. ``AGENTIC`` runs its own cross-encoder rerank loop
    (via ``rerank_fn``) and intentionally skips Phase-5.
    """
    return req.method == SearchMethod.HYBRID and req.enable_llm_rerank


# radius 的默认值不是简单配置读取，而是和 top_k 语义绑定：只有 unlimited 时才自动加质量地板。
def _effective_radius(req: SearchRequest) -> float | None:
    """Resolve the cosine-similarity threshold actually applied to dense hits.

    Priority:

    1. Caller-supplied ``req.radius`` always wins (including ``0.0`` when
       they explicitly want everything).
    2. Otherwise, ``top_k == -1`` (unlimited) defaults to
       ``_DEFAULT_UNLIMITED_RADIUS`` so the response keeps a quality
       floor — matches enterprise's auto-default behaviour.
    3. Otherwise (normal ``top_k > 0`` mode), return ``None`` and trust
       truncation to handle tail quality.
    """
    if req.radius is not None:
        return req.radius
    if req.top_k == -1:
        return _DEFAULT_UNLIMITED_RADIUS
    return None


# gather 需要 awaitable；用这个空协程可以让“没有 dense 分支”也保持同样的并发代码形状。
async def _empty_candidates() -> list[Candidate]:
    return []


# session_id 提取故意只支持顶层字面量：越复杂的过滤表达式，越难给 buffer 读取定义准确范围。
def _extract_top_level_session_id(filters: FilterNode | None) -> str | None:
    """Return the literal value of a top-level ``session_id`` eq scalar.

    The unprocessed-buffer trigger only fires for the simple shape
    ``filters = {"session_id": "<sid>"}``. Anything wrapped in ``AND`` /
    ``OR``, nested deeper, or expressed via an operator map (``{"eq":
    ...}``, ``{"in": ...}``) is treated as "session not pinned" — there
    is no defensible buffer-scope mapping for those compound predicates.
    """
# filters 为空或没有明确 session_id 时，宁可不返回 buffer，也不猜测范围。
    if filters is None:
        return None
    extra = filters.__pydantic_extra__ or {}
    value = extra.get("session_id")
    return value if isinstance(value, str) and value else None


# buffer row 的 content/tool_calls 存成 JSON；这里负责恢复成 API DTO，并保留多模态内容的原始结构。
def _unprocessed_buffer_to_dto(row: UnprocessedBuffer) -> UnprocessedMessageDTO:
    """Render one ``unprocessed_buffer`` row as its public DTO.

    Mirrors :class:`MessageItemDTO`'s ``content`` shorthand: a single-item
    ``[{"type":"text","text":...}]`` payload collapses to the inner string;
    every other shape stays as the opaque ``list[dict]`` so multimodal
    payloads round-trip without lossy flattening.
    """
# content_items 先反序列化，再根据形状决定是否折叠成字符串，兼容纯文本和多模态 payload。
    content_items = json.loads(row.content_items_json)
    if (
        isinstance(content_items, list)
        and len(content_items) == 1
        and isinstance(content_items[0], dict)
        and content_items[0].get("type") == "text"
        and isinstance(content_items[0].get("text"), str)
    ):
        content: str | list[dict[str, object]] = content_items[0]["text"]
    else:
        content = content_items
# tool_calls 允许为空；只有存储层确实带 JSON 时才解析，避免把 None 误变成空列表。
    tool_calls = (
        json.loads(row.tool_calls_json) if row.tool_calls_json is not None else None
    )
# 最后统一补齐 id、分区、发送者、角色、时间和工具调用信息，形成公开返回对象。
    return UnprocessedMessageDTO(
        id=row.message_id,
        app_id=row.app_id,
        project_id=row.project_id,
        session_id=row.session_id,
        sender_id=row.sender_id,
        sender_name=row.sender_name,
        role=row.role,  # type: ignore[arg-type]
        content=content,
        timestamp=to_display_tz(row.timestamp),
        tool_calls=tool_calls,
        tool_call_id=row.tool_call_id,
    )


# 合并候选时按 id 去重并保留高分，主要用于把 bridged skills 折进 direct dense pool。
def _merge_by_id_max(
    primary: list[Candidate], extra: list[Candidate]
) -> list[Candidate]:
    """Union by id, keep higher score. Folds bridged skills into the dense
    pool so downstream fusion doesn't double-count overlap.
    """
# primary 先占位，extra 只有在新候选不存在或分数更高时才覆盖，避免重复候选被下游融合双计。
    by_id: dict[str, Candidate] = {c.id: c for c in primary}
    for c in extra:
        existing = by_id.get(c.id)
        if existing is None or c.score > existing.score:
            by_id[c.id] = c
    return list(by_id.values())


# Sequence 只为了类型注解被导入；这行显式使用它，避免静态检查或 lint 把该导入标成未使用。
_ = Sequence  # quiet unused-import for the typing-only annotation above
