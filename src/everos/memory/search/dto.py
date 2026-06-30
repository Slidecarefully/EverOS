"""Public DTOs for ``POST /api/v1/memory/search``.

Contract per the final design:

* ``owner_type`` is a hard partition. ``user`` returns ``episodes``
  (and optionally ``profiles``); ``agent`` returns ``agent_cases`` +
  ``agent_skills``. The five ``data.*`` arrays always exist; routes not
  applicable to the current ``owner_type`` stay as ``[]``.
* ``atomic_facts`` are **nested** inside :class:`SearchEpisodeItem`,
  never returned as a top-level array.
* Item-side ``owner_type`` / ``type`` fields are intentionally narrowed
  to the currently-emitted Literal so callers get a tight schema. Loosen
  them only when a new emission path (agent episodes, agent profiles)
  ships.

The :class:`FilterNode` model is intentionally permissive
(``extra="allow"``) because the DSL has an open key shape; the
allow-list / safety validation runs in :mod:`everos.memory.search.filters`
at compile time, not via Pydantic.
"""

from __future__ import annotations

import datetime as _dt
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 这个文件处在 search 模块的最外层边界：它定义的是 HTTP API 请求/响应的公共形状，
# 而不是内部 LanceDB row、everalgo Candidate 或 ScoredItem 的形状。
#
# 上游客户端只认识这些 DTO；下游 SearchManager、recaller、shaper 则负责在内部类型和
# 这些 DTO 之间转换。把契约集中放在这里，可以避免检索算法变化时直接污染 wire schema。


class SearchMethod(StrEnum):
    """Public method enum. RRF / LR / vector_anchored are hidden under HYBRID."""

    # 这里暴露给 API 调用方的是四个高层搜索意图，而不是内部所有 ranking/fusion 细节。
    # 例如 HYBRID 下面实际可能走 hierarchy、vector_anchored、skill_hybrid 等不同路线，
    # 但客户端不需要知道这些内部 routing signal；SearchManager 会结合 owner_type/kind 再分派。
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"
    AGENTIC = "agentic"


class FilterNode(BaseModel):
    """One Filters DSL node.

    Recursive ``AND`` / ``OR`` arrays mix with arbitrary scalar fields at
    the same level. Pydantic only checks the combinators; field-level
    safety is enforced when compiling the node to a LanceDB ``where``
    string in :mod:`everos.memory.search.filters`.
    """

    # filters 是一个开放 DSL：除了 AND/OR 这种组合符，业务字段可以是 session_id、
    # sender_id、timestamp、metadata 字段等。这里不能用 Pydantic 把字段写死，
    # 否则 DSL 扩展时每次都要改 DTO。
    #
    # 因此 Pydantic 只负责允许额外字段进入模型；真正的字段白名单、操作符安全检查、
    # SQL/LanceDB where 转义，都推迟到 compile_filters 阶段完成。
    model_config = ConfigDict(extra="allow")

    # AND / OR 本身是递归结构，可以继续嵌套 FilterNode。
    # 这让请求既能表达简单的 {"session_id": "..."}，也能表达组合过滤条件。
    AND: list[FilterNode] | None = None
    OR: list[FilterNode] | None = None


# ── Request ──────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Request body for ``POST /api/v1/memory/search``.

    Callers identify the memory owner via ``user_id`` XOR ``agent_id`` —
    exactly one must be set. Internally the manager + compile_filters keep
    using ``owner_id`` / ``owner_type`` (the storage tables' columns);
    those are exposed as derived properties so the rename only affects
    the wire contract, not the internal recall plumbing.
    """

    # 请求体采用严格模式：除了 filters DSL 这个刻意开放的子模型，顶层字段不允许随便扩展。
    # 这样能把拼错的参数、过期参数、误传字段尽早挡在 API 边界，而不是流入检索层。
    model_config = ConfigDict(extra="forbid")

    # user_id / agent_id 是这套搜索 API 最重要的分区开关。
    # 它们不是普通过滤条件，而是决定整个响应形态：
    # - user_id 表示查 user memory，返回 episodes/profiles；
    # - agent_id 表示查 agent memory，返回 agent_cases/agent_skills。
    #
    # 下面的 validator 会强制二者“有且只有一个”，避免一次请求跨两个 owner partition。
    user_id: str | None = Field(default=None, min_length=1)
    agent_id: str | None = Field(default=None, min_length=1)
    """Memory owner — provide ``user_id`` for user-memory (episodes /
    profiles) or ``agent_id`` for agent-memory (cases / skills); exactly
    one must be set."""

    # app_id / project_id 是 owner 之外的空间隔离维度。
    # SearchManager 会把它们和 owner_id/owner_type 一起编译进 LanceDB where，
    # 保证同一个用户或 agent 在不同 app/project 下的数据不会互相串。
    app_id: str = "default"
    project_id: str = "default"
    """App / project scope (default ``"default"``). Pinned into the LanceDB
    ``where`` so a search never crosses into another space's rows."""

    # query 是所有搜索方法共享的输入文本。
    # KEYWORD 会把它 tokenizer 成 BM25 查询；VECTOR/HYBRID/AGENTIC 会进一步生成 embedding，
    # AGENTIC 还可能用它做 LLM sufficiency / multi-query 扩展。
    query: str = Field(min_length=1)

    # method 是调用方能选择的检索策略层级。
    # 默认 HYBRID，表示优先使用关键词和向量两路信号；具体内部管线仍由 owner_type 和 kind 决定。
    method: SearchMethod = SearchMethod.HYBRID

    # top_k = -1 表示“使用服务端默认上限”，不是无限无界返回。
    # SearchManager 会把 -1 解析成不同 kind 的 cap：episode 默认较大，agent case/skill 较小，
    # 以防重排上下文膨胀。
    top_k: int = -1

    # radius 是 dense/vector 路径上的相似度阈值。
    # 它只对有向量分数的候选有意义；KEYWORD 路径不需要 embedding，也不会用这个阈值。
    radius: float | None = Field(default=None, ge=0.0, le=1.0)

    # profile 只属于 user owner 分区，而且是直接 fetch，不是 episode 搜索的一部分。
    # 因此 include_profile 只是告诉 SearchManager 在 user search 时额外并行取 profile；
    # agent search 下这个字段不会产生 agent profile。
    include_profile: bool = False

    # LLM rerank 被设计成显式 opt-in，避免 HYBRID 默认路径产生额外 LLM 成本和延迟。
    # 但这个 flag 并不是对所有路径都生效：episode hierarchy 有自己的 fact eviction，
    # AGENTIC 也有自己的内部 rerank/decision loop，所以 SearchManager 会按方法和 kind 再解释它。
    enable_llm_rerank: bool = Field(
        default=False,
        description=(
            "Opt-in LLM rerank pass for HYBRID. Applies to agent_case "
            "and agent_skill fusion only; the episode hierarchy path "
            "has built-in fact eviction and ignores this flag. "
            "Ignored by keyword / vector / agentic."
        ),
    )

    # filters 是请求级的额外过滤 DSL。
    # 它不会在 DTO 层直接变成 SQL，而是先保留为 FilterNode，随后由 compile_filters
    # 结合 owner/app/project 统一编译成 LanceDB where。
    filters: FilterNode | None = None

    @model_validator(mode="after")
    def _validate_user_xor_agent(self) -> SearchRequest:
        # 搜索入口必须先确定 owner partition。
        # 如果两个都为空，系统不知道查 user memory 还是 agent memory；
        # 如果两个都提供，则一次请求会同时跨 user/agent 两套数据模型，响应契约也会变得不清晰。
        # 因此这里强制 XOR：恰好一个存在。
        if (self.user_id is None) == (self.agent_id is None):
            raise ValueError("exactly one of user_id / agent_id must be provided")
        return self

    @model_validator(mode="after")
    def _validate_top_k(self) -> SearchRequest:
        # top_k 只允许两类值：
        # - -1：交给服务端使用默认 cap；
        # - 1..100：调用方显式要求的有限返回数。
        #
        # 0 没有实际搜索意义；小于 -1 不具备语义；大于 100 会让召回池、重排、facts 预取
        # 这些后续阶段成本失控，所以在 DTO 边界直接拒绝。
        if self.top_k == 0 or self.top_k < -1 or self.top_k > 100:
            raise ValueError("top_k must be -1 or in 1..100")
        return self

    @property
    def owner_id(self) -> str:
        """Derived from whichever of ``user_id`` / ``agent_id`` is set.

        The xor validator guarantees exactly one is non-None, so the
        ``or`` falls through to a real string (never the ``""`` default).
        """
        # 内部存储表使用统一的 owner_id 字段，不区分 user_id/agent_id 两个列名。
        # 这个 property 把 wire contract 的两个入口字段折叠成内部通用字段，
        # 让 SearchManager 和 compile_filters 可以继续用 owner_id 编排召回。
        return self.user_id or self.agent_id or ""

    @property
    def owner_type(self) -> Literal["user", "agent"]:
        """``"user"`` if ``user_id`` is set, else ``"agent"``."""
        # owner_type 与 owner_id 配套使用。
        # 它不仅参与 where 分区过滤，还决定 SearchManager 后面走 user 分支还是 agent 分支，
        # 从而决定 data.episodes / data.agent_cases 等数组哪个会被填充。
        return "user" if self.user_id is not None else "agent"


# ── Item DTOs ────────────────────────────────────────────────────────────


class SearchAtomicFactItem(BaseModel):
    """A single atomic fact nested inside its parent episode."""

    # atomic fact 在公共响应里只是 episode 的子结构，不是 top-level search result。
    # 这点和 hierarchy eviction 的内部 ScoredItem 不同：内部可以让 fact 赢得排序位置，
    # 但 shaper 最终一定会把它挂回 parent SearchEpisodeItem.atomic_facts。
    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    score: float


class SearchEpisodeItem(BaseModel):
    """Episode hit — always user-scoped in the current emission contract.

    ``type`` is narrowed to ``"Conversation"`` because the only emitted
    episode shape today is conversation-derived; widen when other
    sources ship. Item kind is encoded by class name (no ``owner_type``
    field on the wire), so episode results never carry ambiguity.
    """

    # episode 是 user memory 当前唯一的顶层召回 item。
    # 即使某个 atomic fact 在 hierarchy eviction 中比 parent episode 分数更高，
    # 最终也会通过 reshape_hybrid_output 被包装回 SearchEpisodeItem，
    # 只是 score 可能来自 winning fact，并且 atomic_facts 中带上那个 fact。
    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    """Owning user (``None`` only on malformed cascade rows)."""
    app_id: str = "default"
    project_id: str = "default"
    session_id: str
    timestamp: _dt.datetime
    sender_ids: list[str] = Field(default_factory=list)
    summary: str
    subject: str
    episode: str
    type: Literal["Conversation"]
    score: float

    # atomic_facts 是 hierarchy HYBRID 中 fact eviction 的外部承载位置。
    # 它为空时表示这条结果是普通 episode 命中；非空时表示该 episode 下有更细粒度的
    # fact 被认为与 query 更相关。客户端仍然只需要遍历 episodes，不需要额外处理顶层 facts 数组。
    atomic_facts: list[SearchAtomicFactItem] = Field(default_factory=list)


class SearchProfileItem(BaseModel):
    """Owner profile — at most one per response, only for user owners.

    ``score`` is ``None`` for direct fetches (``include_profile=true``
    on its own does no ranking); a future query-aware lookup may fill
    it in.
    """

    # profile 和 episode 同属 user partition，但它不是通过 query recall 排出来的。
    # 当前实现里 include_profile=true 只是直接 fetch owner profile，所以 score 可以为 None。
    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: str | None
    app_id: str = "default"
    project_id: str = "default"
    profile_data: dict[str, object]
    score: float | None = None


class SearchAgentCaseItem(BaseModel):
    """Agent case hit — always agent-scoped."""

    # agent_case 是 agent owner 分区下的历史案例 / 经验轨迹。
    # 它和 episode 不共用 DTO，是因为它的核心字段不是 conversation episode，
    # 而是 task_intent、approach、quality_score 等 self-evolution 语义字段。
    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"
    session_id: str
    task_intent: str
    approach: str
    quality_score: float
    key_insight: str | None = None
    timestamp: _dt.datetime
    score: float


class SearchAgentSkillItem(BaseModel):
    """Agent skill hit — always agent-scoped."""

    # agent_skill 也是 agent owner 分区下的顶层返回对象，但它表达的是从案例中沉淀出的
    # 可复用能力。source_case_ids 保留它与 agent_case 的血缘关系，方便客户端追溯技能来源。
    model_config = ConfigDict(extra="forbid")

    id: str
    agent_id: str
    app_id: str = "default"
    project_id: str = "default"
    name: str
    description: str
    content: str
    confidence: float
    maturity_score: float
    source_case_ids: list[str] = Field(default_factory=list)
    score: float


class UnprocessedMessageDTO(BaseModel):
    """A raw message still in the boundary-detection buffer.

    No extracted memcell yet, no owner inference yet (attribution
    happens at boundary detection). Returned by ``/search`` **only when**
    ``filters.session_id`` is present as a top-level eq predicate —
    unprocessed messages have no ``user_id`` / ``agent_id`` to filter
    on, so session is the only meaningful query dimension.
    """

    # unprocessed_messages 不是已抽取 memory，也没有进入 episode/profile/case/skill 任一索引。
    # 它代表 boundary detection 之前还在缓冲区里的原始消息。
    # 因为这类消息还没完成 owner inference，不能按 user_id/agent_id 精确过滤，
    # 所以 SearchManager 只有在 filters 顶层明确给出 session_id 时才会把它们附带返回。
    model_config = ConfigDict(extra="forbid")

    id: str
    """Original ``message_id`` from ``/add``."""
    app_id: str = "default"
    project_id: str = "default"
    session_id: str
    sender_id: str
    sender_name: str | None = None
    role: Literal["user", "assistant", "tool"]
    content: str | list[dict[str, object]]
    """``str`` for the single-text-item shorthand; ``list`` of opaque
    objects for the original multi-modal payload (mirrors
    ``MessageItem.content`` from the /add side)."""
    timestamp: _dt.datetime
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None


# ── Response envelope ────────────────────────────────────────────────────


class SearchData(BaseModel):
    """Body of ``response.data``.

    All five arrays are always present so client code can iterate without
    branching on ``owner_type``. Routes not applicable to the request's
    owner type stay as ``[]``. ``unprocessed_messages`` is filled only
    when ``filters.session_id`` is present as a top-level eq scalar —
    in-flight buffer rows are scope-tagged but unattributed (no
    ``user_id``), so session is the only meaningful query dimension.
    """

    # SearchData 是响应稳定性的核心：不根据 owner_type 动态改变字段集合。
    # 这样客户端可以永远读取 data.episodes、data.agent_cases 等数组，而不用先判断字段是否存在。
    # SearchManager 只负责根据 owner_type 填充适用数组；不适用的数组保持默认空列表。
    model_config = ConfigDict(extra="forbid")

    # user owner 分支会填 episodes；include_profile=true 时还会填 profiles。
    # episode 内部可以嵌套 atomic_facts，但这里没有顶层 atomic_facts 数组，
    # 这是公共契约刻意做出的限制。
    episodes: list[SearchEpisodeItem] = Field(default_factory=list)
    profiles: list[SearchProfileItem] = Field(default_factory=list)

    # agent owner 分支会填 agent_cases 和 agent_skills。
    # 它们不会出现在 user owner 搜索结果中，但字段仍然存在并保持 []。
    agent_cases: list[SearchAgentCaseItem] = Field(default_factory=list)
    agent_skills: list[SearchAgentSkillItem] = Field(default_factory=list)

    # unprocessed_messages 是附加诊断/补全信息，不属于已索引 memory 的召回结果。
    # 它只有在顶层 session_id 过滤存在时才有意义，否则无法确定缓冲区消息的查询范围。
    unprocessed_messages: list[UnprocessedMessageDTO] = Field(default_factory=list)
    """In-flight messages still in the boundary-detection buffer for
    the ``filters.session_id`` (if supplied as a top-level eq scalar);
    otherwise stays empty."""


class SearchResponse(BaseModel):
    """Top-level response envelope."""

    # 最外层只包 request_id 和 data。
    # request_id 用于链路追踪 / 日志关联；真正的业务数组全部放在 data 内部。
    model_config = ConfigDict(extra="forbid")

    request_id: str
    data: SearchData
