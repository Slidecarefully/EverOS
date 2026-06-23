"""Memorize use case — ingest + boundary + dual pipeline dispatch.

End-to-end orchestration:

    POST /api/v1/memory/add { session_id, messages[] }
        → ingest.process → IngestResult
        → _boundary.prepare_cells(mode=settings.memorize.mode) → cells
        → asyncio.gather(
            UserMemoryPipeline.run(cells, ...),
            AgentMemoryPipeline.run(cells, ...) if mode == "agent",
          )
        → merge outcome.status → {message_count, status}

The boundary stage owns buffer / merge / boundary / tail — so the same
``cells`` feed both pipelines in agent mode (chat mode skips the agent
pipeline entirely).

Lazy singletons: writer / loader / pipelines / LLM client are all
constructed on first use (service module imports run before lifespan
resolves the memory-root and reads env vars).
"""

from __future__ import annotations

# 这个模块是 memorize 用例的服务编排层。
# 它不直接做消息标准化、边界检测、Episode/Agent 抽取，也不直接写最终记忆文件；
# 它负责把这些阶段按正确顺序串起来，并处理并发、锁、lazy singleton 和结果状态合并。

# asyncio 用在两个地方：
# 1. session lock 外层的 timeout，防止某次 /add 长时间占住同一会话；
# 2. 在 agent 模式下并发运行 user pipeline 和 agent pipeline。
import asyncio

# Path 用于定位随包发布的 config/ 目录，也就是 prompt slots 所在目录。
from pathlib import Path

# Any 用于承接 settings 中的 boundary_cfg；
# Literal 用于约束 mode/status 这类有限字符串，避免调用链里出现非法模式。
from typing import Any, Literal

# MemorizeResult 是对路由层返回值的 Pydantic DTO。
# route 可以直接序列化它，而不需要理解内部 PipelineOutcome 或 BoundaryOutcome。
from pydantic import BaseModel

# get_llm_client 是 LLM client 的统一获取入口。
# boundary detection、user pipeline、策略抽取等都最终依赖它，但本模块只在需要时调用。
from everos.component.llm import get_llm_client

# settings 决定 memorize 的运行模式、边界检测硬限制和 session lock timeout。
# 这些配置在每次 memorize 调用开始时读取，保证运行时配置变化可以被反映。
from everos.config import load_settings

# logger 用于本模块的运行日志。
# 当前代码里没有直接记录日志，但保留模块级 logger 便于后续扩展和排查。
from everos.core.observability.logging import get_logger

# MemoryRoot 提供运行时记忆根目录。
# EpisodeWriter 和 OfflineEngine 的 jobstore/config 都要基于这个根目录初始化。
from everos.core.persistence import MemoryRoot

# OMEConfig 是 OfflineEngine 的配置对象。
# 它指定 jobstore 数据库位置和 OME 配置文件路径。
from everos.infra.ome.config import OMEConfig

# OfflineEngine 是离线事件策略引擎。
# user pipeline 会通过它发出 UserPipelineStarted / EpisodeExtracted 等事件，
# 这些事件再触发 atomic facts、foresight、profile clustering 等异步策略。
from everos.infra.ome.engine import OfflineEngine

# EpisodeWriter 是 user memory pipeline 写 Episode markdown 的底层 writer。
# 它被作为依赖注入到 UserMemoryPipeline，而不是在 pipeline 内部硬编码创建。
from everos.infra.persistence.markdown import EpisodeWriter

# ingest_process 是 memorize 流程的第一步：
# 把 API/service 层传入的 raw payload 归一化成 IngestResult 和 CanonicalMessage 列表。
from everos.memory.extract.ingest import process as ingest_process

# UserMemoryPipeline 负责用户视角输出；
# AgentMemoryPipeline 负责 agent 模式下的 agent case/skill 等轨迹输出。
# 两者都消费 boundary stage 已切好的同一批 cells。
from everos.memory.extract.pipeline import (
    AgentMemoryPipeline,
    UserMemoryPipeline,
)

# PromptLoader 负责加载 boundary detection、episode extraction 等 prompt slot。
# 它基于 config root 初始化，并作为依赖传给 boundary 和 user pipeline。
from everos.memory.prompt_slots import PromptLoader

# 这些导入不是为了在本模块中直接调用策略函数，
# 而是为了在 _get_engine 中注册到 OfflineEngine。
# 一旦 user/agent pipeline emit 事件，engine 才知道哪些 offline_strategy 应该被触发。
from everos.memory.strategies import (
    extract_agent_case,
    extract_agent_skill,
    extract_atomic_facts,
    extract_foresight,
    extract_user_profile,
    trigger_profile_clustering,
    trigger_skill_clustering,
)

# prepare_cells 是 boundary stage 的入口。
# 它负责 buffer 读取、旧尾巴和新消息合并、边界检测、memcell ledger 写入、tail 回写。
from everos.service._boundary import prepare_cells

# session lock 保证同一 session 的多个 /add 请求串行执行。
# 这对 boundary buffer 尤其重要，因为并发读-改-写 tail 会导致 lost update。
from everos.service._session_lock import get_session_lock

# 模块级 logger 带上当前模块名，方便日志系统定位到 memorize service。
logger = get_logger(__name__)


class MemorizeResult(BaseModel):
    """What memorize returns to the caller (route serialises it)."""

    # message_count 是本次请求中 raw payload 的消息数量。
    # 注意它不是 boundary 实际切出的 cell 数，也不是 pipeline 抽取出的记忆条目数。
    message_count: int

    # 对外只暴露 accumulated / extracted 两态。
    # 内部 pipeline 可能有 skipped，但 _merge_status 会把 skipped 合并到 accumulated 语义里。
    status: Literal["accumulated", "extracted"]


# Lazy singletons ────────────────────────────────────────────────────────────

# 这些对象都使用模块级 lazy singleton，而不是 import 时立即创建。
# 原因是 service 模块可能早于应用 lifespan 执行；那时 env、memory root、配置文件可能还没准备好。
# 因此只有真正进入 memorize 流程时，才按需构造 writer、loader、pipeline 和 OME engine。

_episode_writer: EpisodeWriter | None = None
_prompt_loader: PromptLoader | None = None
_user_pipeline: UserMemoryPipeline | None = None
_agent_pipeline: AgentMemoryPipeline | None = None
_ome_engine: OfflineEngine | None = None


def _config_root() -> Path:
    """Return the directory holding bundled prompt slots (``config/``)."""
    # ``src/everos/config/`` ships in the wheel alongside this service module.
    # 当前文件位于 service 模块下，parent.parent 回到 everos 包目录，
    # 再拼出随包发布的 config/ 目录，供 PromptLoader 读取 prompt slots。
    return Path(__file__).resolve().parent.parent / "config"


def _get_episode_writer() -> EpisodeWriter:
    # EpisodeWriter 写入用户 Episode markdown。
    # 它依赖 MemoryRoot.default()，所以不能太早构造；否则应用启动前的路径配置可能尚未生效。
    global _episode_writer

    # 第一次访问时创建，后续复用同一个 writer。
    # 这样既避免重复初始化，也让 UserMemoryPipeline 使用稳定的持久化依赖。
    if _episode_writer is None:
        _episode_writer = EpisodeWriter(MemoryRoot.default())

    return _episode_writer


def _get_prompt_loader() -> PromptLoader:
    # PromptLoader 负责读取 boundary_detection、episode_extract 等 prompt。
    # 它基于 bundled config root 初始化，因此也使用 lazy singleton。
    global _prompt_loader

    if _prompt_loader is None:
        _prompt_loader = PromptLoader(_config_root())

    return _prompt_loader


def _get_user_pipeline() -> UserMemoryPipeline:
    # UserMemoryPipeline 是用户记忆侧的主 pipeline。
    # 它需要 episode writer、prompt loader、LLM client 和 OME engine。
    global _user_pipeline

    if _user_pipeline is None:
        _user_pipeline = UserMemoryPipeline(
            episode_writer=_get_episode_writer(),
            prompt_loader=_get_prompt_loader(),
            llm_client=get_llm_client(),
            engine=_get_engine(),
        )

    return _user_pipeline


def _get_agent_pipeline() -> AgentMemoryPipeline:
    # AgentMemoryPipeline 只在 settings.memorize.mode == "agent" 时使用。
    # 它主要通过 engine 发事件或写 agent 相关输出，不需要 EpisodeWriter。
    global _agent_pipeline

    if _agent_pipeline is None:
        _agent_pipeline = AgentMemoryPipeline(engine=_get_engine())

    return _agent_pipeline


def _get_engine() -> OfflineEngine:
    """Return the singleton OfflineEngine; constructed + registered on first call.

    Lifecycle (start/stop) is wired by ``OmeLifespanProvider``.
    """
    # OfflineEngine 是整个 memory 后台策略的事件调度中心。
    # 它必须在第一次使用时完成策略注册，否则 pipeline emit 出来的事件不会触发后续策略。
    global _ome_engine

    if _ome_engine is None:
        # MemoryRoot.default() 提供 OME jobstore 和 config 路径。
        # 这和 markdown writer 使用同一个 memory root，保证所有持久化状态落在同一空间。
        root = MemoryRoot.default()
        jobstore_path = root.ome_db

        # jobstore 是 OME 的本地数据库文件。
        # 创建父目录可以避免 engine 初始化时因为路径不存在而失败。
        jobstore_path.parent.mkdir(parents=True, exist_ok=True)

        # 构造 OfflineEngine，但不在这里启动生命周期。
        # start/stop 由 OmeLifespanProvider 接管，本函数只负责 singleton 创建和策略注册。
        engine = OfflineEngine(
            config=OMEConfig(
                jobstore_path=jobstore_path,
                config_path=root.ome_config,
            )
        )

        # 注册 user memory 相关策略。
        # UserPipelineStarted 会触发 atomic facts / foresight；
        # EpisodeExtracted 会触发 profile clustering；
        # ProfileClusterUpdated 会触发 profile extraction。
        engine.register(extract_atomic_facts)
        engine.register(extract_foresight)

        # 注册 agent memory 相关策略。
        # agent pipeline 发出的事件会驱动 case/skill 抽取与 skill clustering。
        engine.register(extract_agent_case)
        engine.register(trigger_skill_clustering)
        engine.register(extract_agent_skill)

        # 注册 profile 更新链路。
        # Episode 写入后先聚类，再根据更新后的 cluster 重新抽取用户 profile。
        engine.register(trigger_profile_clustering)
        engine.register(extract_user_profile)

        # 所有策略注册完成后再赋给模块级变量，保证后续拿到的是完整可用的 engine。
        _ome_engine = engine

    return _ome_engine


# Public entry ───────────────────────────────────────────────────────────────


async def memorize(
    payload: dict[str, Any],
    *,
    is_final: bool = False,
) -> MemorizeResult:
    """Execute one add cycle. Dispatched concurrently across pipelines.

    Args:
        payload: ``{"session_id", "messages": [...]}`` — entrypoints DTO
            dumped to dict.
        is_final: ``True`` only for flush (algo guarantees ``tail=[]``).

    Concurrency: serialised per ``session_id`` via
    :func:`everos.service._session_lock.get_session_lock`. The lock
    spans the entire read-merge-boundary-write cycle so concurrent /add
    calls on the same session cannot lose-update each other's tail.
    An outer ``asyncio.timeout`` (configured by
    ``settings.memorize.session_lock_timeout_seconds``) ensures a stuck
    LLM cannot hold the lock indefinitely — on timeout the task is
    cancelled and ``async with`` auto-releases the lock.
    """
    # 每次调用读取 settings，而不是在模块 import 时缓存。
    # 这样 memorize.mode、boundary hard limit、lock timeout 等运行配置可以在应用生命周期内被正确解析。
    settings = load_settings()

    # mode 决定 boundary 和 pipeline 分发方式：
    # - chat：只跑 user pipeline，且 boundary 过滤 tool rows
    # - agent：boundary 保留完整轨迹，并同时跑 user + agent pipelines
    mode = settings.memorize.mode

    # boundary_cfg 提供 hard_token_limit / hard_msg_limit。
    # 这些限制传给 boundary detection，防止超长对话一次性进入 LLM。
    boundary_cfg = settings.boundary_detection

    # session_id 是锁粒度。
    # 同一个 session 的 tail buffer 必须串行更新；不同 session 则可以并行处理。
    session_id = payload["session_id"]

    # 外层 timeout 保护整个 critical section。
    # 如果 ingest、boundary 或 pipeline 中的 LLM 调用卡住，timeout 会取消任务并释放 async context。
    async with asyncio.timeout(settings.memorize.session_lock_timeout_seconds):
        # session lock 包住完整的 ingest → boundary → pipeline dispatch 流程。
        # 其中最关键的是 boundary 的 read-merge-write tail；如果并发执行，会覆盖彼此的 buffer。
        async with get_session_lock(session_id):
            return await _memorize_locked(
                payload,
                mode=mode,
                boundary_cfg=boundary_cfg,
                is_final=is_final,
            )


async def _memorize_locked(
    payload: dict[str, Any],
    *,
    mode: Literal["chat", "agent"],
    boundary_cfg: Any,
    is_final: bool,
) -> MemorizeResult:
    """Inner critical section — runs under the per-session lock."""
    # 进入锁内后，第一步是 ingest。
    # ingest_process 把 service DTO 形态的 payload 转成 IngestResult，
    # 包括 canonical messages、app_id/project_id/session_id 等上下文。
    ingested = await ingest_process(payload)

    # 第二步是 boundary stage。
    # 它会把历史 buffer 和本次 ingested messages 合并，尝试切出 MemCell，
    # 同时写入 memcell ledger，并把未闭合的 tail 回写 buffer。
    boundary = await prepare_cells(
        ingested,
        mode=mode,
        is_final=is_final,
        llm_client=get_llm_client(),
        prompt_loader=_get_prompt_loader(),
        hard_token_limit=boundary_cfg.hard_token_limit,
        hard_msg_limit=boundary_cfg.hard_msg_limit,
    )

    # 如果 boundary 没有切出任何 cell，就没有必要运行 user/agent pipeline。
    # 这通常表示消息仍在累积中，或者 boundary 因缺少条件选择 skipped。
    if not boundary.cells:
        # Nothing went past the boundary stage — no pipelines to dispatch.
        # 对外 message_count 使用 raw payload 的 messages 数量，保持 API 响应和调用输入一致。
        # status 通过 _merge_status 将 boundary.status 和 agent skipped 语义合并为 accumulated/extracted。
        return MemorizeResult(
            message_count=len(payload.get("messages", [])),
            status=_merge_status(boundary.status, "skipped"),
        )

    # 有 cells 时，user pipeline 一定会运行。
    # 它消费 boundary 输出的 cells、memcell_ids 和 sender 列表，
    # 生成 per-user Episode，并发出 UserPipelineStarted / EpisodeExtracted 等事件。
    user_task = _get_user_pipeline().run(
        ingested,
        cells=boundary.cells,
        memcell_ids=boundary.memcell_ids,
        per_cell_all_senders=boundary.per_cell_all_senders,
    )

    # agent 模式下，同一批 boundary cells 也要交给 AgentMemoryPipeline。
    # 这体现了本文件 docstring 里的 dual pipeline dispatch：一次切分，多路消费。
    if mode == "agent":
        agent_task = _get_agent_pipeline().run(
            ingested,
            cells=boundary.cells,
            memcell_ids=boundary.memcell_ids,
        )

        # user 和 agent pipeline 互不依赖，因此可以并发执行。
        # 两者共享同一批 memcell_id，避免各自重复写 memcell ledger 或产生 ID 分叉。
        user_outcome, agent_outcome = await asyncio.gather(user_task, agent_task)

        # 任一路 extracted 都表示本次 memorize 对外可视为 extracted。
        merged_status = _merge_status(user_outcome.status, agent_outcome.status)
    else:
        # chat 模式只运行 user pipeline。
        # agent 侧视为 skipped，再用统一的 _merge_status 逻辑合并结果。
        user_outcome = await user_task
        merged_status = _merge_status(user_outcome.status, "skipped")

    # 最终返回 route 层需要的简化结果。
    # 这里不暴露 md_paths、memcell_ids 或单个 pipeline outcome，避免 API 层耦合内部流水线细节。
    return MemorizeResult(
        message_count=len(payload.get("messages", [])),
        status=merged_status,
    )


def _merge_status(
    user: Literal["accumulated", "extracted", "skipped"],
    agent: Literal["accumulated", "extracted", "skipped"],
) -> Literal["accumulated", "extracted"]:
    """Either ``extracted`` wins; otherwise ``accumulated``."""
    # 对外状态只关心“有没有任何 pipeline 产生了抽取结果”。
    # 因此 extracted 是胜出态：只要 user 或 agent 任一侧 extracted，就返回 extracted。
    if user == "extracted" or agent == "extracted":
        return "extracted"

    # accumulated 和 skipped 都表示没有产生新抽取结果。
    # 对调用方来说，它们都归并为 accumulated：消息可能正在等待更多上下文，或本轮没有可运行 pipeline。
    return "accumulated"
