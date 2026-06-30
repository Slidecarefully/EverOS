"""User memory pipeline — per-sender Episode fan-out on pre-cut cells.

Cells / memcell_ids / message_id-mapping / sender lists are produced by
:mod:`everos.service._boundary` (which also writes the single
``memcell`` sqlite row per cell). This pipeline only handles the
user-perspective output: Episode md + ``UserPipelineStarted`` emit (one
per cell, fired at the start of ``run`` so atomic_fact / foresight /
clustering strategies run in parallel with the in-pipeline Episode work).

Run inside ``service.memorize`` via ``asyncio.gather`` alongside
:class:`AgentMemoryPipeline` (the latter only in ``mode="agent"``).
"""

from __future__ import annotations

# 这个模块位于 boundary stage 之后。
# boundary stage 已经负责把消息切成 MemCell，并且为每个 cell 写入统一的 memcell ledger。
# 因此这里不再做切分、不再生成 memcell_id，而是专注于“用户视角”的 Episode 生成和落盘。

# TYPE_CHECKING 用来避免运行时导入只用于类型标注的对象。
# 这些对象来自 infra 或 LLM 协议层，运行时提前导入可能增加依赖负担或形成循环引用。
from typing import TYPE_CHECKING

# AlgoMemCell 是 everalgo 层产出的 cell 类型。
# 这里用别名强调它来自算法层，而不是 everos.memory 里的领域对象。
from everalgo.types import MemCell as AlgoMemCell

# EpisodeExtractor 是用户记忆算法的核心组件。
# 它接收一个已经切好的 MemCell，并用 LLM 抽取出适合写入用户记忆的 Episode 叙事。
from everalgo.user_memory import EpisodeExtractor

# 时间工具用于把 Episode 中的 timestamp 标准化为 markdown inline metadata 需要的 ISO 格式。
# from_timestamp 处理毫秒时间戳，to_iso_format 负责最终字符串化。
from everos.component.utils.datetime import from_timestamp, to_iso_format

# logger 用于记录缺少 LLM client 等防御性跳过场景。
# 这些场景不一定是代码错误，但需要在运行日志中可观察。
from everos.core.observability.logging import get_logger

# Episode 是 everos 领域层的用户记忆对象。
# IngestResult 带有 session/app/project 等上下文。
# PipelineOutcome 是 pipeline 对外返回的统一结果结构。
from everos.memory import Episode, IngestResult, PipelineOutcome

# 事件用于驱动 OME / cascade / 异步策略：
# - UserPipelineStarted：在 Episode 抽取前发出，让 atomic_fact、foresight、clustering 等并行开始
# - EpisodeExtracted：Episode 写入 markdown 后发出，让后续索引或 cascade 能消费结果
from everos.memory.events import EpisodeExtracted, UserPipelineStarted

# PromptLoader 负责加载 episode extraction prompt。
# 把 prompt 从 pipeline 逻辑中抽离出来，便于按环境、语言或实验版本替换。
from everos.memory.prompt_slots import PromptLoader

# 下面这些类型只在类型检查时使用：
# - LLMClient：EpisodeExtractor 依赖的模型客户端
# - OfflineEngine：事件发射引擎
# - EpisodeWriter：markdown 持久化写入器
if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

    from everos.infra.ome.engine import OfflineEngine
    from everos.infra.persistence.markdown import EpisodeWriter

# 模块级 logger 带上当前模块名，方便日志定位到 user memory pipeline。
logger = get_logger(__name__)

# pipeline track 名称。
# 这个 track 用于 PipelineOutcome，也可用于上层区分 user_memory 和 agent_memory 的处理结果。
_TRACK = "user_memory"


class UserMemoryPipeline:
    """Per-sender Episode extraction on a list of pre-cut MemCells."""

    def __init__(
        self,
        episode_writer: EpisodeWriter,
        prompt_loader: PromptLoader,
        llm_client: LLMClient | None,
        engine: OfflineEngine,
    ) -> None:
        # EpisodeExtractor requires `llm` at construction. Skip-with-warning
        # when no LLM is configured — the boundary stage will have skipped
        # the run already; this is just a defensive null check.
        # EpisodeExtractor 需要在构造时拿到 LLM client。
        # 这里允许 llm_client 为 None，是为了让 pipeline 对异常配置更稳健：
        # 上游通常已经会因为无 LLM 跳过 boundary/run，但本类仍做一次防御。
        self._ep_ext = (
            EpisodeExtractor(llm=llm_client) if llm_client is not None else None
        )

        # episode_writer 负责把 Episode 写成 markdown entry。
        # pipeline 本身只决定写什么内容，不直接关心 markdown 文件路径和 entry id 的生成细节。
        self._episode_writer = episode_writer

        # prompt_loader 提供 episode_extract prompt。
        # 这样 prompt 的选择与 pipeline 的控制流保持解耦。
        self._prompt_loader = prompt_loader

        # engine 用于发出 OME 事件。
        # 本 pipeline 的很多后续动作不是同步写在这里，而是通过事件交给异步策略处理。
        self._engine = engine

    async def run(
        self,
        ingested: IngestResult,
        cells: list[AlgoMemCell],
        memcell_ids: list[str],
        per_cell_all_senders: list[list[str]],
    ) -> PipelineOutcome:
        """Emit UserPipelineStarted per cell, then extract Episodes + write md."""
        # cells 已经由 boundary stage 预先切好。
        # 如果没有 cell，说明当前还处于累积状态，没有可抽取的用户 Episode。
        if not cells:
            return PipelineOutcome(track=_TRACK, status="accumulated", message_count=0)

        # 没有 EpisodeExtractor 就无法调用 LLM 抽取 Episode。
        # 这里返回 skipped 而不是抛异常，是因为缺少 LLM 通常属于部署/配置问题，
        # 上游可以通过 PipelineOutcome 和 warning 日志感知，而不会让整条 memorize 流程崩掉。
        if self._ep_ext is None:
            logger.warning(
                "user_memory_pipeline_no_llm_client",
                extra={"session_id": ingested.session_id, "cells": len(cells)},
            )
            return PipelineOutcome(track=_TRACK, status="skipped", message_count=0)

# Emit upfront so OME-async strategies such as atomic_fact and foresight
# can start in parallel with the in-pipeline Episode extraction work.
# These strategies consume the MemCell directly and do not depend on
# Episode output.
#
# Profile clustering is different: it is NOT triggered here. It runs only
# after an Episode has been extracted and written, via the EpisodeExtracted
# event below, because clustering embeds episode_text while storing
# memcell_id as the cluster member.
# 
# 先发 UserPipelineStarted，再做 Episode 抽取。
# 这一步的顺序很关键：atomic_fact / foresight 等策略只依赖 MemCell，
# 不依赖后面生成的 Episode，所以可以提前启动并与 Episode 抽取并行。
#
# 但 profile clustering 不属于这里提前启动的策略。
# profile clustering 需要等待 EpisodeExtracted，因为它要用 ep.episode 做 embedding，
# 再把对应的 memcell_id 加入 owner 的 cluster。
        
        for cell, memcell_id in zip(cells, memcell_ids, strict=True):
            # strict=True 保证 cells 与 memcell_ids 数量完全一致。
            # 如果 boundary stage 输出错位，这里会立即失败，避免事件带错 memcell_id。
            await self._emit_pipeline_started(
                memcell_id=memcell_id,
                session_id=ingested.session_id,
                app_id=ingested.app_id,
                project_id=ingested.project_id,
                cell=cell,
            )

        # 加载 Episode 抽取 prompt。
        # 这个 prompt 会传给 everalgo.user_memory.EpisodeExtractor，
        # 控制 LLM 如何把一个 MemCell 写成用户可读的 episode narrative。
        episode_prompt = self._prompt_loader.load("episode_extract")

        # md_paths 收集本次写入过的 markdown 文件路径，用于 PipelineOutcome 返回给调用方或日志。
        md_paths: list[str] = []

        # msg_count 统计本次 user_memory pipeline 实际覆盖的 cell item 数。
        # 它不是 Episode 数量；一个 cell 可能 fan-out 成多个用户 Episode。
        msg_count = 0

        # 三个列表按 cell 索引并行：
        # - cell：要抽取的记忆单元
        # - memcell_id：该 cell 在边界阶段生成的统一父 ID
        # - all_senders：cell 内所有参与者，用于 Episode 的 sender_ids 审计字段
        for cell, memcell_id, all_senders in zip(
            cells, memcell_ids, per_cell_all_senders, strict=True
        ):
            # 每个 cell 的 item 数加入总消息计数。
            # 这里统计的是 MemCell 内部 item，而不是原始 DTO 消息数。
            msg_count += len(cell.items)

            # user_senders 只保留 role=user 的 sender_id。
            # user memory 是“按用户视角”fan-out，因此没有用户发言的 cell 不生成用户 Episode。
            user_senders = _unique_user_senders(cell)
            if not user_senders:
                continue

            # One generic LLM call per cell (sender_id=None drives the algo's
            # whole-memcell EPISODE_GENERATION_PROMPT — explicitly cheaper
            # than the per-user fan-out per the algo's docstring). Fan-out
            # is then md-only: every user sender owns a copy of the same
            # narrative under its own owner_id path.
            # 这里刻意每个 cell 只调用一次 LLM，而不是每个 user_sender 调一次。
            # sender_id=None 表示让算法基于整个 MemCell 生成一份通用 episode narrative。
            # 后面的 per-user fan-out 只是在 markdown/owner_id 层复制归属，避免多用户 cell 产生重复 LLM 成本。
            algo_ep = await self._ep_ext.aextract(
                cell, sender_id=None, prompt=episode_prompt
            )

            # 对同一个 cell 中的每个用户 sender，都写一份以该 sender 为 owner 的 Episode。
            # 这样多用户对话可以让每个用户视角都拥有对应的记忆条目。
            for sender_id in user_senders:
                # Episode.from_algo 把算法层 episode 转成 everos 领域层 Episode。
                # owner_id 是当前 fan-out 的用户；sender_ids 是该 cell 的全部参与者；
                # parent_id 指向 boundary stage 生成的 memcell_id，用于建立 episode → memcell 的回链。
                ep = Episode.from_algo(
                    algo_ep,
                    owner_id=sender_id,
                    session_id=ingested.session_id,
                    sender_ids=all_senders,
                    parent_id=memcell_id,
                )

                # markdown writer 需要 inline metadata 和 sections 两部分。
                # 该转换依赖 everos.memory.Episode，所以放在 memory pipeline 层，而不是 infra writer 层。
                inline, sections = _episode_to_entry_body(ep)

                # 将 Episode 追加写入 owner_id 对应的 markdown 文件。
                # app_id/project_id 保证同一个用户在不同应用或项目下的记忆文件互相隔离。
                eid = await self._episode_writer.append_entry(
                    ep.owner_id,
                    inline=inline,
                    sections=sections,
                    app_id=ingested.app_id,
                    project_id=ingested.project_id,
                )

                # 根据 writer 返回的 entry id 反查实际文件路径。
                # 一个文件可能包含多个 entry，这里收集路径主要用于 outcome 可观察性。
                md_paths.append(
                    str(
                        self._episode_writer.path_for(
                            ep.owner_id,
                            eid.date,
                            app_id=ingested.app_id,
                            project_id=ingested.project_id,
                        )
                    )
                )

                # Episode 已经写入 markdown 后，再发 EpisodeExtracted。
                # 这样后续 cascade/索引策略消费事件时，可以假设 markdown entry 已经存在。
                await self._engine.emit(
                    EpisodeExtracted(
                        memcell_id=memcell_id,
                        episode_entry_id=eid.format(),
                        episode_text=ep.episode,
                        episode_timestamp_ms=ep.timestamp,
                        owner_id=ep.owner_id,
                        app_id=ingested.app_id,
                        project_id=ingested.project_id,
                    )
                )

        # 返回 user_memory pipeline 的处理结果。
        # status 固定为 extracted，表示 pipeline 已经完成对输入 cells 的处理；
        # 即使某些 cell 没有 user_senders 而没有写出 Episode，整体流程仍然已执行完成。
        return PipelineOutcome(
            track=_TRACK,
            status="extracted",
            message_count=msg_count,
            extracted_md_paths=md_paths,
        )

    async def _emit_pipeline_started(
        self,
        memcell_id: str,
        session_id: str,
        app_id: str,
        project_id: str,
        cell: AlgoMemCell,
    ) -> None:
        # UserPipelineStarted 是以 memcell 为粒度的启动事件。
        # 它不等待 Episode 生成完成，目的是让其他只依赖 MemCell 的策略尽快并行执行。
        await self._engine.emit(
            UserPipelineStarted(
                memcell_id=memcell_id,
                session_id=session_id,
                app_id=app_id,
                project_id=project_id,
                memcell=cell,
            )
        )


# ── Helpers ───────────────────────────────────────────────────────────────


def _unique_user_senders(cell: AlgoMemCell) -> list[str]:
    """Distinct role=user sender_ids in a cell, preserving order.

    Drives per-sender Episode fan-out: each user perspective gets its own
    Episode for the cell. Skips non-``ChatMessage`` items (agent
    trajectories' ``ToolCallResult`` has no ``role``).
    """
    # 使用 list 保序去重。
    # 对同一个 cell，用户第一次出现的顺序就是 fan-out 的顺序。
    senders: list[str] = []

    # cell.items 在 agent 模式下可能混入 ToolCallRequest/ToolCallResult。
    # 这些 item 不一定有 role 或 sender_id，因此必须使用 getattr 安全访问。
    for item in cell.items:
        # user memory 只为 role=user 的消息生成 owner 视角。
        # assistant 消息、tool result、tool request 都不是 Episode 的 owner。
        if getattr(item, "role", None) != "user":
            continue

        # sender_id 是实际 owner_id 的来源。
        # 如果缺失，就不能生成稳定的用户归属，因此跳过。
        sid = getattr(item, "sender_id", None)

        # 去重但保留第一次出现的顺序。
        # 这样同一用户在一个 cell 中多次发言，也只会生成一份 Episode。
        if sid and sid not in senders:
            senders.append(sid)

    return senders


def _episode_to_entry_body(
    episode: Episode,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain Episode into ``(inline, sections)`` for md rendering.

    Lives in the pipeline (memory) layer rather than the writer (infra)
    because it depends on :class:`everos.memory.Episode` — infra is not
    allowed to import memory per the layered architecture contract.

    Inline persists the audit / scope fields cascade needs to rebuild
    the LanceDB row: ``owner_id`` / ``session_id`` / ``timestamp`` /
    ``parent_id`` / ``sender_ids``. ``parent_id`` is the source memcell
    id (minted by the boundary stage), and the cascade handler reads it
    back so the LanceDB ``episode`` row keeps its back-link to the source.

    The md entry's ``entry_id`` (managed by the chassis writer) is the
    single source of *entry* identity; cascade derives a global episode
    id from ``<owner_id>_<entry_id>`` on the fly.
    """
    # Episode.timestamp 可能是算法层传来的毫秒整数，也可能已经是字符串/时间对象。
    # 如果是 int，就先转 datetime 再转 ISO，保证 markdown inline metadata 中时间格式稳定。
    ts_iso = (
        to_iso_format(from_timestamp(episode.timestamp))
        if isinstance(episode.timestamp, int)
        else str(episode.timestamp)
    )

    # inline 是写在 markdown entry 头部的结构化元数据。
    # 这些字段用于后续 cascade 重建 LanceDB row、追踪归属和建立 source back-link。
    inline: dict[str, object] = {
        "owner_id": episode.owner_id,
        "session_id": episode.session_id,
        "timestamp": ts_iso,
        "parent_type": "memcell",
        "parent_id": episode.parent_id,
    }

    # sender_ids 是 cell 中所有参与者。
    # 只有非空时才写入，避免 markdown metadata 中出现无意义的空字段。
    if episode.sender_ids:
        inline["sender_ids"] = list(episode.sender_ids)

    # extra 保存 Episode 中除核心 inline 字段和正文外的其他字段。
    # 这些字段会进一步拆成 markdown sections，例如 Subject / Summary。
    extra = episode.model_dump(
        exclude={
            "owner_id",
            "episode",
            "timestamp",
            "session_id",
            "sender_ids",
            "parent_id",
        }
    )

    # subject 和 summary 是常见的可读段落字段。
    # 从 extra 中 pop 出来，避免后续如果扩展处理 extra 时重复写入。
    subject = extra.pop("subject", None)
    summary = extra.pop("summary", None)

    # sections 是 markdown entry 的正文分区。
    # 与 inline metadata 不同，sections 面向人类阅读和语义内容展示。
    sections: dict[str, str] = {}

    # 只有存在 subject 时才写 Subject 区块。
    # 这让输出 markdown 更紧凑，也避免空标题。
    if subject:
        sections["Subject"] = str(subject)

    # 只有存在 summary 时才写 Summary 区块。
    if summary:
        sections["Summary"] = str(summary)

    # Content 是 Episode 的核心叙事正文，始终写入。
    sections["Content"] = episode.episode

    # writer 接下来会把 inline 和 sections 渲染成最终 markdown entry。
    return inline, sections
