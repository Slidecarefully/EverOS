"""Boundary stage — shared upstream step for the dual-pipeline memorize flow.

Owns the buffer / merge / boundary / tail-persistence sequence so the same
``cells`` feed both :class:`everos.memory.extract.pipeline.UserMemoryPipeline`
and :class:`everos.memory.extract.pipeline.AgentMemoryPipeline` (the
latter only runs when ``mode == "agent"``).

Mode dispatch:

- ``"chat"``  → :func:`everalgo.boundary.detect_boundaries` on a filtered
  ``ChatMessage`` list (tool rows / assistant-with-tool_calls dropped).
- ``"agent"`` → :class:`everalgo.agent_memory.AgentBoundaryDetector` on the
  full ``ConversationItem`` list (tool rows preserved).

Both paths share a single unprocessed-buffer track (``"memorize"``) because
boundary detection is single-pass; switching mode requires a fresh service
process (see ``settings.memorize.mode``).

The boundary stage also owns the **sqlite ``memcell`` ledger**: each cell
gets exactly one row regardless of mode (since the algorithm produces one
canonical cell). Downstream pipelines (user + agent) reference the same
``memcell_id``; PK collisions used to occur when each pipeline tried to
insert its own row per cell.
"""

from __future__ import annotations

# 这个模块处在 memory ingest 之后、具体 user/agent 抽取 pipeline 之前。
# 它的核心职责不是“抽取记忆”，而是先把一段连续对话切成边界明确的 MemCell，
# 并把还不足以切分的尾巴保存回 buffer，等下一批消息一起处理。

# json 主要用于把内部模型中的复杂字段序列化进 sqlite 表。
# 这里的持久化表不是最终语义记忆，而是边界阶段的中间账本和缓冲区。
import json

# uuid 用于生成 everos 自己拥有的 memcell_id。
# 这些 ID 会被 user pipeline 和 agent pipeline 共同引用，因此不能依赖下游各自生成。
import uuid

# TYPE_CHECKING 只在类型检查阶段导入较重或会造成循环依赖的类型。
# Literal 则限定 mode/status 的合法取值，NamedTuple 用于声明一个轻量但结构稳定的返回对象。
from typing import TYPE_CHECKING, Literal, NamedTuple

# AgentBoundaryDetector 处理 agent 模式。
# agent 模式必须保留工具调用轨迹，所以它接收的是完整 ConversationItem 序列，
# 而不是 chat 模式下过滤后的纯 ChatMessage 序列。
from everalgo.agent_memory import AgentBoundaryDetector

# detect_boundaries 是 chat 模式的边界检测入口。
# 它面向更普通的聊天消息，不保留 tool row，也会接收 hard token/msg limit。
from everalgo.boundary import detect_boundaries

# 这些是 everalgo 侧的 wire type。
# 本模块前半部分使用 everos.memory.CanonicalMessage 作为内部统一消息，
# 调用 boundary 算法前再映射成 everalgo 能理解的 ChatMessage / ConversationItem / ToolCall*。
from everalgo.types import (
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

# everos.memory.ToolCall 和 everalgo.types.ToolCall 名字相同但属于不同层。
# 这里用别名 AlgoToolCall 明确表示“即将传给算法侧的 tool call 结构”，
# 避免在 assistant/tool-call 映射时混淆两个模型。
from everalgo.types import ToolCall as AlgoToolCall

# from_timestamp / to_timestamp_ms 是两层之间的时间格式桥梁。
# sqlite/everos 模型里通常保存 datetime；everalgo 边界检测侧使用毫秒时间戳。
from everos.component.utils.datetime import from_timestamp, to_timestamp_ms

# logger 用来记录边界检测重试、缺少 LLM client 等运行态问题。
# 这些情况并不都适合直接抛给调用方，但需要能在日志中追踪。
from everos.core.observability.logging import get_logger

# sqlite persistence 层提供三类表/仓储：
# - UnprocessedBuffer：暂存还不能形成完整 cell 的消息尾巴
# - Memcell：边界阶段生成的 cell 账本
# - conversation_status_repo：记录会话处理进度时间戳
from everos.infra.persistence.sqlite import (
    Memcell,
    UnprocessedBuffer,
    conversation_status_repo,
    memcell_repo,
    unprocessed_buffer_repo,
)

# CanonicalMessage 是 ingest 之后的标准消息格式。
# IngestResult 是上游 ingest 阶段的输出，本模块从这里继续处理。
# ToolCall 是 everos 内部的工具调用模型，主要用于 buffer 反序列化和 agent 映射。
from everos.memory import CanonicalMessage, IngestResult, ToolCall

# 这些类型只用于函数签名。
# 放到 TYPE_CHECKING 下面可以避免运行时导入带来的循环依赖或额外初始化。
if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

    from everos.memory.prompt_slots import PromptLoader

# 模块级 logger 统一带上当前模块名，方便排查 boundary 阶段的问题。
logger = get_logger(__name__)

# memorize 是该阶段共享的 track 名称。
# 它同时用于 unprocessed buffer 和 memcell ledger，因为二者描述的是同一条边界检测流水线。
_TRACK = "memorize"
"""Shared track used for both the unprocessed-buffer and the memcell
ledger — boundary detection is mode-dispatched but single-pass, so it
does not need per-pipeline separation."""

# raw_type 写进 memcell row，用来说明 cell 原始语义来源。
# chat 模式把输入视为普通 Conversation；agent 模式把输入视为 AgentTrajectory。
# 下游可以根据 raw_type 理解 payload 的上下文，而不用反推 mode。
_RAW_TYPE_BY_MODE: dict[str, str] = {
    "chat": "Conversation",
    "agent": "AgentTrajectory",
}


# Mode 控制边界检测时如何过滤和映射消息。
# Status 则描述本次 prepare_cells 的结果：
# accumulated 表示继续等待更多消息，extracted 表示已切出 cell，skipped 表示本次不处理。
Mode = Literal["chat", "agent"]
Status = Literal["accumulated", "extracted", "skipped"]


class BoundaryOutcome(NamedTuple):
    """Result handed to the dual pipelines.

    Lists are parallel: index ``i`` describes cell ``i``.
    ``memcell_ids`` are minted here and shared across both pipelines
    (Episode.parent_id / UserPipelineStarted.memcell_id both reference
    the same id — single sqlite ``memcell`` row per cell).
    ``message_count`` is the count of fresh (newly-arrived, post-filter)
    canonical rows from this call; the response DTO reads it directly.
    """

    # cells 是 boundary 算法切出的记忆单元，是后续 user/agent pipeline 的共同输入。
    cells: list[MemCell]

    # memcell_ids 与 cells 按索引一一对应。
    # 这些 ID 在本阶段生成并持久化，确保下游两个 pipeline 引用的是同一条 memcell 账本记录。
    memcell_ids: list[str]

    # 每个 cell 对应的原始 everos message_id 列表。
    # 由于算法返回的是 MemCell，不直接携带 everos 消息 ID，
    # 本模块需要基于 1:1 顺序关系补上这层映射。
    per_cell_message_ids: list[list[str]]

    # 每个 cell 中出现过的所有 sender_id。
    # 这不是“owner”，而是下游按用户 fan-out 时需要的参与者集合。
    per_cell_all_senders: list[list[str]]

    # 本次边界阶段的处理状态。
    # 调用方可以根据它判断是否继续进入下游 pipeline。
    status: Status

    # fresh 消息数量，已经按 mode 过滤后计算。
    # 这个值服务于响应 DTO 或指标统计，不等同于 merged/buffer 总长度。
    message_count: int


async def prepare_cells(
    ingested: IngestResult,
    *,
    mode: Mode,
    is_final: bool,
    llm_client: LLMClient | None,
    prompt_loader: PromptLoader,
    hard_token_limit: int,
    hard_msg_limit: int,
) -> BoundaryOutcome:
    """Run the boundary stage end-to-end and persist tail back to buffer."""
    # app_id/project_id 是多应用、多项目隔离维度。
    # 后续 buffer、memcell、conversation_status 的读写都必须带上它们，
    # 否则不同项目的同一 session_id 可能互相污染。
    app_id = ingested.app_id
    project_id = ingested.project_id

    # fresh 表示“这次请求新进来的、并且适合当前 mode 的消息”。
    # chat 模式会丢掉工具相关行；agent 模式会保留完整轨迹。
    # 这一步放在读取 buffer 之前，是为了让本次 message_count 能反映真正参与当前模式的新消息。
    fresh = _filter_for_mode(ingested.messages, mode)

    # 如果当前没有任何新消息，并且这不是最终 flush，就没有必要读取 buffer 或调用 LLM。
    # status=skipped 表示本轮没有有效输入推动边界阶段前进。
    if not fresh and not is_final:
        return _empty_outcome(status="skipped", message_count=0)

    # 先读取之前遗留在 unprocessed buffer 中的尾部消息。
    # 这些消息上一次还不足以形成完整 cell，所以要与 fresh 合并后重新尝试边界检测。
    buffer_rows = await unprocessed_buffer_repo.list_for_track(
        ingested.session_id, _TRACK, app_id=app_id, project_id=project_id
    )

    # buffer 表里存的是 sqlite row，需要反序列化回 CanonicalMessage，
    # 这样 buffered 和 fresh 才能进入同一套 merge/detect 流程。
    buffered = [_row_to_canonical(r) for r in buffer_rows]

    # buffered + fresh 可能存在重复消息。
    # 这里先按 message_id 去重，再按时间和 ID 排序，保证后续算法看到的是稳定的时间序列。
    merged = _merge_dedupe_sort(buffered, fresh)

    # 如果合并后仍然没有消息，说明当前调用没有可处理内容。
    # 这里返回 accumulated 而不是 skipped，是因为 final flush 或空 buffer 场景下，
    # 语义上是“没有形成 cell”，而不是“本轮过滤掉了输入”。
    if not merged:
        return _empty_outcome(status="accumulated", message_count=0)

    # Need a role=user anchor for downstream episode extraction; assistant-
    # only / tool-only batches sit in the buffer until a user message lands.
    # 下游 episode extraction 需要用户消息作为锚点来判断“这段记忆和谁有关”。
    # 如果当前不是 final，而且 merged 中没有 user 消息，就先不切 cell。
    # 这样 assistant-only 或 tool-only 的片段不会被过早抽成缺少上下文的记忆。
    if not is_final and not any(m.role == "user" for m in merged):
        # 把本次合并后的内容整体写回 buffer，等待后续 user 消息到达。
        await _replace_buffer(ingested.session_id, merged, app_id, project_id)

        # 虽然没有进入边界检测，但会话已经看到新的消息了，
        # 因此更新 last_message_ts，方便状态追踪和外部观察进度。
        await _touch_last_message_ts(ingested.session_id, merged, app_id, project_id)

        # accumulated 表示消息被保存起来了，只是还没形成可抽取的 cell。
        return _empty_outcome(status="accumulated", message_count=len(fresh))

    # 边界检测依赖 LLM。没有 llm_client 时，不能安全地产生 cell。
    # 这里选择先保存 buffer，再返回 skipped，避免消息丢失。
    if llm_client is None:
        await _replace_buffer(ingested.session_id, merged, app_id, project_id)
        logger.warning(
            "memorize_no_llm_client",
            extra={"session_id": ingested.session_id, "buffered": len(merged)},
        )
        return _empty_outcome(status="skipped", message_count=len(fresh))

    # boundary prompt 由 PromptLoader 管理。
    # 这样检测逻辑和 prompt 内容解耦，方便不同部署或实验切换 prompt。
    boundary_prompt = prompt_loader.load("boundary_detection")

    # 真正执行边界检测。
    # _detect 内部根据 mode 选择 chat 或 agent 路径，并把 CanonicalMessage 映射成 everalgo 类型。
    # 返回值分为 cells 和 tail：cells 是已经闭合的片段，tail 是还需要继续留在 buffer 的结尾片段。
    cells, tail = await _detect(
        merged,
        mode=mode,
        llm_client=llm_client,
        prompt=boundary_prompt,
        is_final=is_final,
        hard_token_limit=hard_token_limit,
        hard_msg_limit=hard_msg_limit,
    )

    # 如果算法没有切出任何 cell，说明整个 merged 仍被认为处于对话中段。
    # 此时不能丢掉它们，也不能进入下游抽取，只能重新写回 buffer。
    if not cells:
        # boundary returned an empty cells set → roll the merged slice
        # back into the buffer (algo says it's still mid-conversation).
        await _replace_buffer(ingested.session_id, merged, app_id, project_id)
        await _touch_last_message_ts(ingested.session_id, merged, app_id, project_id)
        return _empty_outcome(status="accumulated", message_count=len(fresh))

    # 到这里，cells 已经是可以交给下游 pipeline 的闭合单元。
    # 先为每个 cell 生成 memcell_id，保证后续持久化和返回给下游的是同一批 ID。
    memcell_ids = [_mint_memcell_id() for _ in cells]

    # 根据算法返回 cell.items 的长度，把 merged 中的原始消息切回每个 cell 对应的 message_id。
    # 这依赖一个关键约定：输入给算法的 item 与 merged 中的 CanonicalMessage 保持 1:1 顺序。
    per_cell_message_ids = _split_messages_per_cell(merged, cells)

    # 为每个 cell 收集出现过的 sender_id。
    # 该集合后续用于多用户对话中的用户维度 fan-out，而不是用来指定唯一所有者。
    per_cell_all_senders = [_unique_all_senders(c) for c in cells]

    # Write one memcell row per cell (shared across user / agent pipelines).
    # MemCell has no single owner — multi-user dialogue slices stay owner-
    # agnostic. Per-user fan-out (Episode / AtomicFact / Foresight / Profile)
    # happens downstream via ``sender_ids``.
    # raw_type 记录该 cell 是由 chat 还是 agent 语义生成。
    # 同样的 MemCell 账本结构可以同时服务不同下游 pipeline，但 raw_type 让排查和重放更清楚。
    raw_type = _RAW_TYPE_BY_MODE[mode]

    # 把每个 MemCell 转成 sqlite memcell row。
    # zip(..., strict=True) 明确要求 cells 和 memcell_ids 长度完全一致，
    # 如果前面生成逻辑出错，会在这里尽早失败，而不是写入错位账本。
    rows = [
        _build_memcell_row(
            cell=cell,
            memcell_id=memcell_id,
            session_id=ingested.session_id,
            app_id=app_id,
            project_id=project_id,
            raw_type=raw_type,
            message_ids=per_cell_message_ids[i],
            sender_ids=per_cell_all_senders[i],
        )
        for i, (cell, memcell_id) in enumerate(zip(cells, memcell_ids, strict=True))
    ]

    # memcell ledger 在进入下游 user/agent pipeline 之前写入。
    # 这样即使下游某一路失败，仍能通过 memcell row 追踪本次边界阶段已经产出的 canonical cell。
    await memcell_repo.insert_many(rows)

    # 记录本次成功产出的最新 cell timestamp。
    # default=0 只是防御性写法；正常走到这里时 cells 一定非空。
    last_cell_ts = max((cell.timestamp for cell in cells), default=0)

    # 有有效时间戳时，更新 conversation_status 中的 last_memcell_ts。
    # 注意这里写的是“已经形成 memcell 的进度”，和前面的 last_message_ts 含义不同。
    if last_cell_ts:
        await conversation_status_repo.touch_last_memcell_ts(
            ingested.session_id,
            _TRACK,
            from_timestamp(last_cell_ts),
            app_id=app_id,
            project_id=project_id,
        )

    # tail 是算法认为还没闭合的结尾部分。
    # _slice_tail 用 tail 的长度从 merged 尾部切回 CanonicalMessage，
    # 然后写回 unprocessed buffer，供下一次 prepare_cells 继续拼接处理。
    tail_canonical = _slice_tail(merged, tail)
    await _replace_buffer(ingested.session_id, tail_canonical, app_id, project_id)

    # 返回值把下游需要的 cell、共享 ID、消息映射、sender 映射都按相同索引组织好。
    # 下游不需要重新理解 buffer 或边界检测细节，只要按 index 消费即可。
    return BoundaryOutcome(
        cells=cells,
        memcell_ids=memcell_ids,
        per_cell_message_ids=per_cell_message_ids,
        per_cell_all_senders=per_cell_all_senders,
        status="extracted",
        message_count=len(fresh),
    )


# ── Mode-specific filter ──────────────────────────────────────────────────


def _filter_for_mode(
    msgs: list[CanonicalMessage], mode: Mode
) -> list[CanonicalMessage]:
    """Chat mode drops tool rows; agent mode keeps everything."""
    # chat 模式只关心用户/助手自然语言轮次。
    # assistant 如果携带 tool_calls，说明它不是普通回复，而是工具调用请求；
    # tool role 则是工具执行结果。这些都会破坏 chat boundary detector 的输入假设。
    if mode == "chat":
        return [m for m in msgs if m.role in ("user", "assistant") and not m.tool_calls]

    # agent 模式要还原完整 agent trajectory，因此不能丢弃 tool request/result。
    # 返回 list(msgs) 是为了复制一层列表，避免调用方误以为后续会原地修改输入。
    return list(msgs)


# ── Boundary dispatch ─────────────────────────────────────────────────────


# 边界检测依赖 LLM 输出结构化结果，偶发 JSON/格式解析失败是可恢复问题。
# 这里限定最多重试 3 次，避免无限循环，同时给瞬时模型输出问题留出恢复空间。
_BOUNDARY_MAX_ATTEMPTS = 3


async def _detect(
    merged: list[CanonicalMessage],
    *,
    mode: Mode,
    llm_client: LLMClient,
    prompt: str,
    is_final: bool,
    hard_token_limit: int,
    hard_msg_limit: int,
) -> tuple[list[MemCell], list[ConversationItem]]:
    # Retry on ValueError to absorb transient LLM JSON-parse failures from
    # the everalgo boundary detector; non-ValueError errors propagate.
    # last_err 保存最后一次 ValueError。
    # 循环结束后重新抛出它，既保留失败原因，也避免静默吞错。
    last_err: ValueError | None = None

    # 每次 attempt 都重新构造算法输入并调用 LLM。
    # 如果失败原因是模型输出格式临时异常，下一次调用有机会得到合法结果。
    for attempt in range(_BOUNDARY_MAX_ATTEMPTS):
        try:
            # chat 路径：只把 CanonicalMessage 映射成 ChatMessage。
            # 这条路径对应普通对话边界检测，hard_token_limit/hard_msg_limit 直接传给算法。
            if mode == "chat":
                chat_msgs = [_to_chat_message(m) for m in merged]
                result = await detect_boundaries(
                    chat_msgs,
                    llm=llm_client,
                    prompt=prompt,
                    is_final=is_final,
                    hard_token_limit=hard_token_limit,
                    hard_msg_limit=hard_msg_limit,
                )

                # everalgo 返回的 cells/tail 可能不是普通 list。
                # 这里转成 list，让 prepare_cells 后续处理的数据结构更明确。
                return list(result.cells), list(result.tail)

            # Agent mode — facade does filter→detect→remap to preserve tool
            # items. AgentBoundaryDetector intentionally does not expose hard
            # limits; the boundary primitive's defaults apply.
            # agent 路径需要保留 user/assistant/tool 三类轨迹。
            # _to_conversation_item 会根据 role/tool_calls/tool_call_id 映射成不同 ConversationItem 变体。
            items = [_to_conversation_item(m) for m in merged]

            # AgentBoundaryDetector 内部负责适配 agent 轨迹的边界逻辑。
            # 注意这里没有传 hard limit，因为该 detector 当前接口不暴露这些参数。
            detector = AgentBoundaryDetector(llm=llm_client)
            result = await detector.adetect(items, is_final=is_final, prompt=prompt)
            return list(result.cells), list(result.tail)

        # 仅捕获 ValueError。
        # 这通常代表 LLM JSON 解析或算法期望结构不满足；
        # 其他异常可能是 I/O、认证、编程错误，应直接向上冒泡。
        except ValueError as err:
            last_err = err
            logger.warning(
                "boundary_detect_retry",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": _BOUNDARY_MAX_ATTEMPTS,
                    "mode": mode,
                    "error": str(err),
                },
            )

    # 能走到这里说明所有重试都失败。
    # assert 用于告诉类型检查器 last_err 不再是 None，同时保留最后一次真实异常。
    assert last_err is not None
    raise last_err


# ── CanonicalMessage → algo wire types ────────────────────────────────────


def _to_chat_message(m: CanonicalMessage) -> ChatMessage:
    # chat detector 只需要普通聊天消息字段。
    # 这里不处理 tool_calls/tool_call_id，因为 chat 模式的过滤阶段已经把这些消息排除了。
    return ChatMessage(
        id=m.message_id,
        role=m.role,  # type: ignore[arg-type]
        sender_id=m.sender_id,
        sender_name=m.sender_name,
        content=m.text,
        timestamp=to_timestamp_ms(m.timestamp),
    )


def _to_conversation_item(m: CanonicalMessage) -> ConversationItem:
    """Map one canonical row to one ``ConversationItem`` (1:1).

    Dispatch rules — order matters:

    1. ``role="tool"`` (paired with a ``tool_call_id``) → :class:`ToolCallResult`.
    2. ``role="assistant"`` carrying non-empty ``tool_calls`` →
       :class:`ToolCallRequest`; the optional ``content`` text rides along.
    3. ``role`` in {``"user"``, ``"assistant"``} (text-only) →
       :class:`ChatMessage`.

    Caller is expected to provide well-formed inputs (no orphan tool rows,
    no role≠tool with ``tool_call_id``). The fall-through case logs and
    raises so unexpected shapes don't silently corrupt the cell index map.
    """
    # everalgo 的 ConversationItem 使用毫秒时间戳。
    # 先统一转换，避免每个分支重复处理时间字段。
    ts_ms = to_timestamp_ms(m.timestamp)

    # tool role + tool_call_id 表示这是某个工具调用的执行结果。
    # 结果本身没有 sender_id，因为它不是人类或 assistant 的发言，
    # 但必须携带 tool_call_id 才能和前面的 ToolCallRequest 对齐。
    if m.role == "tool" and m.tool_call_id:
        return ToolCallResult(
            tool_call_id=m.tool_call_id,
            content=m.text,
            timestamp=ts_ms,
        )

    # assistant + tool_calls 表示助手正在请求调用工具。
    # 这种消息不能被当成普通 assistant 文本，否则 agent 轨迹会丢失“决策调用工具”的结构。
    if m.role == "assistant" and m.tool_calls:
        return ToolCallRequest(
            tool_calls=[
                AlgoToolCall(
                    id=tc.id,
                    function=ToolCallFunction(
                        # OpenAI tool call function 字段可能缺 name/arguments。
                        # 这里用空字符串兜底，让算法侧始终拿到字符串类型，
                        # 同时不在边界阶段引入额外的 schema 修复逻辑。
                        name=tc.function.get("name", ""),
                        arguments=tc.function.get("arguments", ""),
                    ),
                )
                for tc in m.tool_calls
            ],
            timestamp=ts_ms,
            # assistant 在发起 tool call 时可能同时带有一段解释性文本。
            # 空字符串转为 None，可以更准确表达“没有文本内容”。
            content=m.text or None,
            sender_id=m.sender_id,
            sender_name=m.sender_name,
        )

    # 普通 user/assistant 文本消息映射成 ChatMessage。
    # 这是 agent 轨迹中的自然语言部分，和 tool request/result 一起组成完整 ConversationItem 序列。
    if m.role in ("user", "assistant"):
        return ChatMessage(
            id=m.message_id,
            role=m.role,  # type: ignore[arg-type]
            sender_id=m.sender_id,
            sender_name=m.sender_name,
            content=m.text,
            timestamp=ts_ms,
        )

    # Orphan tool row or unexpected role — break loudly; corrupting the
    # cell→message index map silently is worse than a 5xx.
    # 如果落到这里，说明输入破坏了 agent item 与 canonical row 的映射假设。
    # 直接抛错比继续生成错位 cell 更安全，因为后续会基于顺序切分 message_id。
    raise ValueError(
        f"cannot map canonical row to ConversationItem: role={m.role!r} "
        f"message_id={m.message_id!r} has_tool_call_id={m.tool_call_id is not None}"
    )


# ── Buffer + status helpers ───────────────────────────────────────────────


async def _replace_buffer(
    session_id: str,
    rows: list[CanonicalMessage],
    app_id: str,
    project_id: str,
) -> None:
    # replace 语义是“用 rows 覆盖该 session/track 下现有 buffer”。
    # 因此调用方只要传入当前应保留的 tail/merged 即可，不需要自己先 delete 再 insert。
    await unprocessed_buffer_repo.replace(
        session_id,
        _TRACK,
        # buffer 表保存的是 UnprocessedBuffer row。
        # 写入前要把 CanonicalMessage 序列化成数据库模型，包括 content_items/tool_calls 的 JSON 字段。
        [_canonical_to_row(m, app_id, project_id) for m in rows],
        app_id=app_id,
        project_id=project_id,
    )


async def _touch_last_message_ts(
    session_id: str,
    merged: list[CanonicalMessage],
    app_id: str,
    project_id: str,
) -> None:
    # last_message_ts 表示当前 boundary 阶段已经“看到过”的最新原始消息时间。
    # 即使还没有形成 memcell，只要消息进入 buffer，就应该更新这个状态。
    await conversation_status_repo.touch_last_message_ts(
        session_id,
        _TRACK,
        max(m.timestamp for m in merged),
        app_id=app_id,
        project_id=project_id,
    )


def _canonical_to_row(
    m: CanonicalMessage, app_id: str, project_id: str
) -> UnprocessedBuffer:
    # CanonicalMessage 是内存中的标准对象；UnprocessedBuffer 是 sqlite 持久化对象。
    # 这个函数集中处理二者之间的字段复制和 JSON 序列化，
    # 避免 buffer 写入逻辑散落在 prepare_cells 的多个分支里。
    return UnprocessedBuffer(
        message_id=m.message_id,
        app_id=app_id,
        project_id=project_id,
        session_id=m.session_id,
        track=_TRACK,
        sender_id=m.sender_id,
        sender_name=m.sender_name,
        role=m.role,
        timestamp=m.timestamp,
        # content_items 可能包含多模态解析后的结构化内容。
        # sqlite row 中用 JSON 字符串保存，读取时再恢复为 list/dict。
        content_items_json=json.dumps(m.content_items),
        text=m.text,
        # tool_calls 只有在 agent 轨迹中才重要。
        # 没有 tool_calls 时写 None，而不是 "[]"，可以让读取逻辑更清楚地区分“无字段”和“空列表”。
        tool_calls_json=(
            json.dumps([tc.model_dump() for tc in m.tool_calls])
            if m.tool_calls
            else None
        ),
        tool_call_id=m.tool_call_id,
    )


def _row_to_canonical(r: UnprocessedBuffer) -> CanonicalMessage:
    # 读取 buffer 时，tool_calls_json 需要恢复成内部 ToolCall 模型列表。
    # 默认 None 表示这条消息不携带工具调用请求。
    tool_calls: list[ToolCall] | None = None
    if r.tool_calls_json:
        tool_calls = [ToolCall.model_validate(d) for d in json.loads(r.tool_calls_json)]

    # content_items_json 是写入 buffer 时保存的原始/解析后内容。
    # 如果数据库里为空，则回退成空列表，保证 CanonicalMessage 构造时类型稳定。
    content_items = json.loads(r.content_items_json) if r.content_items_json else []

    # ``r.timestamp`` is UtcDatetime — the BaseTable load-event hook
    # re-attaches ``tzinfo=UTC`` on ORM hydrate, so no defensive coercion
    # is needed here.
    # 这里没有再调用 from_timestamp/to_timestamp_ms，
    # 因为 ORM hydrate 后的 timestamp 已经是内部期望的 datetime 类型。
    return CanonicalMessage(
        message_id=r.message_id,
        session_id=r.session_id,
        sender_id=r.sender_id,
        sender_name=r.sender_name,
        role=r.role,  # type: ignore[arg-type]
        timestamp=r.timestamp,
        content_items=content_items,
        text=r.text,
        tool_calls=tool_calls,
        tool_call_id=r.tool_call_id,
    )


# ── Merge / split / sender helpers ────────────────────────────────────────


def _merge_dedupe_sort(
    buffered: list[CanonicalMessage],
    new: list[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Dedupe by message_id; sort by (timestamp, message_id) ascending."""
    # 先把历史 buffer 放进 dict。
    # 如果新消息里重复出现同一个 message_id，下面的 setdefault 会保留已有版本，
    # 这可以避免重试/重复投递导致同一消息被算法消费两次。
    seen: dict[str, CanonicalMessage] = {m.message_id: m for m in buffered}

    # 新消息只在 message_id 尚不存在时加入。
    # 这里选择“不覆盖”是为了保护已经进入 buffer 的 canonical 结果，
    # 因为它可能包含之前解析过的多模态或规范化字段。
    for m in new:
        seen.setdefault(m.message_id, m)

    # 排序键包含 timestamp 和 message_id。
    # timestamp 保证对话时间顺序；message_id 作为并列时间戳时的稳定 tie-breaker。
    return sorted(seen.values(), key=lambda m: (m.timestamp, m.message_id))


def _slice_tail(
    merged: list[CanonicalMessage],
    tail: list[ConversationItem],
) -> list[CanonicalMessage]:
    """The tail is a trailing slice of ``merged`` (per algo contract)."""
    # everalgo 返回的 tail 是 ConversationItem 列表，不是 CanonicalMessage。
    # 但算法契约保证 tail 对应 merged 的末尾连续切片，所以只需要根据长度反切即可。
    n = len(tail)

    # 没有 tail 表示所有 merged 内容都已经被切进 cells，
    # buffer 可以清空。
    if n == 0:
        return []

    # 从 merged 尾部取 n 条作为下一轮要继续累积的 buffer。
    # 这避免了在 ConversationItem 上反查 message_id 的复杂逻辑。
    return merged[-n:]


def _split_messages_per_cell(
    merged: list[CanonicalMessage],
    cells: list[MemCell],
) -> list[list[str]]:
    """Map each cell index → list of everos message_ids.

    The boundary stage maintains a 1:1 ordering between canonical rows and
    items handed to algo, so we walk ``merged`` left-to-right consuming
    ``len(cell.items)`` rows per cell.
    """
    # result[i] 保存 cells[i] 对应的 message_id 列表。
    result: list[list[str]] = []

    # ptr 指向 merged 中尚未分配给 cell 的第一条消息。
    # 每处理一个 cell，就按 cell.items 的长度向后推进。
    ptr = 0
    for cell in cells:
        n = len(cell.items)

        # cell.items 的长度来自算法输出；
        # merged 的顺序来自算法输入。二者的 1:1 契约让这里可以直接按长度切片。
        result.append([merged[ptr + i].message_id for i in range(n)])
        ptr += n

    return result


def _unique_all_senders(cell: MemCell) -> list[str]:
    """Distinct sender_ids in a cell, preserving first-occurrence order.

    ``ToolCallResult`` does not carry a ``sender_id`` (tool runners are not
    speakers); ``getattr`` keeps the helper agnostic to the item variant.
    """
    # 使用 list 而不是 set，是为了保留 sender 首次出现顺序。
    # 这个顺序可能对调试、可解释性或后续 fan-out 有帮助。
    senders: list[str] = []

    # cell.items 可能混合 ChatMessage、ToolCallRequest、ToolCallResult。
    # 不是每种 item 都有 sender_id，所以不能直接访问属性。
    for item in cell.items:
        sid = getattr(item, "sender_id", None)

        # 只记录非空 sender_id，并去重。
        # ToolCallResult 通常不会进入 senders，因为它没有真正的说话人。
        if sid and sid not in senders:
            senders.append(sid)

    return senders


def _build_memcell_row(
    *,
    cell: MemCell,
    memcell_id: str,
    session_id: str,
    app_id: str,
    project_id: str,
    raw_type: str,
    message_ids: list[str],
    sender_ids: list[str],
) -> Memcell:
    # 把算法产出的 MemCell 包装成 sqlite memcell ledger row。
    # 这个 row 是边界阶段对外的稳定事实：某些 message_ids 组成了某个 memcell_id。
    return Memcell(
        memcell_id=memcell_id,
        app_id=app_id,
        project_id=project_id,
        session_id=session_id,
        track=_TRACK,
        raw_type=raw_type,

        # message_ids/sender_ids 是辅助索引信息。
        # 它们以 JSON 保存，方便一条 row 中保留 cell 与原始消息、参与者之间的完整关系。
        message_ids_json=json.dumps(message_ids),
        sender_ids_json=json.dumps(sender_ids),

        # payload_json 保存完整 MemCell 内容。
        # 使用 model_dump_json 让 Pydantic 模型自己负责内部字段序列化。
        payload_json=cell.model_dump_json(),

        # cell.timestamp 来自算法侧毫秒时间戳，需要转回 everos/sqlite 使用的 datetime。
        timestamp=from_timestamp(cell.timestamp),
    )


def _mint_memcell_id() -> str:
    """Generate an everos-owned memcell identifier."""
    # memcell_id 使用固定前缀加短 uuid。
    # 固定前缀便于日志和数据库中识别对象类型；uuid 片段提供足够低的碰撞概率。
    return f"mc_{uuid.uuid4().hex[:12]}"


def _empty_outcome(*, status: Status, message_count: int) -> BoundaryOutcome:
    # 多个早退分支都会返回“没有 cells”的结果。
    # 集中构造可以保证空结果的各个并行列表始终保持一致，避免遗漏字段。
    return BoundaryOutcome(
        cells=[],
        memcell_ids=[],
        per_cell_message_ids=[],
        per_cell_all_senders=[],
        status=status,
        message_count=message_count,
    )
