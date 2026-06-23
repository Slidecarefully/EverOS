"""extract_foresight strategy — derive Foresights from a fresh MemCell.

Per-sender extraction (mirrors Episode): a foresight is a forward-looking
statement *about* a specific user, so the algo is invoked once per user
sender and each invocation produces foresights whose ``owner_id`` is
already correct. (AtomicFact, by contrast, uses a subject-agnostic
one-call fan-out.)

Per-owner batching: each sender's full foresight list is appended in
one batched ``append_entries`` call rather than ``N`` single appends,
dropping IO complexity to ``O(N)`` per owner and narrowing the
per-path lock window.
"""

from __future__ import annotations

# 这个模块是一个 OME offline strategy，触发点是 UserPipelineStarted。
# 与 AtomicFact strategy 不同，foresight 是“关于某个具体用户未来可能发生什么”的判断，
# 因此它不能对整个 MemCell 只抽一次再复制给每个用户，而必须按 user sender 分别调用算法。

# defaultdict 用于按 owner_id 聚合待写入的 foresight entry。
# 先分组再批量写入，可以减少 markdown 文件的读改写次数和锁持有次数。
from collections import defaultdict

# Mapping 表达 writer 接收的是只读映射形态的 inline/sections。
# 这里不强迫必须是 dict，虽然当前 helper 实际返回 dict。
from collections.abc import Mapping

# ForesightExtractor 是 everalgo 用户记忆算法层组件。
# 它会读取 MemCell，并结合 sender_id 生成面向该用户的前瞻性记忆。
from everalgo.user_memory import ForesightExtractor

# get_llm_client 在 strategy 执行时提供 LLM client。
# 由于 strategy 是事件驱动函数，不像 pipeline 类那样在构造阶段注入依赖，
# 所以这里在运行时创建 extractor 时获取模型客户端。
from everos.component.llm import get_llm_client

# Foresight timestamp 写入 markdown inline metadata 前需要标准化。
# from_timestamp 把毫秒时间戳转成时间对象，to_iso_format 再输出稳定 ISO 字符串。
from everos.component.utils.datetime import from_timestamp, to_iso_format

# logger 用于记录每个 MemCell 触发后的 foresight 抽取结果。
# 即使没有抽出内容，也能通过日志看到事件被处理过。
from everos.core.observability.logging import get_logger

# MemoryRoot 提供 markdown writer 使用的默认记忆根目录。
# writer 懒加载时需要它来定位 users/<owner_id>/... 文件树。
from everos.core.persistence import MemoryRoot

# StrategyContext 是 OME strategy 的标准上下文参数。
# 当前策略不直接使用 ctx.emit，因为它不声明 emits，但签名仍需要保留。
from everos.infra.ome.context import StrategyContext

# offline_strategy 装饰器把这个 async 函数注册进离线策略系统。
# 其中包含策略名、触发器、产出事件类型和重试策略。
from everos.infra.ome.decorator import offline_strategy

# Immediate 表示事件到达后立即调度执行。
# 这里监听 UserPipelineStarted，使 foresight 抽取与 Episode pipeline 内部工作并行发生。
from everos.infra.ome.triggers import Immediate

# ForesightWriter 只负责 markdown 持久化。
# 它位于 infra 层，不应该知道 Foresight 领域模型如何拆成 inline/sections。
from everos.infra.persistence.markdown import ForesightWriter

# UserPipelineStarted 是本策略的触发事件。
# 事件中携带 MemCell、memcell_id、session/app/project 等完整上下文。
from everos.memory.events import UserPipelineStarted

# Foresight 是 everos memory 领域模型。
# 算法层 foresight 会通过 Foresight.from_algo 转换成这个模型，再写入 markdown。
from everos.memory.models import Foresight

# 模块级 logger 便于在日志中定位到 extract_foresight strategy。
logger = get_logger(__name__)

# writer 使用模块级懒加载缓存。
# 这样模块 import 时不会立即初始化文件系统 writer，只有真正有事件需要写入时才创建。
_writer: ForesightWriter | None = None


def _get_writer() -> ForesightWriter:
    # 使用 global 是因为 _writer 是模块级缓存变量。
    # 该函数把“是否已经初始化 writer”的判断集中起来，避免调用方重复处理。
    global _writer

    # 第一次调用时使用默认 MemoryRoot 构造 ForesightWriter。
    # 后续事件复用同一个 writer 实例，减少重复初始化成本。
    if _writer is None:
        _writer = ForesightWriter(root=MemoryRoot.default())

    # 返回可用 writer；调用方只关心写入，不关心懒加载细节。
    return _writer


@offline_strategy(
    name="extract_foresight",
    trigger=Immediate(on=[UserPipelineStarted]),
    emits=[],
    max_retries=2,
)
async def extract_foresight(event: UserPipelineStarted, ctx: StrategyContext) -> None:
    # 这个 strategy 由 UserPipelineStarted 触发。
    # 它的输入粒度是一个已经由 boundary stage 切好的 MemCell，输出是每个用户 owner 的 foresight markdown entry。

    # 1. List the user senders in this memcell.
    # 先拿到事件携带的 MemCell。
    # 后续所有抽取都围绕这个 cell 展开，不再重新查数据库或重新切边界。
    memcell = event.memcell

    # Foresight 是用户视角的未来判断，因此只对 role=user 的 sender 运行。
    # set 用于去重，sorted 用于让同一 cell 的执行顺序稳定，便于日志和测试复现。
    sender_ids = sorted({m.sender_id for m in memcell.items if m.role == "user"})

    # 如果没有用户 sender，就不需要创建 extractor，也不需要调用 LLM。
    # 这里用条件表达式让后面的循环自然跳过；sender_ids 为空时 foresights 最终也为空。
    extractor = ForesightExtractor(llm=get_llm_client()) if sender_ids else None

    # 2. Run the LLM extractor once per sender (prompt is per-sender).
    # foresights 收集所有 owner 的领域模型结果。
    # 注意这里的数量可能是 sender 数 × 每个 sender 抽出的 foresight 数。
    foresights: list[Foresight] = []

    # 每个用户 sender 单独调用一次算法。
    # 这是因为 ForesightExtractor 的 prompt 会以 sender_id 为条件，产出已经绑定 owner_id 的结果。
    for sid in sender_ids:
        # extractor 在 sender_ids 非空时一定已初始化。
        # 这里没有再做 None 检查，是因为循环只有在 sender_ids 非空时才会进入。
        algo_foresights = await extractor.aextract(memcell, sender_id=sid)

        # 将算法层结果转成 everos 领域模型。
        # from_algo 会保留算法产出的 owner_id；session_id 和 parent_id 则由事件上下文补齐。
        # parent_id 指向源 memcell_id，方便后续从 foresight 追溯到原始对话切片。
        foresights.extend(
            Foresight.from_algo(
                algo_fs,
                session_id=event.session_id,
                parent_id=event.memcell_id,
            )
            for algo_fs in algo_foresights
        )

    # 3. Group foresights by owner so each sender's full list lands in one
    #    batched write.
    # markdown writer 的批量写入是按 owner 文件进行的。
    # 所以先把所有 foresight 按 owner_id 分桶，再对每个 owner 写一次。
    by_owner: dict[str, list[tuple[Mapping[str, object], Mapping[str, str]]]] = (
        defaultdict(list)
    )

    # 每个 Foresight 在入桶前先转换成 markdown writer 所需的 inline/sections 二元组。
    # 这一步仍放在 memory strategy 层，因为它依赖 memory 领域模型。
    for fs in foresights:
        by_owner[fs.owner_id].append(_foresight_to_entry_body(fs))

    # 4. Write each owner's full list with one batched append_entries.
    # 真正需要写文件时才获取 writer。
    # 即使 foresights 为空，获取 writer 也不会改变语义，只是后续循环不会写入。
    writer = _get_writer()

    # 每个 owner_id 只执行一次 append_entries。
    # app_id/project_id 来自事件，确保不同 app/project 的用户记忆文件互相隔离。
    for owner_id, items in by_owner.items():
        await writer.append_entries(
            owner_id, items, app_id=event.app_id, project_id=event.project_id
        )

    # 记录本次 strategy 的产出规模。
    # owner_ids 从实际 foresights 中回收，而不是直接用 sender_ids，
    # 这样日志能反映“最终确实产生了 foresight 的 owner”。
    logger.info(
        "foresights_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=len(foresights),
        owner_ids=sorted({f.owner_id for f in foresights}),
    )


def _foresight_to_entry_body(
    fs: Foresight,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain Foresight into ``(inline, sections)`` for md rendering.

    Mirrors ``_episode_to_entry_body`` / ``_atomic_fact_to_entry_body``.
    Optional time-window fields (``start_time`` / ``end_time`` /
    ``duration_days``) are emitted only when set so md stays compact.
    """
    # inline 是 markdown entry 的结构化元数据。
    # 它保存 owner/session/time/source 等审计字段，供后续 cascade、索引或调试使用。
    inline: dict[str, object] = {
        "owner_id": fs.owner_id,
        "session_id": fs.session_id,
        "timestamp": to_iso_format(from_timestamp(fs.timestamp)),
        "parent_type": "memcell",
        "parent_id": fs.parent_id,
    }

    # start_time/end_time/duration_days 描述 foresight 的时间窗口。
    # 这些字段不是所有 foresight 都有，所以只在存在时写入，避免 markdown metadata 变得臃肿。
    if fs.start_time:
        inline["start_time"] = fs.start_time
    if fs.end_time:
        inline["end_time"] = fs.end_time
    if fs.duration_days is not None:
        inline["duration_days"] = fs.duration_days

    # sections 是面向人类阅读的正文部分。
    # Foresight 保存预测内容，Evidence 保存模型依据，二者分开便于后续展示和审查。
    sections = {"Foresight": fs.foresight, "Evidence": fs.evidence}

    # 返回 writer.append_entries 需要的 entry body。
    return inline, sections
