"""extract_atomic_facts strategy — derive AtomicFacts from a fresh MemCell.

One LLM call per memcell, then md-level fan-out to every user sender.
Mirrors :class:`UserMemoryPipeline`'s Episode handling: the algo
prompt is subject-agnostic (``INPUT_TEXT`` + ``TIME`` only, no
``sender_id`` placeholder — see
``everalgo.user_memory.atomic_fact.AtomicFactExtractor.aextract``), so
calling it once per sender would waste LLM tokens and let non-
determinism drift the per-sender md files apart. Instead, run the
extractor once with ``sender_id=None`` (algo's "generic owner"
signal) and rebroadcast the same fact list under each user sender.

Per-owner batching: each sender's full fact list is appended in one
batched ``append_entries`` call rather than ``len(algo_facts)`` single
appends, dropping the per-cell IO complexity from ``O(N²)`` to
``O(N)`` (one read + one write per owner instead of N of each) and
narrowing the per-path lock window from N read-modify-write cycles to
one.

Note ``extract_foresight`` does run per-sender because its prompt
template *does* condition on the target sender; do not collapse that
strategy in the same way without re-checking the prompt.
"""

from __future__ import annotations

# 这个模块是 OME offline strategy，不是主 pipeline 本体。
# 它由 UserPipelineStarted 事件触发：当 user memory pipeline 针对某个 MemCell 启动时，
# 这里并行地从同一个 MemCell 中抽取 AtomicFact，并写入每个用户 owner 的 markdown 文件。

# defaultdict 用于按 owner_id 聚合待写入的 fact entry。
# 这样后面可以对每个 owner 做一次批量 append_entries，而不是每条 fact 单独写文件。
from collections import defaultdict

# Mapping 表示 markdown writer 所需的 entry body 结构只需要映射接口，
# 不要求调用方传入具体 dict 类型；这让函数签名更贴合“只读键值结构”的语义。
from collections.abc import Mapping

# AtomicFactExtractor 是 everalgo 层的算法组件。
# 它接收一个 MemCell，并通过 LLM 抽取事实列表；这里不会按用户重复调用它。
from everalgo.user_memory import AtomicFactExtractor

# get_llm_client 在 strategy 执行时获取默认 LLM client。
# 该 strategy 是事件驱动的异步任务，不像 pipeline 构造函数那样提前注入 extractor。
from everos.component.llm import get_llm_client

# AtomicFact 的时间戳需要写入 markdown inline metadata。
# from_timestamp 把毫秒时间戳转换成时间对象，to_iso_format 再把它转成稳定的 ISO 字符串。
from everos.component.utils.datetime import from_timestamp, to_iso_format

# logger 用于记录本策略的执行结果。
# 特别是没有 user sender 时也会打日志，方便观察某些 memcell 为什么没有产出 facts。
from everos.core.observability.logging import get_logger

# MemoryRoot 提供默认记忆根目录。
# Writer 是懒加载单例时，需要通过它确定 markdown 持久化位置。
from everos.core.persistence import MemoryRoot

# StrategyContext 是 OME strategy 的上下文参数。
# 当前函数签名需要接收它以符合框架约定，尽管本策略暂时没有直接使用 ctx。
from everos.infra.ome.context import StrategyContext

# offline_strategy 装饰器把普通 async 函数注册成 OME 离线策略。
# 注册信息包括策略名、触发事件、是否发出新事件以及重试次数。
from everos.infra.ome.decorator import offline_strategy

# Immediate 表示事件一到就触发该 strategy。
# 这里监听 UserPipelineStarted，因此事实抽取会和 Episode 抽取并行推进。
from everos.infra.ome.triggers import Immediate

# AtomicFactWriter 负责把事实 entry 写入 markdown。
# 该 writer 位于 infra 层，只处理文件写入，不理解 memory 领域模型的转换细节。
from everos.infra.persistence.markdown import AtomicFactWriter

# UserPipelineStarted 是本策略的触发事件。
# 事件中携带 memcell、memcell_id、session/app/project 等足够上下文。
from everos.memory.events import UserPipelineStarted

# AtomicFact 是 everos memory 领域模型。
# 它由算法层 fact 转换而来，并补上 owner_id、session_id、parent_id 等审计字段。
from everos.memory.models import AtomicFact

# 模块级 logger 带上当前模块名，方便在日志中定位到 extract_atomic_facts strategy。
logger = get_logger(__name__)

# writer 使用模块级懒加载缓存。
# 这样每次事件触发时不必重复构造 AtomicFactWriter，也避免在模块 import 时就触碰文件系统。
_writer: AtomicFactWriter | None = None


def _get_writer() -> AtomicFactWriter:
    # writer 是进程内共享对象。
    # 第一次调用时根据默认 MemoryRoot 创建，之后复用同一个实例。
    global _writer

    # 懒加载的好处是：只有真正执行策略并需要写 markdown 时才初始化 writer。
    # 这也让测试或导入模块时更轻量。
    if _writer is None:
        _writer = AtomicFactWriter(root=MemoryRoot.default())

    # 返回已初始化的 writer；调用方无需关心缓存是否已经存在。
    return _writer


@offline_strategy(
    name="extract_atomic_facts",
    trigger=Immediate(on=[UserPipelineStarted]),
    emits=[],
    max_retries=2,
)
async def extract_atomic_facts(
    event: UserPipelineStarted, ctx: StrategyContext
) -> None:
    # 这个函数由 OME 在收到 UserPipelineStarted 后调用。
    # 它不返回 PipelineOutcome，也不发出后续事件；结果主要体现为 markdown 写入和日志记录。

    # 1. List the user senders in this memcell; bail early if there are none.
    # event.memcell 是 boundary stage 切好的同一个 MemCell。
    # 这里先从 cell.items 中找出 role=user 的 sender_id，因为 AtomicFact 最终要按用户 owner 写入。
    memcell = event.memcell

    # 使用 set 去重后 sorted 排序，得到稳定的 owner_id 顺序。
    # 注意这里只有 user role 才会成为 owner；assistant/tool item 不会拥有用户事实文件。
    sender_ids = sorted({m.sender_id for m in memcell.items if m.role == "user"})

    # 如果一个 memcell 中没有用户消息，就没有用户视角可以 fan-out。
    # 这种情况下直接记录 count=0 并返回，避免无意义地调用 LLM。
    if not sender_ids:
        logger.info(
            "atomic_facts_extracted",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
            count=0,
            owner_ids=[],
        )
        return

    # 2. Run the LLM extractor once (algo prompt is subject-agnostic).
    # AtomicFactExtractor 的 prompt 不依赖具体 sender_id，
    # 因此同一个 memcell 只需要调用一次 LLM，得到一份通用 fact list。
    extractor = AtomicFactExtractor(llm=get_llm_client())

    # sender_id=None 是“generic owner”信号。
    # 它让算法按整个 memcell 抽事实，而不是尝试站在某个特定用户视角重写事实。
    algo_facts = await extractor.aextract(memcell, sender_id=None)

    # 3. Fan the fact list out to one domain AtomicFact per (sender, algo_fact).
    # 算法层的 facts 还没有 owner/session/parent 等 everos 领域归属信息。
    # 这里通过双层循环，把同一份 algo_facts 复制到每个用户 owner 名下。
    facts: list[AtomicFact] = [
        AtomicFact.from_algo(
            algo_fact,
            owner_id=sid,
            session_id=event.session_id,
            parent_id=event.memcell_id,
        )
        for sid in sender_ids
        for algo_fact in algo_facts
    ]

    # 4. Group facts by owner so each sender's full list lands in one
    #    batched write.
    # writer.append_entries 期望收到一个 owner 的多个 entry body。
    # 因此这里先按 owner_id 分组，把每个 AtomicFact 转成 markdown entry 所需的 inline/sections。
    by_owner: dict[str, list[tuple[Mapping[str, object], Mapping[str, str]]]] = (
        defaultdict(list)
    )

    # _atomic_fact_to_entry_body 做领域模型到 markdown entry 结构的转换。
    # 分组后，每个 owner 只需要一次批量写入，减少读改写和文件锁次数。
    for fact in facts:
        by_owner[fact.owner_id].append(_atomic_fact_to_entry_body(fact))

    # 5. Write each owner's full list with one batched append_entries.
    # 获取懒加载 writer。
    # 放在 LLM 抽取和分组之后，是因为如果前面没有可写事实，就不需要初始化 writer。
    writer = _get_writer()

    # 对每个 owner 做一次批量 append。
    # app_id/project_id 来自事件，确保不同应用/项目的同一 owner 事实文件互相隔离。
    for owner_id, items in by_owner.items():
        await writer.append_entries(
            owner_id, items, app_id=event.app_id, project_id=event.project_id
        )

    # 最后记录本次抽取结果。
    # count 是 fan-out 后的领域 AtomicFact 数量，不是 algo_facts 的原始数量。
    logger.info(
        "atomic_facts_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=len(facts),
        owner_ids=sender_ids,
    )


def _atomic_fact_to_entry_body(
    fact: AtomicFact,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain AtomicFact into ``(inline, sections)`` for md rendering.

    Mirrors ``_episode_to_entry_body`` in the user_memory pipeline. Lives in
    the memory layer (strategy module) rather than the writer (infra)
    because it depends on :class:`everos.memory.AtomicFact` — infra is
    not allowed to import memory per the layered architecture contract.
    """
    # inline 是 markdown entry 的结构化元数据。
    # 这些字段不是正文内容，而是后续 cascade / indexing / audit 重建事实归属时需要的上下文。
    inline: dict[str, object] = {
        "owner_id": fact.owner_id,
        "session_id": fact.session_id,
        "timestamp": to_iso_format(from_timestamp(fact.timestamp)),
        "parent_type": "memcell",
        "parent_id": fact.parent_id,
    }

    # sections 是 markdown entry 的正文分区。
    # AtomicFact 比 Episode 更简单，核心内容只有一个 Fact 区块。
    sections = {"Fact": fact.fact}

    # 返回 writer.append_entries 需要的二元组：
    # 第一部分用于 inline metadata，第二部分用于 markdown sections。
    return inline, sections
