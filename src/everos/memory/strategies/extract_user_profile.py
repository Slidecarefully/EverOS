"""extract_user_profile strategy — synthesise the user's profile from clusters.

Listens to :class:`ProfileClusterUpdated` (fired after
``trigger_profile_clustering`` assigns a memcell to a cluster), pulls
the relevant memcells across all "fresh" clusters, and runs
:class:`everalgo.user_memory.ProfileExtractor` to INIT / UPDATE the
user's profile markdown.

Opensource parity (``mem_memorize.py`` Phase 2):

- **Throttle**: ``total_memcell_count % profile_extraction_interval == 0``;
  default interval = 1 (every memcell triggers a re-extraction).
- **Target clusters**: every cluster whose ``last_ts`` is newer than the
  user's existing profile timestamp, plus the current cluster (so the
  freshly-arrived memcell is always counted even when its cluster's
  ``last_ts`` is older than the profile baseline).
- **Input shape**: raw chat messages — algo's ``_render_conversation``
  unwraps the items list. The sqlite ``memcell.payload_json`` column is
  the long-term archive that lets us replay this beyond
  ``unprocessed_buffer``'s lifetime.

Single-sender assumption today: ``event.owner_id`` is treated as the
profile subject. Multi-user clusters land their additional sender's
profile in a follow-up turn (each cluster gets re-evaluated on every
``ProfileClusterUpdated`` for any participating user).
"""

from __future__ import annotations

# 这个 strategy 是用户画像更新链路的最后一段。
# 前面的 trigger_profile_clustering 会把新的 memcell 放入某个 profile cluster，
# 然后发出 ProfileClusterUpdated；本模块收到事件后，挑选需要重算的 cluster，
# 拉取其中的原始 MemCell，再调用 ProfileExtractor 生成或更新 users/<user_id>/user.md。

# AlgoMemCell 是 everalgo 算法层的 MemCell 类型。
# SQLite memcell.payload_json 保存的是该类型的 JSON，因此读取后需要重新反序列化成它。
from everalgo.types import MemCell as AlgoMemCell

# AlgoProfile 是算法层 profile 模型。
# markdown frontmatter 中的旧 profile 会被转换回 AlgoProfile，作为 UPDATE 的 old_profile 输入。
from everalgo.types import Profile as AlgoProfile

# ProfileExtractor 是用户画像抽取/更新算法。
# 它接收一组按时间排序的 MemCell，以及可选 old_profile，输出新的完整 profile。
from everalgo.user_memory import ProfileExtractor

# get_llm_client 在执行时提供 LLM client。
# profile 抽取需要调用 LLM，对当前候选 memcells 做 INIT 或 UPDATE。
from everos.component.llm import get_llm_client

# logger 记录节流、候选不足、成功提取等 profile 链路状态。
from everos.core.observability.logging import get_logger

# MemoryRoot 用于构造 markdown reader/writer 的默认根目录。
from everos.core.persistence import MemoryRoot

# StrategyContext 是 OME strategy 标准上下文。
# 当前函数不直接使用 ctx.emit，但保留参数以符合框架签名。
from everos.infra.ome.context import StrategyContext

# offline_strategy 将函数注册为离线策略。
# 本策略监听 ProfileClusterUpdated，不主动 emits 其他事件。
from everos.infra.ome.decorator import offline_strategy

# Immediate 表示 cluster 更新事件到达后立即触发 profile 抽取。
from everos.infra.ome.triggers import Immediate

# ProfileReader/Writer 负责读写 users/<user_id>/user.md。
# UserProfileFrontmatter 是 markdown frontmatter 的 schema，用于结构化读写 profile 元数据。
from everos.infra.persistence.markdown import (
    ProfileReader,
    ProfileWriter,
    UserProfileFrontmatter,
)

# cluster_repo 提供用户 cluster 集合读写。
# memcell_repo 让 profile 抽取能从长期 memcell ledger 中回放原始对话切片。
from everos.infra.persistence.sqlite import cluster_repo, memcell_repo

# ProfileClusterUpdated 是本策略的触发事件，携带当前更新的 cluster 和 owner 信息。
from everos.memory.events import ProfileClusterUpdated

# partition lock 用于串行化同一用户 profile 文件的读 → LLM merge → 覆盖写流程。
from everos.memory.strategies._partition_locks import get_partition_lock

# 模块级 logger 便于在日志中定位 extract_user_profile strategy。
logger = get_logger(__name__)

PROFILE_EXTRACTION_INTERVAL = 1
"""Opensource parity: re-extract on every Nth clustered memcell.
``N=1`` matches the opensource default; tune via :class:`Settings` once
the storage budget for profile re-extractions becomes a concern."""
# PROFILE_EXTRACTION_INTERVAL 控制重算频率。
# 当前为 1，表示每来一个 clustered memcell 都会尝试更新 profile；
# 如果以后成本上升，可以调大这个值，让每 N 个 memcell 才触发一次 LLM 重算。

PROFILE_MIN_MEMCELLS = 1
"""Opensource parity: skip when the candidate cluster set holds fewer
than ``N`` memcells across all selected clusters."""
# PROFILE_MIN_MEMCELLS 控制候选 memcell 数量下限。
# 当前为 1，表示只要有一个候选 memcell 就可以抽取；调大后可减少上下文过薄时的 LLM 调用。


# reader/writer 都使用模块级懒加载缓存。
# 这样导入 strategy 模块时不会立刻初始化 markdown 持久化对象，只有真正执行时才创建。
_writer: ProfileWriter | None = None
_reader: ProfileReader | None = None


def _get_writer() -> ProfileWriter:
    # writer 用于覆盖写 users/<owner_id>/user.md。
    # 使用 global 是为了复用模块级缓存实例。
    global _writer

    # 首次写 profile 时才根据默认 MemoryRoot 构造 writer。
    if _writer is None:
        _writer = ProfileWriter(root=MemoryRoot.default())

    return _writer


def _get_reader() -> ProfileReader:
    # reader 用于读取用户现有 profile frontmatter。
    # 它与 writer 分开缓存，因为读写职责不同。
    global _reader

    # 首次读取 profile 时才初始化 reader。
    if _reader is None:
        _reader = ProfileReader(root=MemoryRoot.default())

    return _reader


@offline_strategy(
    name="extract_user_profile",
    trigger=Immediate(on=[ProfileClusterUpdated]),
    emits=[],
    max_retries=2,
)
async def extract_user_profile(
    event: ProfileClusterUpdated, ctx: StrategyContext
) -> None:
    # 本函数处理一次 cluster 更新后，对某个 owner 的 profile 是否需要重算。
    # 它会读取该 owner 的全部 user_memory clusters，再从中选出相对于旧 profile 来说“新鲜”的候选集。

    # Serialise on owner_id: user.md is a single per-user file and the
    # body is a read → LLM merge → overwrite sequence. Different users
    # run fully in parallel.
    # 由于 user.md 是单用户单文件，更新流程是读取旧 profile → 调 LLM 合并 → 覆盖写。
    # 同一个 owner 并发执行会造成读写竞争，所以必须加分区锁。
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("extract_user_profile", partition):
        # 1. Throttle: skip unless the Nth clustered memcell tick lands here.
        # 先读取该 owner 在当前 app/project 下的所有 user_memory clusters。
        # profile 更新不是只看当前 cluster，而是会根据旧 profile 时间戳挑选多个 fresh clusters。
        user_clusters = await cluster_repo.list_for_owner(
            event.owner_id,
            "user_memory",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # total_count 是该 owner 所有 clusters 中 memcell 数量总和。
        # 它用于实现 opensource parity 的节流逻辑。
        total_count = sum(c.count for c in user_clusters)

        # 当 interval > 1 时，只有累计 memcell 数正好落在 N 的倍数上才继续。
        # interval=1 时该条件不会触发，表示每次 cluster 更新都允许 profile 重算。
        if (
            PROFILE_EXTRACTION_INTERVAL > 1
            and total_count % PROFILE_EXTRACTION_INTERVAL != 0
        ):
            logger.info(
                "profile_extraction_throttled",
                owner_id=event.owner_id,
                total_count=total_count,
                interval=PROFILE_EXTRACTION_INTERVAL,
            )
            return

        # 2. Pick clusters fresher than the existing profile (always include
        #    the one we just updated).
        # 读取当前已有 profile。
        # 如果存在旧 profile，它的 profile_timestamp_ms 就是“已吸收信息”的基线。
        existing = await _get_reader().read(
            event.owner_id,
            schema=UserProfileFrontmatter,
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 没有旧 profile 时，基线为 0，相当于所有 cluster 都可能成为候选。
        last_profile_ts = existing[0].profile_timestamp_ms if existing else 0

        # 目标 cluster 有两类：
        # 1. last_ts 晚于旧 profile 的新鲜 cluster；
        # 2. 当前刚更新的 cluster，无论它的 last_ts 是否超过基线都强制包含。
        # 第二条保证本次触发事件对应的 memcell 不会因为时间戳边界问题被漏掉。
        target_clusters = [
            c
            for c in user_clusters
            if c.last_ts > last_profile_ts or c.id == event.cluster_id
        ]

        # 如果没有任何候选 cluster，就没有必要调用 LLM 或写 profile。
        if not target_clusters:
            return

        # 3. Bail if the candidate set is too thin to be worth an LLM call.
        # 把候选 clusters 中的 member memcell_id 拉平成一个列表。
        # cluster member 存的是 memcell_id，因为 profile extractor 需要原始对话切片，而不是 episode summary。
        member_ids = [m for c in target_clusters for m in c.members]

        # 如果候选 memcell 数量低于阈值，就跳过本轮 profile 抽取。
        # 这可以避免用过少上下文反复更新 profile。
        if len(member_ids) < PROFILE_MIN_MEMCELLS:
            logger.info(
                "profile_extraction_below_min_memcells",
                owner_id=event.owner_id,
                memcell_count=len(member_ids),
                threshold=PROFILE_MIN_MEMCELLS,
            )
            return

        # 4. Pull memcell payloads from SQLite, rehydrate to algo types,
        #    time-sort.
        # 从 sqlite memcell ledger 中按 ID 读取原始 payload。
        # unprocessed_buffer 可能早已清空，但 memcell.payload_json 是长期归档，可用于 profile 重放。
        memcell_rows = await memcell_repo.find_by_ids(member_ids)

        # 将 payload_json 还原成算法层 MemCell，并按 timestamp 排序。
        # ProfileExtractor 需要按时间顺序观察用户相关对话，避免乱序影响画像更新。
        algo_memcells = sorted(
            (AlgoMemCell.model_validate_json(r.payload_json) for r in memcell_rows),
            key=lambda mc: mc.timestamp,
        )

        # 如果成员 ID 没有对应到任何 memcell row，则无法构建 profile 输入，直接返回。
        if not algo_memcells:
            return

        # 5. Run the LLM extractor — INIT (no prior) or UPDATE (existing).
        # 如果已有 profile，就转回算法层 AlgoProfile 作为 old_profile，进入 UPDATE 模式；
        # 否则 old_profile=None，进入 INIT 模式。
        old_profile = _to_algo_profile(existing[0]) if existing else None

        # 构造 ProfileExtractor 并调用 LLM。
        # sender_id 使用 event.owner_id，表示这次 profile 的主体用户。
        extractor = ProfileExtractor(llm=get_llm_client())
        new_profile = await extractor.aextract(
            algo_memcells, sender_id=event.owner_id, old_profile=old_profile
        )

        # 6. Write the fresh profile back to users/<user_id>/user.md.
        # 将新的算法层 profile 转成 markdown frontmatter + body，并覆盖写回用户 profile 文件。
        await _persist_profile(
            new_profile,
            owner_id=event.owner_id,
            app_id=event.app_id,
            project_id=event.project_id,
        )

    # 锁释放后记录成功日志。
    # 这里仍可访问 target_clusters/algo_memcells/old_profile，因为它们在锁内完成赋值。
    logger.info(
        "user_profile_extracted",
        owner_id=event.owner_id,
        cluster_count=len(target_clusters),
        memcell_count=len(algo_memcells),
        mode="UPDATE" if old_profile is not None else "INIT",
    )


# ── helpers ──────────────────────────────────────────────────────────────


def _to_algo_profile(fm: UserProfileFrontmatter) -> AlgoProfile:
    """Rehydrate an algo :class:`Profile` from the markdown frontmatter."""
    # markdown reader 读出的是 infra 层 frontmatter schema。
    # ProfileExtractor 需要的是算法层 Profile，因此这里做一次模型转换。
    return AlgoProfile.model_validate(
        {
            "owner_id": fm.user_id,
            "summary": fm.summary,
            "timestamp": fm.profile_timestamp_ms,
            "explicit_info": list(fm.explicit_info),
            "implicit_traits": list(fm.implicit_traits),
        }
    )


async def _persist_profile(
    profile: AlgoProfile, *, owner_id: str, app_id: str, project_id: str
) -> None:
    """Write the freshly extracted profile to ``users/<user_id>/user.md``."""
    # profile 是算法层对象；ProfileWriter 需要 frontmatter schema 和 markdown body。
    # 先把 owner_id/summary/timestamp 之外的扩展字段取出来，用于构造 frontmatter。
    extras = profile.model_dump(exclude={"owner_id", "summary", "timestamp"})

    # explicit_info 和 implicit_traits 是 profile 的核心结构化信息。
    # 用 or [] 兜底，避免 None 写入 frontmatter。
    explicit_info = extras.get("explicit_info") or []
    implicit_traits = extras.get("implicit_traits") or []

    # UserProfileFrontmatter 是用户 profile markdown 的结构化头部。
    # id 使用 profile_<owner_id>，确保同一用户 profile 身份稳定。
    frontmatter = UserProfileFrontmatter(
        id=f"profile_{owner_id}",
        user_id=owner_id,
        summary=profile.summary,
        explicit_info=list(explicit_info),
        implicit_traits=list(implicit_traits),
        profile_timestamp_ms=profile.timestamp,
    )

    # body 当前写入 summary，使 markdown 正文保持可读。
    # 同时 frontmatter 保存结构化字段，供后续 profile update 读取。
    await _get_writer().write(
        owner_id,
        frontmatter=frontmatter,
        body=profile.summary,
        app_id=app_id,
        project_id=project_id,
    )
