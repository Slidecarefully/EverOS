"""trigger_profile_clustering strategy — group user memcells by episode topic.

Listens to :class:`EpisodeExtracted` (emitted per-episode after the user
pipeline writes its md), embeds the ``episode_text``, and merges the
resulting size-1 :class:`everalgo.clustering.Cluster` into the user's
existing user-memory cluster set.

Profile-track parity with opensource: uses :func:`cluster_by_geometry`
(rather than the LLM-refined variant) — opensource routes
``has_case=False`` (user-memory) memcells through the embedding-only
path. The members on the merged cluster are ``memcell_id`` rather than
``episode_entry_id`` because the downstream profile-extraction step
needs to feed full memcells (chat messages) back into
:class:`everalgo.user_memory.ProfileExtractor`, not the per-sender
episode summaries.
"""

from __future__ import annotations

# 这个 strategy 连接 Episode 抽取和 Profile 抽取。
# UserMemoryPipeline 写完某个用户的 Episode 后会发 EpisodeExtracted；
# 本模块把 episode_text 向量化并合并进该用户的 user_memory cluster，
# 然后发 ProfileClusterUpdated，推动下游用户画像更新。

# numpy 用于把 embedder 返回的 Python list 转成 float32 ndarray。
# everalgo.clustering 的几何计算依赖向量运算，float32 也更贴近 embedding 存储/计算习惯。
import numpy as np

# AlgoCluster 是算法层 cluster 数据结构。
# 新 episode 会先被包装成一个 size-1 cluster，再与既有 clusters 做几何合并。
from everalgo.clustering import Cluster as AlgoCluster

# cluster_by_geometry 是同步的 embedding-only 合并逻辑。
# 它基于余弦相似度和时间窗口判断，不调用 LLM。
from everalgo.clustering import cluster_by_geometry

# get_embedder 提供 embedding 模型客户端。
# 这里对 episode_text 做 embedding，用于主题聚类。
from everos.component.embedding import get_embedder

# logger 记录 cluster 更新结果，包括是否 merge、cluster_count 等。
from everos.core.observability.logging import get_logger

# StrategyContext 用于发出下游 ProfileClusterUpdated 事件。
from everos.infra.ome.context import StrategyContext

# offline_strategy 将函数注册成 OME 离线策略。
# 本策略监听 EpisodeExtracted，并声明会 emits ProfileClusterUpdated。
from everos.infra.ome.decorator import offline_strategy

# Immediate 表示 EpisodeExtracted 到达后立即触发 clustering。
from everos.infra.ome.triggers import Immediate

# cluster_repo 负责 sqlite 中 cluster 的读取和 upsert。
# mint_cluster_id 在创建新的 size-1 cluster 时生成 cluster id。
from everos.infra.persistence.sqlite import cluster_repo, mint_cluster_id

# EpisodeExtracted 是触发事件，ProfileClusterUpdated 是本策略发出的下游事件。
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated

# 分区锁用于串行化同一 owner 在同一 app/project 下的 cluster 读 → 决策 → 写流程。
from everos.memory.strategies._partition_locks import get_partition_lock

# 模块级 logger 便于在日志中定位 trigger_profile_clustering strategy。
logger = get_logger(__name__)


@offline_strategy(
    name="trigger_profile_clustering",
    trigger=Immediate(on=[EpisodeExtracted]),
    emits=[ProfileClusterUpdated],
    max_retries=2,
)
async def trigger_profile_clustering(
    event: EpisodeExtracted, ctx: StrategyContext
) -> None:
    # 该 strategy 每次处理一个已抽取 Episode。
    # 它不会直接更新 profile，而是先把 episode 对应的源 memcell 合并进用户 cluster，
    # 再通过 ProfileClusterUpdated 事件让 extract_user_profile 选择是否重算 profile。

    # Serialise on owner_id: the strategy reads the user's full cluster
    # set, picks merge target by geometry, then upserts — concurrent runs
    # on the same owner_id would race the read → decide → write cycle.
    # Different users run fully in parallel.
    # Lock per (app, project, owner): clusters are scoped to a space, so a
    # different space's run must not serialise on (or merge into) this one.
    # 同一 owner 的 cluster 集合更新是读全量 → 计算 merge 目标 → upsert 的非原子流程。
    # 因此对 app/project/owner 加锁，避免并发 episode 同时读到旧 cluster 集合后互相覆盖。
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("trigger_profile_clustering", partition):
        # 1. Embed the episode_text into a vector.
        # episode_text 是 UserMemoryPipeline 写出的用户视角 Episode 内容。
        # 聚类用的是 episode summary/narrative 的语义向量，而 cluster member 保存的是源 memcell_id。
        vector_list = await get_embedder().embed(event.episode_text)

        # 将 embedding list 转成 numpy float32 数组。
        # 这一步让后续 cluster_by_geometry 能直接进行向量计算。
        vector = np.asarray(vector_list, dtype=np.float32)

        # 2. Load this user's existing user-memory clusters (scoped to space).
        # 读取当前 owner 在当前 app/project 下所有 user_memory clusters。
        # 这些 existing clusters 是新 size-1 cluster 的候选合并目标。
        existing = await cluster_repo.list_for_owner(
            event.owner_id,
            "user_memory",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 3. Build a size-1 cluster for the fresh memcell (id minted upfront).
        # 新 episode 先被表示成一个只有一个成员的 cluster。
        # preview 保存 episode_text 便于展示或后续调试；members 保存 memcell_id，而不是 episode_entry_id。
        new_cluster = AlgoCluster(
            id=mint_cluster_id(),
            centroid=vector,
            count=1,
            last_ts=event.episode_timestamp_ms,
            preview=[event.episode_text],
            members=[event.memcell_id],
        )

        # 4. Geometry-merge it into an existing cluster (or keep as-is).
        # ``cluster_by_geometry`` is a pure synchronous CPU function (cosine +
        # time-window math, no I/O) returning ``Cluster | None`` directly, so
        # it must not be awaited (``await None`` raises when there is no
        # existing cluster to merge into).
        # cluster_by_geometry 会尝试把 new_cluster 合并到某个 existing cluster。
        # 如果没有合适目标，则返回 None，表示应该保留这个新 cluster。
        merged = cluster_by_geometry(new_cluster, existing)

        # to_save 是最终要持久化的 cluster。
        # 合并成功时保存 merged；否则保存原始 size-1 new_cluster。
        to_save = merged if merged is not None else new_cluster

        # 5. Persist the (possibly-merged) cluster back to SQLite.
        # upsert_with_members 同时保存 cluster 元数据和成员关系。
        # member_type="memcell" 明确说明 members 中的 ID 指向 memcell ledger，而不是 episode entry。
        await cluster_repo.upsert_with_members(
            to_save,
            owner_id=event.owner_id,
            owner_type="user",
            kind="user_memory",
            member_type="memcell",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 6. Emit ProfileClusterUpdated → downstream extract_user_profile.
        # 无论是新建 cluster 还是合并进旧 cluster，下游都需要知道哪个 cluster 被更新。
        assert to_save.id is not None  # both branches above set id
        await ctx.emit(
            ProfileClusterUpdated(
                memcell_id=event.memcell_id,
                cluster_id=to_save.id,
                owner_id=event.owner_id,
                app_id=event.app_id,
                project_id=event.project_id,
            )
        )

    # 锁释放后记录结果日志。
    # merged is not None 表示本次 episode 被并入已有 cluster；否则表示形成了一个新 cluster。
    logger.info(
        "profile_cluster_updated",
        memcell_id=event.memcell_id,
        cluster_id=to_save.id,
        owner_id=event.owner_id,
        merged=merged is not None,
        cluster_count=to_save.count,
    )
