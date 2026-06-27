"""Method → hybrid pipeline selector.

Translates the public 4-method enum into everos's internal pipeline routing signal.
``AGENTIC`` is intercepted by the manager before this function is called.
Passing ``AGENTIC`` here is a caller contract violation and raises
``ValueError`` as a defensive guard.

* ``KEYWORD`` / ``VECTOR`` → ``None`` → manager skips ``everalgo.rank``.
* ``HYBRID``  → ``"hierarchy"`` (episode / atomic_fact) — four-layer pipeline
  (RRF → MaxSim → RRF merge → single-pass eviction)
  or ``"vector_anchored"`` (agent_case) — everalgo vector-anchored fusion (alpha=0.7)
  or ``"skill_hybrid"`` (agent_skill) — custom rrf → cross-encoder rerank → optional
  verify.
"""

from __future__ import annotations

from typing import Literal

from .dto import SearchMethod

# 这个模块的职责刻意保持很窄：它不做召回、不做排序，也不接触存储；
# 只把外部 API 暴露的 SearchMethod，结合当前要检索的 memory kind，
# 翻译成 SearchManager 后续可以理解的“路线信号”。
#
# 换句话说，这里是 public enum 和内部 pipeline 名称之间的路由表。
# 这样 SearchManager 不需要把所有 kind 的分支规则散落在各个检索函数里，
# 只要问 resolve_pipeline：“这个 method + kind 应该走哪条路？”
# 再根据返回值决定是直接单路召回，还是进入某个融合 / 重排管线。
KindName = Literal["episode", "atomic_fact", "agent_case", "agent_skill"]


def resolve_pipeline(
    method: SearchMethod,
    kind: KindName,
) -> tuple[str | None, None]:
    """Return ``(pipeline_signal, None)`` for a ``(method, kind)`` pair.

    ``pipeline_signal`` of ``None`` means "do not call ``everalgo.rank.arank``;
    the manager runs single-route recall and returns directly".
    ``"hierarchy"`` routes to the four-layer episode pipeline in
    ``memory.search.hierarchy`` (RRF → MaxSim → RRF merge → eviction).
    ``"vector_anchored"`` routes to ``everalgo.rank.arank`` with vector-anchored
    fusion (alpha=0.7, saturation_k=5.0) — matches the opensource case retrieval.
    ``"skill_hybrid"`` routes to the custom skill hybrid orchestrator in
    ``memory.search.skill_hybrid`` (rrf → cross-encoder rerank → optional verify).
    """

    # 第一层先处理“单一路径召回”的方法。
    #
    # KEYWORD 和 VECTOR 本身已经明确了召回路线：
    # - KEYWORD 只需要 sparse/BM25 召回；
    # - VECTOR 只需要 dense/ANN 召回。
    #
    # 它们不需要 sparse+dense 融合，也不需要 everalgo.rank.arank 参与。
    # 因此这里返回 None 作为 pipeline_signal，SearchManager 收到 None 后，
    # 会在自己的单路召回分支里直接调用 sparse_recall 或 dense_recall，
    # 然后 shape 成 DTO 返回。
    if method in (SearchMethod.KEYWORD, SearchMethod.VECTOR):
        return None, None

    # 第二层处理 HYBRID。
    #
    # HYBRID 的含义不是“所有 kind 都走同一套融合算法”，而是：
    # 当前 kind 同时需要利用 keyword 和 vector 两路信号，但每类数据的
    # 最优融合方式不同，所以这里继续按 kind 分派到不同内部管线。
    if method == SearchMethod.HYBRID:
        # episode / atomic_fact 走 hierarchy。
        #
        # 对用户记忆 episode 来说，单纯把 episode 级 sparse 和 dense 做融合还不够：
        # 长 episode 的整体向量可能会稀释某个细粒度事实的语义匹配。
        # 所以 HYBRID episode 会交给 hierarchy 管线：
        # 先做 episode 级 RRF，再通过 atomic_fact 做 MaxSim 父级回捞，
        # 再把两路 episode 结果合并，最后允许高分 fact 替换父 episode 进入结果。
        #
        # atomic_fact 也归到这个信号，是为了保持“事实驱动的层级召回”在
        # episode 相关路径里统一表达；真正如何使用这个信号由上层 manager 决定。
        if kind in ("episode", "atomic_fact"):
            return "hierarchy", None

        # agent_case 走 vector_anchored。
        #
        # case 通常是较完整的经验 / 案例文本，dense 语义相似度往往是主信号；
        # sparse 命中仍然有价值，但更多是补充精确词、实体、术语。
        # 因此这里返回 vector_anchored，让后续 everalgo.rank.arank 使用
        # 以向量分数为锚的融合策略，而不是 episode 那套 fact 层级管线。
        if kind == "agent_case":
            return "vector_anchored", None

        # agent_skill: custom hybrid orchestrator (rrf → cross-encoder → optional
        # verify)
        #
        # 走到这里时，kind 只能是 agent_skill。
        #
        # skill 的检索目标不是“找一个相似案例”，而是“找一个可复用能力”；
        # 它需要先合并 sparse / dense 候选，再用 cross-encoder 判断技能文本
        # 是否真的和 query 的任务意图相关，必要时还可能做额外 verify。
        # 所以它不直接交给通用 arank，而是返回 skill_hybrid，
        # 让 SearchManager 转到 memory.search.skill_hybrid 的专门编排。
        return "skill_hybrid", None

    # AGENTIC 不应该走到这里。
    #
    # 在正常调用链里，SearchManager 会先拦截 SearchMethod.AGENTIC，
    # 直接进入 agentic / cluster / multi-query 等专门流程；
    # resolve_pipeline 只负责 KEYWORD、VECTOR、HYBRID 的静态路线解析。
    #
    # 因此如果 method 既不是单路方法，也不是 HYBRID，说明调用方违反了
    # 这个模块的契约。这里显式抛 ValueError，能让问题暴露在路由入口，
    # 而不是让后续检索逻辑收到一个无法解释的 pipeline_signal。
    raise ValueError(f"unsupported method: {method!r}")
