"""Ingest pipeline entry — normalise external input into canonical form.

Input shape (received from the service layer, decoupled from any
specific DTO module):

    {
      "session_id": "...",
      "messages": [
        {
          "sender_id": "...",
          "sender_name": "...",        # optional
          "role": "user" | "assistant" | "tool",
          "timestamp": 1740564000000,  # unix ms
          "content": "..." | [ContentItem dicts],
          "tool_calls": [...] | None,  # OpenAI shape
          "tool_call_id": "..." | None,
        },
        ...
      ]
    }

Output: :class:`everos.memory.IngestResult`.
"""

from __future__ import annotations

from typing import Any

# 这里引入多模态 LLM 客户端，并不是所有消息都会用到它。
# 只有当输入内容里存在尚未解析的图片、音频、文件等非纯文本内容时，
# 后面的解析流程才会懒加载并调用这个客户端，避免纯文本场景产生额外成本。
from everos.component.llm import get_multimodal_llm_client

# 服务层传入的是 unix 毫秒时间戳；内存层需要统一的 datetime 表示。
# from_timestamp 承担了时间格式归一化的职责，避免业务代码散落时间转换细节。
from everos.component.utils.datetime import from_timestamp

# 配置用于控制多模态解析等运行时参数，例如并发上限。
# 这里没有在模块加载时提前读取配置，而是在真正需要多模态解析时读取，
# 可以减少不必要的初始化，并保持运行时配置的可控性。
from everos.config import load_settings

# CanonicalMessage 是进入 memory 模块后的标准消息结构。
# IngestResult 是整个 ingest 阶段的输出边界。
# ToolCall 则用于把外部 OpenAI 兼容的 tool_calls 结构统一成内部模型。
from everos.memory import CanonicalMessage, IngestResult, ToolCall

# parser 里的工具函数负责判断和补全多模态内容：
# - enrich_content_items：真正填充非文本内容的可读描述或结构化信息
# - has_unparsed_multimodal：判断是否还有未解析的多模态项
# - require_multimodal：在需要多模态能力时做依赖/配置检查
from everos.memory.extract.parser import (
    enrich_content_items,
    has_unparsed_multimodal,
    require_multimodal,
)

# gen_message_id 根据 session、时间戳和消息顺序生成稳定的消息 ID。
# 这样同一批输入在相同顺序下可以得到一致 ID，方便后续去重、索引或追踪。
from .id_gen import gen_message_id

# multimodal 模块把外部传入的 content 统一转换为内部 ContentItem 列表，
# 并在解析完成后进一步推导出可用于检索、摘要或存储的文本表示。
from .multimodal import coerce_items, derive_text


async def process(payload: dict[str, Any]) -> IngestResult:
    """Normalise the raw add payload into an :class:`IngestResult`.

    The function is ``async`` for symmetry with the rest of the pipeline,
    even though current logic is pure CPU.
    """
    # payload 是服务层传入的原始 add 请求。
    # session_id 是强制字段，因为每条消息都必须归属到某个会话，
    # 后续生成 message_id、构造 CanonicalMessage、返回 IngestResult 都会依赖它。
    session_id: str = payload["session_id"]

    # app_id 和 project_id 是可选的上层隔离维度。
    # 如果调用方没有传入，就回退到 "default"，保证内存层始终能拿到明确的归属信息，
    # 避免后续存储或查询时出现 None 分支。
    app_id: str = payload.get("app_id") or "default"
    project_id: str = payload.get("project_id") or "default"

    # messages 是本次 ingest 的核心数据。
    # 这里保留原始 dict 列表，后面会逐条转换成 CanonicalMessage。
    raw_messages: list[dict[str, Any]] = payload["messages"]

    # canonical 用来收集已经归一化后的消息。
    # non_text_total 记录本次处理过程中仍然存在或被统计到的非文本内容数量，
    # 最终会写入 IngestResult，供调用方了解这批数据里有多少非文本输入参与了处理。
    canonical: list[CanonicalMessage] = []
    non_text_total = 0

    # 按原始顺序处理消息非常重要。
    # idx 不只是循环下标，还参与 message_id 的生成；同一时间戳下的多条消息
    # 也可以通过 idx 保持稳定且可区分的身份。
    for idx, m in enumerate(raw_messages):
        # 外部 content 可能是纯字符串，也可能已经是 ContentItem 字典列表。
        # coerce_items 把这些输入形态统一为内部 content_items，
        # 后面的多模态检测、文本推导就不必关心原始 DTO 的差异。
        content_items = coerce_items(m["content"])

        # 如果 content_items 中存在尚未解析的多模态内容，
        # 当前 ingest 阶段会在写入 CanonicalMessage 前先尝试补全它们。
        # 这一步保证后续 memory/extract/search 逻辑拿到的是尽可能完整、可读的内容。
        if has_unparsed_multimodal(content_items):
            # 只有真的遇到未解析多模态内容时才要求多模态能力。
            # 如果环境没有正确配置相关能力，这里会尽早失败，
            # 而不是等到后续存储或检索阶段才出现更难定位的问题。
            require_multimodal()

            # enrich_content_items 会原地丰富 content_items。
            # 它使用多模态 LLM 客户端为图片等非文本项生成可用的文本/结构化描述；
            # max_concurrency 来自配置，用来限制并发，避免同时解析大量媒体时压垮模型服务。
            await enrich_content_items(
                content_items,
                llm=get_multimodal_llm_client(),
                max_concurrency=load_settings().multimodal.max_concurrency,
            )

        # derive_text 是内容归一化后的第二层抽取：
        # 它从 content_items 中得到一段代表整条消息的 text，
        # 同时返回非文本项数量 non_text，供整个 ingest 结果汇总统计。
        text, non_text = derive_text(content_items)
        non_text_total += non_text

        # 输入约定 timestamp 是 unix 毫秒。
        # 这里显式转成 int，既能兼容上游传来的数字字符串，
        # 也能在非法值出现时尽早抛出清晰的转换错误。
        ts_ms: int = int(m["timestamp"])

        # message_id 的生成依赖 session_id、时间戳和消息顺序。
        # 这三个信息组合起来既体现消息来源，也保留同一批消息内的顺序稳定性。
        message_id = gen_message_id(session_id, ts_ms, idx)

        # 内部 CanonicalMessage 不直接保存裸毫秒时间戳，
        # 而是统一保存经过 from_timestamp 转换后的时间对象。
        ts = from_timestamp(ts_ms)

        # 到这里，一条原始消息已经完成了核心归一化：
        # 内容被统一成 content_items，文本被推导成 text，时间和 ID 已标准化，
        # tool_calls 也会在 _coerce_tool_calls 中转成内部 ToolCall 模型。
        canonical.append(
            CanonicalMessage(
                message_id=message_id,
                session_id=session_id,
                sender_id=m["sender_id"],
                sender_name=m.get("sender_name"),
                role=m["role"],
                timestamp=ts,
                content_items=content_items,
                text=text,
                tool_calls=_coerce_tool_calls(m.get("tool_calls")),
                tool_call_id=m.get("tool_call_id"),
            )
        )

    # 整个 process 的输出不是简单的消息列表，而是 IngestResult。
    # 这样调用方可以同时拿到会话归属、应用/项目归属、标准消息集合，
    # 以及本次处理涉及的非文本内容统计。
    return IngestResult(
        session_id=session_id,
        app_id=app_id,
        project_id=project_id,
        messages=canonical,
        unparsed_non_text_count=non_text_total,
    )


def _coerce_tool_calls(
    raw: list[dict[str, Any]] | list[Any] | None,
) -> list[ToolCall] | None:
    # tool_calls 是可选字段。
    # None 或空列表都表示这条消息没有工具调用信息，
    # 因此直接返回 None，保持 CanonicalMessage 里的语义清晰。
    if not raw:
        return None

    # out 收集统一后的 ToolCall 实例。
    # 这个函数要兼容多种上游形态：已经是 ToolCall、Pydantic 模型、
    # 或普通 dict，因此不能简单地假设 raw 的元素一定是字典。
    out: list[ToolCall] = []

    # 逐个转换 tool call，避免某个上游 DTO 类型直接泄漏进 memory 层。
    # 经过这里之后，CanonicalMessage 中保存的 tool_calls 都是内部 ToolCall 模型。
    for tc in raw:
        # 如果调用方已经传入内部 ToolCall，说明它已经处于目标结构，
        # 可以直接复用，避免重复校验或二次转换。
        if isinstance(tc, ToolCall):
            out.append(tc)

        # 一些上游对象可能是 Pydantic 模型，而不是普通 dict。
        # model_dump 可以把它们转成可验证的字典结构，
        # 再交给 ToolCall.model_validate 生成内部模型。
        elif hasattr(tc, "model_dump"):
            out.append(ToolCall.model_validate(tc.model_dump()))

        # 最后一类情况是普通 dict 或其他可被 Pydantic 校验的结构。
        # model_validate 在这里既做转换，也承担字段合法性校验。
        else:
            out.append(ToolCall.model_validate(tc))

    # 返回统一后的 ToolCall 列表。
    # 调用方无需再关心 tool_calls 原本来自 OpenAI shape、Pydantic DTO，
    # 还是已经构造好的内部模型。
    return out
