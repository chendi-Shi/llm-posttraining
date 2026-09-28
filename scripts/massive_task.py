from __future__ import annotations


TASK_INSTRUCTION = (
    "判断这条中文语音助理请求的意图。只输出一个意图标签，不要输出解释、标点或其他文字。"
)


def user_prompt(utterance: str) -> str:
    return f"{TASK_INSTRUCTION}\n\n请求：{utterance}"


def parse_prediction(raw_text: str, intent_labels: list[str]) -> str | None:
    label = raw_text.strip()
    return label if label in intent_labels else None
