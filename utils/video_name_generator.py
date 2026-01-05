#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频名称生成器 - 使用 LLM 根据内容生成视频名称
"""

from typing import Optional
from .llm_client import LLMClient


# 预设的生成类型
NAME_STYLE_PROMPTS = {
    "简洁": "生成一个简洁的视频名称，不超过10个字",
    "创意": "生成一个有创意、吸引眼球的视频名称，可以使用比喻或双关",
    "专业": "生成一个专业正式的视频名称，适合商业用途",
    "情感": "生成一个能引起情感共鸣的视频名称，带有感情色彩",
    "悬念": "生成一个带有悬念感的视频名称，让人想点击观看",
}


def generate_video_name(
    content: str,
    style: str = "简洁",
    client: Optional[LLMClient] = None,
    api_key: Optional[str] = None,
    api_proxy: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    根据内容生成视频名称

    Args:
        content: 视频内容描述
        style: 生成类型，可选值: 简洁、创意、专业、情感、悬念
        client: LLMClient 实例，如果不传则根据其他参数创建
        api_key: API 密钥（当 client 为 None 时必填）
        api_proxy: API 代理地址
        model: 模型名称

    Returns:
        生成的视频名称
    """
    # 如果没有传入 client，则创建一个
    if client is None:
        if api_key is None:
            raise ValueError("必须提供 client 或 api_key")

        client = LLMClient(
            api_key=api_key,
            api_proxy=api_proxy or "https://api.openai.com/v1",
            model=model or "gpt-3.5-turbo"
        )

    # 获取风格提示词
    style_prompt = NAME_STYLE_PROMPTS.get(style, NAME_STYLE_PROMPTS["简洁"])

    system_prompt = f"""你是一个视频命名专家。请根据用户提供的视频内容描述，{style_prompt}。

要求：
1. 只输出视频名称，不要有任何其他内容
2. 不要使用引号包裹
3. 名称要与内容相关
4. 不要有标点符号"""

    user_prompt = f"视频内容：{content}"

    name = client.ask(
        prompt=user_prompt,
        system=system_prompt,
        temperature=0.8,
        max_tokens=50
    )

    # 清理结果
    name = name.strip().strip('"\'""''')

    return name


def get_available_styles() -> list:
    """获取所有可用的生成类型"""
    return list(NAME_STYLE_PROMPTS.keys())
