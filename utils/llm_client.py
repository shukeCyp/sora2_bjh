#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Client - 通用大模型调用工具
支持设置 api_key, api_proxy, model
"""

import json
import requests
from typing import Optional, List, Dict, Any, Generator


class LLMClient:
    """通用大模型调用客户端"""

    def __init__(
        self,
        api_key: str,
        api_proxy: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        timeout: int = 60
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥
            api_proxy: API 代理地址 (默认 OpenAI 官方地址)
            model: 模型名称 (默认 gpt-3.5-turbo)
            timeout: 请求超时时间(秒)
        """
        self.api_key = api_key
        self.api_proxy = api_proxy.rstrip('/')
        self.model = model
        self.timeout = timeout

    @property
    def chat_endpoint(self) -> str:
        """获取 chat completions 端点"""
        return f"{self.api_proxy}/chat/completions"

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用 chat completions 接口

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称，不指定则使用初始化时的模型
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            stream: 是否使用流式输出
            **kwargs: 其他参数

        Returns:
            API 响应结果
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            self.chat_endpoint,
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout
        )

        response.raise_for_status()
        return response.json()

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式调用 chat completions 接口

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            **kwargs: 其他参数

        Yields:
            生成的文本片段
        """
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            self.chat_endpoint,
            headers=self._get_headers(),
            json=payload,
            stream=True,
            timeout=self.timeout
        )

        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode('utf-8')

            if line_str.startswith('data: '):
                data_str = line_str[6:]

                if data_str == '[DONE]':
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get('choices', [])

                    if choices:
                        delta = choices[0].get('delta', {})
                        content = delta.get('content')

                        if content:
                            yield content

                except json.JSONDecodeError:
                    continue

    def ask(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        简单问答接口

        Args:
            prompt: 用户提问
            system: 系统提示词
            **kwargs: 其他参数

        Returns:
            模型回复的文本
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        response = self.chat(messages, **kwargs)

        return response['choices'][0]['message']['content']

    def ask_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式问答接口

        Args:
            prompt: 用户提问
            system: 系统提示词
            **kwargs: 其他参数

        Yields:
            生成的文本片段
        """
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        yield from self.chat_stream(messages, **kwargs)


# 使用示例
if __name__ == '__main__':
    # 初始化客户端
    client = LLMClient(
        api_key="your-api-key",
        api_proxy="https://api.openai.com/v1",
        model="gpt-3.5-turbo"
    )

    # 简单问答
    # response = client.ask("你好，介绍一下你自己")
    # print(response)

    # 流式问答
    # for chunk in client.ask_stream("写一首关于春天的诗"):
    #     print(chunk, end='', flush=True)
