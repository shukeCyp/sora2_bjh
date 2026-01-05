#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sora Video Generator - 批量视频生成工具
使用 PyQt5 界面，支持 CSV 导入，50线程并发调用 API
"""

import sys
import os
import csv
import json
import requests
import threading
import codecs
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QFileDialog,
    QProgressBar, QLabel, QHeaderView, QMessageBox, QLineEdit,
    QSpinBox, QGroupBox, QStatusBar, QDialog, QComboBox,
    QFormLayout, QTextEdit, QDialogButtonBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, pyqtSlot, QMetaObject, Q_ARG
from PyQt5.QtGui import QColor

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

# 默认配置
DEFAULT_CONFIG = {
    "llm_api_key": "",
    "llm_api_proxy": "https://api.openai.com/v1",
    "llm_model": "gpt-3.5-turbo",
    "title_style": "简洁",
    "title_prompt": "根据视频内容生成一个吸引人的标题",
    "download_threads": 20,
    "auto_download": False,
    "auto_retry": False,
    "max_retries": 3
}


def detect_encoding(file_path: str) -> str:
    """检测文件编码"""
    # 常见编码列表，按优先级排序
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'big5', 'latin-1', 'cp1252']

    # 先尝试检测 BOM
    with open(file_path, 'rb') as f:
        raw = f.read(4)
        if raw.startswith(codecs.BOM_UTF8):
            return 'utf-8-sig'
        elif raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
            return 'utf-16'
        elif raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
            return 'utf-32'

    # 读取文件内容用于编码检测
    with open(file_path, 'rb') as f:
        raw_content = f.read()

    # 尝试各种编码
    for encoding in encodings:
        try:
            raw_content.decode(encoding)
            print(f"[编码检测] 检测到编码: {encoding}")
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue

    # 如果都失败，返回 utf-8 并使用 errors='replace'
    print("[编码检测] 无法确定编码，使用 utf-8 with errors='replace'")
    return 'utf-8'


def load_config() -> dict:
    """加载配置文件"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 合并默认配置，确保所有字段存在
                return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"[配置] 加载配置文件失败: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """保存配置文件"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[配置] 配置已保存到: {CONFIG_FILE}")
    except Exception as e:
        print(f"[配置] 保存配置文件失败: {e}")


class TaskStatus(Enum):
    PENDING = "待处理"
    PROCESSING = "处理中"
    SUCCESS = "成功"
    FAILED = "失败"


# AI标题生成风格选项
TITLE_STYLES = ["简洁", "创意", "专业", "情感", "悬念"]


class SettingsDialog(QDialog):
    """设置对话框"""

    def __init__(self, parent=None, config: dict = None):
        super().__init__(parent)
        self.config = config or load_config()
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("设置")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)

        # === 大模型配置 ===
        llm_group = QGroupBox("大模型配置 (用于AI生成标题)")
        llm_layout = QFormLayout(llm_group)

        self.api_key_input = QLineEdit(self.config.get("llm_api_key", ""))
        self.api_key_input.setPlaceholderText("请输入 API Key")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        llm_layout.addRow("API Key:", self.api_key_input)

        self.api_proxy_input = QLineEdit(self.config.get("llm_api_proxy", ""))
        self.api_proxy_input.setPlaceholderText("https://api.openai.com/v1")
        llm_layout.addRow("API Proxy:", self.api_proxy_input)

        self.model_input = QLineEdit(self.config.get("llm_model", ""))
        self.model_input.setPlaceholderText("gpt-3.5-turbo")
        llm_layout.addRow("Model:", self.model_input)

        layout.addWidget(llm_group)

        # === AI标题设置 ===
        title_group = QGroupBox("AI标题生成设置")
        title_layout = QFormLayout(title_group)

        self.style_combo = QComboBox()
        self.style_combo.addItems(TITLE_STYLES)
        current_style = self.config.get("title_style", "简洁")
        if current_style in TITLE_STYLES:
            self.style_combo.setCurrentText(current_style)
        title_layout.addRow("生成风格:", self.style_combo)

        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("输入自定义提示词，用于指导AI生成标题...")
        self.prompt_input.setText(self.config.get("title_prompt", ""))
        self.prompt_input.setMaximumHeight(100)
        title_layout.addRow("自定义提示:", self.prompt_input)

        layout.addWidget(title_group)

        # === 下载设置 ===
        download_group = QGroupBox("下载设置")
        download_layout = QFormLayout(download_group)

        self.download_threads_spin = QSpinBox()
        self.download_threads_spin.setRange(1, 50)
        self.download_threads_spin.setValue(self.config.get("download_threads", 20))
        download_layout.addRow("下载线程数:", self.download_threads_spin)

        self.auto_download_checkbox = QCheckBox("生成成功后自动下载视频")
        self.auto_download_checkbox.setChecked(self.config.get("auto_download", False))
        download_layout.addRow("自动下载:", self.auto_download_checkbox)

        layout.addWidget(download_group)

        # === 生成设置 ===
        gen_group = QGroupBox("生成设置")
        gen_layout = QFormLayout(gen_group)

        self.auto_retry_checkbox = QCheckBox("生成失败后自动重试")
        self.auto_retry_checkbox.setChecked(self.config.get("auto_retry", False))
        gen_layout.addRow("自动重试:", self.auto_retry_checkbox)

        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setRange(1, 10)
        self.max_retries_spin.setValue(self.config.get("max_retries", 3))
        gen_layout.addRow("最大重试次数:", self.max_retries_spin)

        layout.addWidget(gen_group)

        # === 按钮 ===
        button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def save_settings(self):
        """保存设置"""
        self.config["llm_api_key"] = self.api_key_input.text().strip()
        self.config["llm_api_proxy"] = self.api_proxy_input.text().strip() or DEFAULT_CONFIG["llm_api_proxy"]
        self.config["llm_model"] = self.model_input.text().strip() or DEFAULT_CONFIG["llm_model"]
        self.config["title_style"] = self.style_combo.currentText()
        self.config["title_prompt"] = self.prompt_input.toPlainText().strip()
        self.config["download_threads"] = self.download_threads_spin.value()
        self.config["auto_download"] = self.auto_download_checkbox.isChecked()
        self.config["auto_retry"] = self.auto_retry_checkbox.isChecked()
        self.config["max_retries"] = self.max_retries_spin.value()

        save_config(self.config)
        self.accept()

    def get_config(self) -> dict:
        """获取配置"""
        return self.config


@dataclass
class VideoTask:
    """视频生成任务"""
    row_index: int
    image_path: str
    prompt: str
    resolution: str
    duration: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    video_url: Optional[str] = None
    error_msg: Optional[str] = None
    download_path: Optional[str] = None
    retry_count: int = 0  # 当前重试次数


class APIWorker(QThread):
    """API 调用工作线程"""
    # 信号定义
    task_started = pyqtSignal(int)  # row_index
    task_progress = pyqtSignal(int, int, str)  # row_index, progress, message
    task_completed = pyqtSignal(int, bool, str)  # row_index, success, result/error
    all_completed = pyqtSignal()

    def __init__(self, tasks: List[VideoTask], api_url: str, api_key: str, max_workers: int = 50, config: dict = None):
        super().__init__()
        self.tasks = tasks
        self.api_url = api_url
        self.api_key = api_key
        self.max_workers = max_workers
        self.config = config or {}
        self._stop_flag = False
        self._mutex = QMutex()
        print(f"[APIWorker] 初始化完成 - 任务数: {len(tasks)}, API地址: {api_url}, 并发数: {max_workers}")
        print(f"[APIWorker] 自动重试: {self.config.get('auto_retry', False)}, 最大重试次数: {self.config.get('max_retries', 3)}")

    def stop(self):
        print("[APIWorker] 收到停止信号")
        self._mutex.lock()
        self._stop_flag = True
        self._mutex.unlock()

    def is_stopped(self):
        self._mutex.lock()
        stopped = self._stop_flag
        self._mutex.unlock()
        return stopped

    def get_model_name(self, resolution: str, duration: str) -> str:
        """根据分辨率和时长获取模型名称"""
        # 默认模型映射
        res_map = {
            "portrait": "portrait",
            "landscape": "landscape",
            "竖屏": "portrait",
            "横屏": "landscape",
            "1080x1920": "portrait",
            "1920x1080": "landscape",
        }

        dur_map = {
            "5": "5s",
            "10": "10s",
            "5s": "5s",
            "10s": "10s",
        }

        res_key = res_map.get(resolution.lower().strip(), "landscape")
        dur_key = dur_map.get(duration.lower().strip(), "10s")

        return f"sora-video-{res_key}-{dur_key}"

    def call_api(self, task: VideoTask) -> tuple:
        """调用 API 生成视频"""
        print(f"[API] 任务 {task.row_index} 开始调用API - prompt: {task.prompt[:50]}...")

        if self.is_stopped():
            print(f"[API] 任务 {task.row_index} 已取消")
            return False, "任务已取消"

        model = "sora-video-portrait-10s"
        print(f"[API] 任务 {task.row_index} 使用模型: {model}")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": task.prompt}],
            "stream": True
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            print(f"[API] 任务 {task.row_index} 发送请求...")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=300  # 5分钟超时
            )

            print(f"[API] 任务 {task.row_index} 响应状态码: {response.status_code}")

            if response.status_code != 200:
                print(f"[API] 任务 {task.row_index} HTTP错误: {response.status_code}")
                return False, f"HTTP错误: {response.status_code}"

            video_url = None

            for line in response.iter_lines():
                if self.is_stopped():
                    return False, "任务已取消"

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

                            # 检查进度
                            reasoning = delta.get('reasoning_content')
                            if reasoning and isinstance(reasoning, dict):
                                progress = reasoning.get('progress', 0)
                                message = reasoning.get('message', '')
                                if progress:
                                    print(f"[API] 任务 {task.row_index} 进度: {progress}% - {message}")
                                    self.task_progress.emit(task.row_index, progress, message)

                            # 检查最终结果
                            content = delta.get('content')
                            if content:
                                try:
                                    result = json.loads(content)
                                    if result.get('type') == 'video':
                                        video_url = result.get('url')
                                        print(f"[API] 任务 {task.row_index} 获取到视频URL: {video_url}")
                                except json.JSONDecodeError:
                                    pass
                    except json.JSONDecodeError:
                        continue

            if video_url:
                print(f"[API] 任务 {task.row_index} 成功完成")
                return True, video_url
            else:
                print(f"[API] 任务 {task.row_index} 失败: 未获取到视频URL")
                return False, "未获取到视频URL"

        except requests.exceptions.Timeout:
            print(f"[API] 任务 {task.row_index} 请求超时")
            return False, "请求超时"
        except requests.exceptions.RequestException as e:
            print(f"[API] 任务 {task.row_index} 请求错误: {str(e)}")
            return False, f"请求错误: {str(e)}"
        except Exception as e:
            print(f"[API] 任务 {task.row_index} 未知错误: {str(e)}")
            return False, f"未知错误: {str(e)}"

    def process_task(self, task: VideoTask):
        """处理单个任务（包含重试逻辑）"""
        self.task_started.emit(task.row_index)

        max_retries = self.config.get("max_retries", 3)
        auto_retry = self.config.get("auto_retry", False)

        # 首次尝试
        success, result = self.call_api(task)

        # 如果失败且启用自动重试
        if not success and auto_retry:
            while task.retry_count < max_retries and not self.is_stopped():
                task.retry_count += 1
                print(f"[重试] 任务 {task.row_index} 开始第 {task.retry_count} 次重试...")
                self.task_progress.emit(task.row_index, 0, f"重试 {task.retry_count}/{max_retries}")
                success, result = self.call_api(task)
                if success:
                    print(f"[重试] 任务 {task.row_index} 第 {task.retry_count} 次重试成功")
                    break
                else:
                    print(f"[重试] 任务 {task.row_index} 第 {task.retry_count} 次重试失败")

        self.task_completed.emit(task.row_index, success, result)
        return task.row_index, success, result

    def run(self):
        """运行线程池"""
        import time
        print(f"[APIWorker] 线程池启动 - 并发数: {self.max_workers}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            # 每秒提交一个任务，避免一次性全部提交
            for i, task in enumerate(self.tasks):
                if self.is_stopped():
                    print("[APIWorker] 检测到停止信号，停止提交新任务")
                    break
                future = executor.submit(self.process_task, task)
                futures[future] = task
                print(f"[APIWorker] 已提交任务 {i + 1}/{len(self.tasks)}")
                # 除了最后一个任务，每次提交后等待1秒
                if i < len(self.tasks) - 1:
                    time.sleep(1)

            print(f"[APIWorker] 共提交 {len(futures)} 个任务，等待完成...")

            for future in as_completed(futures):
                if self.is_stopped():
                    print("[APIWorker] 检测到停止信号，正在关闭线程池...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    future.result()
                except Exception as e:
                    task = futures[future]
                    print(f"[APIWorker] 任务 {task.row_index} 异常: {str(e)}")
                    self.task_completed.emit(task.row_index, False, str(e))

        print("[APIWorker] 所有任务处理完成")
        self.all_completed.emit()


class DownloadWorker(QThread):
    """批量下载工作线程"""
    # 信号定义
    task_started = pyqtSignal(int, str)  # row_index, message
    task_completed = pyqtSignal(int, bool, str)  # row_index, success, result/error
    all_completed = pyqtSignal(int, int)  # success_count, fail_count

    def __init__(self, tasks: List[VideoTask], config: dict, save_dir: str, max_workers: int = 20):
        super().__init__()
        self.tasks = tasks
        self.config = config
        self.save_dir = save_dir
        self.max_workers = max_workers
        self._stop_flag = False
        self._mutex = QMutex()
        print(f"[DownloadWorker] 初始化完成 - 任务数: {len(tasks)}, 保存目录: {save_dir}, 并发数: {max_workers}")

    def stop(self):
        print("[DownloadWorker] 收到停止信号")
        self._mutex.lock()
        self._stop_flag = True
        self._mutex.unlock()

    def is_stopped(self):
        self._mutex.lock()
        stopped = self._stop_flag
        self._mutex.unlock()
        return stopped

    def generate_title(self, prompt: str) -> str:
        """调用AI生成视频标题"""
        api_key = self.config.get("llm_api_key", "")
        api_proxy = self.config.get("llm_api_proxy", "https://api.openai.com/v1")
        model = self.config.get("llm_model", "gpt-3.5-turbo")
        style = self.config.get("title_style", "简洁")
        custom_prompt = self.config.get("title_prompt", "")

        print(f"\n[AI标题] ========== 开始生成标题 ==========")
        print(f"[AI标题] Prompt: {prompt[:100]}")
        print(f"[AI标题] 配置 - API Key: {'**' if api_key else '未设置'}")
        print(f"[AI标题] 配置 - API Proxy: {api_proxy}")
        print(f"[AI标题] 配置 - Model: {model}")
        print(f"[AI标题] 配置 - Style: {style}")
        print(f"[AI标题] 配置 - Custom Prompt: {custom_prompt[:50] if custom_prompt else '未设置'}")

        if not api_key:
            print(f"[AI标题] 警告：API Key未设置，使用降级方案")
            fallback = prompt[:20].replace("/", "_").replace("\\", "_")
            print(f"[AI标题] 降级标题: {fallback}")
            return fallback

        # 风格提示词
        style_prompts = {
            "简洁": "生成一个简洁的视频名称，不超过10个字",
            "创意": "生成一个有创意、吸引眼球的视频名称",
            "专业": "生成一个专业正式的视频名称",
            "情感": "生成一个能引起情感共鸣的视频名称",
            "悬念": "生成一个带有悬念感的视频名称",
        }
        style_hint = style_prompts.get(style, style_prompts["简洁"])

        system_prompt = f"""你是一个视频命名专家。{style_hint}。
{custom_prompt}

要求：
1. 直接输出视频名称，必须有内容
2. 不要思考、分析或推理，直接给出名称
3. 不要使用引号包裹
4. 名称要与内容相关
5. 不要有特殊字符如 / \\ : * ? " < > |"""

        user_message = f"视频内容：{prompt}"

        try:
            url = f"{api_proxy.rstrip('/')}/chat/completions"
            print(f"[AI标题] 请求URL: {url}")

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            print(f"[AI标题] 请求头:")
            print(f"[AI标题]   - Authorization: Bearer {api_key[:10]}...")
            print(f"[AI标题]   - Content-Type: application/json")

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.8,
                "max_tokens": 8196,
                "top_p": 1.0
            }
            print(f"[AI标题] 请求体:")
            print(f"[AI标题]   - model: {payload['model']}")
            print(f"[AI标题]   - messages 数量: {len(payload['messages'])}")
            print(f"[AI标题]   - messages[0]:")
            print(f"[AI标题]     role: {payload['messages'][0]['role']}")
            print(f"[AI标题]     content 长度: {len(payload['messages'][0]['content'])}")
            print(f"[AI标题]     content (前200字): {payload['messages'][0]['content'][:200]}")
            print(f"[AI标题]   - messages[1]:")
            print(f"[AI标题]     role: {payload['messages'][1]['role']}")
            print(f"[AI标题]     content: {payload['messages'][1]['content']}")
            print(f"[AI标题]   - temperature: {payload['temperature']}")
            print(f"[AI标题]   - max_tokens: {payload['max_tokens']}")
            print(f"[AI标题]   - top_p: {payload['top_p']}")
            print(f"[AI标题] 完整请求JSON:")
            print(f"[AI标题] {json.dumps(payload, ensure_ascii=False, indent=2)}")

            print(f"[AI标题] 发送请求中...")
            response = requests.post(url, headers=headers, json=payload, timeout=300)

            print(f"[AI标题] 响应状态码: {response.status_code}")
            print(f"[AI标题] 响应头: {dict(response.headers)}")

            if response.status_code != 200:
                print(f"[AI标题] 错误：HTTP {response.status_code}")
                print(f"[AI标题] 响应内容: {response.text[:500]}")
                response.raise_for_status()

            result = response.json()
            print(f"[AI标题] 响应JSON:")
            print(f"[AI标题]   - keys: {list(result.keys())}")
            if 'choices' in result:
                print(f"[AI标题]   - choices 数量: {len(result['choices'])}")
                if result['choices']:
                    choice = result['choices'][0]
                    print(f"[AI标题]   - choices[0] keys: {list(choice.keys())}")
                    if 'finish_reason' in choice:
                        finish_reason = choice['finish_reason']
                        print(f"[AI标题]   - finish_reason: {finish_reason}")
                        if finish_reason == "length":
                            print(f"[AI标题]   ⚠️ 警告：finish_reason=length，说明模型输出被截断")
                        elif finish_reason == "stop":
                            print(f"[AI标题]   ✓ finish_reason=stop，正常完成")
                    if 'message' in choice:
                        msg = choice['message']
                        print(f"[AI标题]   - message keys: {list(msg.keys())}")
                        if 'content' in msg:
                            content = msg['content']
                            print(f"[AI标题]   - content 长度: {len(content)}")
                            print(f"[AI标题]   - content: '{content}'")
                            if not content or content.strip() == "":
                                print(f"[AI标题] 警告：content 为空！")
            if 'usage' in result:
                usage = result['usage']
                print(f"[AI标题]   - usage:")
                print(f"[AI标题]     prompt_tokens: {usage.get('prompt_tokens', 0)}")
                print(f"[AI标题]     completion_tokens: {usage.get('completion_tokens', 0)}")
                completion_details = usage.get('completion_tokens_details', {})
                if completion_details:
                    print(f"[AI标题]     completion_tokens_details:")
                    print(f"[AI标题]       text_tokens: {completion_details.get('text_tokens', 0)}")
                    print(f"[AI标题]       reasoning_tokens: {completion_details.get('reasoning_tokens', 0)}")
                    if completion_details.get('reasoning_tokens', 0) > 0 and completion_details.get('text_tokens', 0) == 0:
                        print(f"[AI标题]       ⚠️ 警告：只有推理token，没有文本token！")
                print(f"[AI标题]     total_tokens: {usage.get('total_tokens', 0)}")

            title = result['choices'][0]['message']['content'].strip()
            print(f"[AI标题] 原始标题: '{title}'")
            print(f"[AI标题] 原始标题长度: {len(title)}")

            if not title:
                print(f"[AI标题] 警告：标题为空，使用降级方案")
                fallback = prompt[:20]
                for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']:
                    fallback = fallback.replace(char, '_')
                print(f"[AI标题] ========== 标题生成失败（内容为空）==========")
                print(f"[AI标题] 降级标题: {fallback}\n")
                return fallback

            # 清理特殊字符
            original_title = title
            for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']:
                title = title.replace(char, '_')

            if title != original_title:
                print(f"[AI标题] 清理后标题: {title}")

            final_title = title.strip('"\'""''')
            print(f"[AI标题] 最终标题: '{final_title}'")
            print(f"[AI标题] ========== 标题生成成功 ==========\n")
            return final_title

        except requests.exceptions.Timeout as e:
            print(f"[AI标题] 错误：请求超时 - {str(e)}")
            print(f"[AI标题] 使用降级方案")
        except requests.exceptions.ConnectionError as e:
            print(f"[AI标题] 错误：连接失败 - {str(e)}")
            print(f"[AI标题] 请检查 API Proxy 地址是否正确: {api_proxy}")
            print(f"[AI标题] 使用降级方案")
        except requests.exceptions.RequestException as e:
            print(f"[AI标题] 错误：请求异常 - {str(e)}")
            print(f"[AI标题] 使用降级方案")
        except json.JSONDecodeError as e:
            print(f"[AI标题] 错误：JSON解析失败 - {str(e)}")
            print(f"[AI标题] 响应内容: {response.text[:500]}")
            print(f"[AI标题] 使用降级方案")
        except KeyError as e:
            print(f"[AI标题] 错误：响应格式错误，缺少字段 {str(e)}")
            print(f"[AI标题] 完整响应: {result}")
            print(f"[AI标题] 使用降级方案")
        except Exception as e:
            print(f"[AI标题] 错误：未知异常 - {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[AI标题] 堆栈跟踪:")
            print(traceback.format_exc())
            print(f"[AI标题] 使用降级方案")

        # 失败时使用prompt前20个字符
        fallback = prompt[:20]
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']:
            fallback = fallback.replace(char, '_')
        print(f"[AI标题] ========== 标题生成失败，使用降级标题 ==========")
        print(f"[AI标题] 降级标题: {fallback}\n")
        return fallback

    def download_video(self, task: VideoTask) -> tuple:
        """下载单个视频"""
        if self.is_stopped():
            return False, "任务已取消"

        if not task.video_url:
            return False, "没有视频URL"

        try:
            # 生成AI标题
            self.task_started.emit(task.row_index, "正在生成标题...")
            title = self.generate_title(task.prompt)
            print(f"[下载] 任务 {task.row_index} 生成标题: {title}")

            if self.is_stopped():
                return False, "任务已取消"

            # 下载视频
            self.task_started.emit(task.row_index, "正在下载...")
            response = requests.get(task.video_url, stream=True, timeout=300)
            response.raise_for_status()

            # 保存文件
            file_ext = ".mp4"
            file_name = f"{title}{file_ext}"
            file_path = os.path.join(self.save_dir, file_name)

            # 如果文件已存在，添加序号
            counter = 1
            while os.path.exists(file_path):
                file_name = f"{title}_{counter}{file_ext}"
                file_path = os.path.join(self.save_dir, file_name)
                counter += 1

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.is_stopped():
                        return False, "任务已取消"
                    f.write(chunk)

            print(f"[下载] 任务 {task.row_index} 下载完成: {file_path}")
            return True, file_path

        except requests.exceptions.Timeout:
            return False, "下载超时"
        except requests.exceptions.RequestException as e:
            return False, f"下载错误: {str(e)}"
        except Exception as e:
            return False, f"未知错误: {str(e)}"

    def process_task(self, task: VideoTask):
        """处理单个下载任务"""
        success, result = self.download_video(task)
        self.task_completed.emit(task.row_index, success, result)
        return task.row_index, success, result

    def run(self):
        """运行下载线程池"""
        print(f"[DownloadWorker] 线程池启动 - 并发数: {self.max_workers}")
        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.process_task, task): task for task in self.tasks}
            print(f"[DownloadWorker] 已提交 {len(futures)} 个下载任务")

            for future in as_completed(futures):
                if self.is_stopped():
                    print("[DownloadWorker] 检测到停止信号，正在关闭线程池...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    _, success, _ = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    task = futures[future]
                    print(f"[DownloadWorker] 任务 {task.row_index} 异常: {str(e)}")
                    self.task_completed.emit(task.row_index, False, str(e))
                    fail_count += 1

        print(f"[DownloadWorker] 所有下载任务处理完成 - 成功: {success_count}, 失败: {fail_count}")
        self.all_completed.emit(success_count, fail_count)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.tasks: List[VideoTask] = []
        self.worker: Optional[APIWorker] = None
        self.download_worker: Optional[DownloadWorker] = None
        self.config = load_config()  # 加载配置
        self.auto_download_dir: Optional[str] = None  # 自动下载目录
        self.init_ui()
        print("[MainWindow] 主窗口初始化完成")

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("Sora 视频批量生成工具")
        self.setMinimumSize(1000, 700)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout(central_widget)

        # === 配置区域 ===
        config_group = QGroupBox("API 配置")
        config_layout = QHBoxLayout(config_group)

        config_layout.addWidget(QLabel("API地址:"))
        self.api_url_input = QLineEdit("http://localhost:8000/v1/chat/completions")
        self.api_url_input.setMinimumWidth(300)
        config_layout.addWidget(self.api_url_input)

        config_layout.addWidget(QLabel("API密钥:"))
        self.api_key_input = QLineEdit("han1234")
        self.api_key_input.setMinimumWidth(150)
        config_layout.addWidget(self.api_key_input)

        config_layout.addWidget(QLabel("并发数:"))
        self.thread_count_spin = QSpinBox()
        self.thread_count_spin.setRange(1, 100)
        self.thread_count_spin.setValue(50)
        config_layout.addWidget(self.thread_count_spin)

        config_layout.addStretch()
        main_layout.addWidget(config_group)

        # === 操作按钮区域 ===
        button_layout = QHBoxLayout()

        self.import_btn = QPushButton("导入 CSV")
        self.import_btn.clicked.connect(self.import_csv)
        button_layout.addWidget(self.import_btn)

        self.start_btn = QPushButton("开始生成")
        self.start_btn.clicked.connect(self.start_generation)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        self.download_btn = QPushButton("批量下载")
        self.download_btn.clicked.connect(self.start_download)
        self.download_btn.setEnabled(False)
        button_layout.addWidget(self.download_btn)

        self.retry_btn = QPushButton("失败重试")
        self.retry_btn.clicked.connect(self.retry_failed)
        self.retry_btn.setEnabled(False)
        button_layout.addWidget(self.retry_btn)

        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        button_layout.addStretch()

        # 设置按钮放在右侧
        self.settings_btn = QPushButton("设置")
        self.settings_btn.clicked.connect(self.open_settings)
        button_layout.addWidget(self.settings_btn)

        main_layout.addLayout(button_layout)

        # === 进度条 ===
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("进度: 0/0")
        progress_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        # === 表格 ===
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "图片路径", "提示词", "分辨率", "时长", "状态", "进度", "视频URL", "下载路径"
        ])

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.Stretch)

        main_layout.addWidget(self.table)

        # === 状态栏 ===
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("就绪 - 请导入 CSV 文件")

        # 统计标签
        self.stats_label = QLabel("成功: 0 | 失败: 0 | 待处理: 0")
        self.statusBar.addPermanentWidget(self.stats_label)

    def import_csv(self):
        """导入 CSV 文件"""
        print("[CSV] 开始导入CSV文件...")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV 文件", "", "CSV 文件 (*.csv);;所有文件 (*)"
        )

        if not file_path:
            print("[CSV] 用户取消选择文件")
            return

        print(f"[CSV] 选择文件: {file_path}")

        try:
            self.tasks.clear()
            self.table.setRowCount(0)

            # 状态映射：从中文状态值到枚举
            status_map = {
                "待处理": TaskStatus.PENDING,
                "处理中": TaskStatus.PROCESSING,
                "成功": TaskStatus.SUCCESS,
                "失败": TaskStatus.FAILED,
            }

            # 自动检测文件编码
            detected_encoding = detect_encoding(file_path)
            print(f"[CSV] 使用编码: {detected_encoding}")

            with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
                reader = csv.DictReader(f)

                for row_index, row in enumerate(reader):
                    # 解析状态（支持导出的结果文件）
                    status_str = row.get('status', '').strip()
                    status = status_map.get(status_str, TaskStatus.PENDING)

                    # 获取视频URL（支持导出的结果文件）
                    video_url = row.get('video_url', '').strip() or None

                    # 获取错误信息
                    error_msg = row.get('error_msg', '').strip() or None

                    # 获取下载路径
                    download_path = row.get('download_path', '').strip() or None

                    task = VideoTask(
                        row_index=row_index,
                        image_path=row.get('image_path', ''),
                        prompt=row.get('prompt', ''),
                        resolution=row.get('resolution', 'landscape'),
                        duration=row.get('duration', '10s'),
                        status=status,
                        progress=100 if status == TaskStatus.SUCCESS else 0,
                        video_url=video_url,
                        error_msg=error_msg,
                        download_path=download_path
                    )
                    self.tasks.append(task)
                    self.add_table_row(task)

            # 统计导入结果
            success_count = sum(1 for t in self.tasks if t.status == TaskStatus.SUCCESS)
            failed_count = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
            pending_count = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)

            print(f"[CSV] 成功导入 {len(self.tasks)} 条记录")
            print(f"[CSV] 状态统计 - 成功: {success_count}, 失败: {failed_count}, 待处理: {pending_count}")

            has_tasks = len(self.tasks) > 0
            self.start_btn.setEnabled(has_tasks and pending_count > 0)
            self.download_btn.setEnabled(has_tasks and success_count > 0)
            self.retry_btn.setEnabled(has_tasks and failed_count > 0)
            self.export_btn.setEnabled(has_tasks)
            self.update_stats()

            msg = f"已导入 {len(self.tasks)} 条记录"
            if success_count > 0 or failed_count > 0:
                msg += f" (成功: {success_count}, 失败: {failed_count}, 待处理: {pending_count})"
            self.statusBar.showMessage(msg)

        except Exception as e:
            print(f"[CSV] 导入错误: {str(e)}")
            QMessageBox.critical(self, "导入错误", f"无法读取 CSV 文件:\n{str(e)}")

    def add_table_row(self, task: VideoTask):
        """添加表格行"""
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(task.image_path))
        self.table.setItem(row, 1, QTableWidgetItem(task.prompt))
        self.table.setItem(row, 2, QTableWidgetItem(task.resolution))
        self.table.setItem(row, 3, QTableWidgetItem(task.duration))

        # 状态列
        status_item = QTableWidgetItem(task.status.value)
        if task.status == TaskStatus.SUCCESS:
            status_item.setBackground(QColor(144, 238, 144))  # 浅绿
        elif task.status == TaskStatus.FAILED:
            status_item.setBackground(QColor(255, 182, 193))  # 浅红
        self.table.setItem(row, 4, status_item)

        # 进度列
        progress_text = f"{task.progress}%" if task.progress > 0 else "0%"
        self.table.setItem(row, 5, QTableWidgetItem(progress_text))

        # 视频URL列
        self.table.setItem(row, 6, QTableWidgetItem(task.video_url or ""))

        # 下载路径列
        self.table.setItem(row, 7, QTableWidgetItem(task.download_path or ""))

    def update_table_row(self, row_index: int, status: TaskStatus, progress: int = 0,
                         video_url: str = "", message: str = "", download_path: str = ""):
        """更新表格行"""
        if row_index >= self.table.rowCount():
            return

        # 更新状态
        status_item = self.table.item(row_index, 4)
        if status_item:
            status_item.setText(status.value)

            # 设置颜色
            if status == TaskStatus.SUCCESS:
                status_item.setBackground(QColor(144, 238, 144))  # 浅绿
            elif status == TaskStatus.FAILED:
                status_item.setBackground(QColor(255, 182, 193))  # 浅红
            elif status == TaskStatus.PROCESSING:
                status_item.setBackground(QColor(255, 255, 224))  # 浅黄

        # 更新进度
        progress_item = self.table.item(row_index, 5)
        if progress_item:
            progress_item.setText(f"{progress}%")

        # 更新视频URL
        if video_url:
            url_item = self.table.item(row_index, 6)
            if url_item:
                url_item.setText(video_url)

        # 更新下载路径
        if download_path:
            download_item = self.table.item(row_index, 7)
            if download_item:
                download_item.setText(download_path)

    def update_stats(self):
        """更新统计信息"""
        success = sum(1 for t in self.tasks if t.status == TaskStatus.SUCCESS)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        pending = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)
        processing = sum(1 for t in self.tasks if t.status == TaskStatus.PROCESSING)

        self.stats_label.setText(f"成功: {success} | 失败: {failed} | 处理中: {processing} | 待处理: {pending}")

        total = len(self.tasks)
        completed = success + failed
        self.progress_label.setText(f"进度: {completed}/{total}")
        if total > 0:
            self.progress_bar.setValue(int(completed * 100 / total))

    def start_generation(self):
        """开始生成"""
        print("[生成] 开始生成视频...")
        if not self.tasks:
            print("[生成] 错误: 没有任务")
            QMessageBox.warning(self, "警告", "请先导入 CSV 文件")
            return

        # 获取待处理的任务
        pending_tasks = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        print(f"[生成] 待处理任务数: {len(pending_tasks)}")

        if not pending_tasks:
            QMessageBox.information(self, "提示", "没有待处理的任务")
            return

        # 检查是否开启自动下载
        if self.config.get("auto_download", False):
            # 检查是否有下载路径
            tasks_with_path = [t for t in pending_tasks if t.download_path]
            tasks_without_path = [t for t in pending_tasks if not t.download_path]

            if tasks_without_path:
                # 有任务没有下载路径，询问用户选择目录
                reply = QMessageBox.question(
                    self, "自动下载",
                    f"已开启自动下载功能，有 {len(tasks_without_path)} 个任务没有指定下载路径。\n"
                    "是否选择一个默认下载目录？\n\n"
                    "点击「是」选择下载目录，点击「否」跳过自动下载。",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    save_dir = QFileDialog.getExistingDirectory(
                        self, "选择自动下载目录", ""
                    )
                    if save_dir:
                        self.auto_download_dir = save_dir
                        print(f"[生成] 自动下载目录: {save_dir}")
                    else:
                        print("[生成] 用户取消选择目录，禁用自动下载")
                        self.auto_download_dir = None
                else:
                    self.auto_download_dir = None
            else:
                # 所有任务都有下载路径，使用各自的路径
                self.auto_download_dir = "use_task_path"
                print("[生成] 所有任务都有下载路径，将使用各自的路径")
        else:
            self.auto_download_dir = None

        # 禁用按钮
        self.import_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.retry_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        # 创建工作线程
        self.worker = APIWorker(
            tasks=pending_tasks,
            api_url=self.api_url_input.text(),
            api_key=self.api_key_input.text(),
            max_workers=self.thread_count_spin.value(),
            config=self.config
        )

        # 连接信号
        self.worker.task_started.connect(self.on_task_started)
        self.worker.task_progress.connect(self.on_task_progress)
        self.worker.task_completed.connect(self.on_task_completed)
        self.worker.all_completed.connect(self.on_all_completed)

        # 启动
        self.worker.start()
        print("[生成] 工作线程已启动")
        self.statusBar.showMessage("正在生成视频...")

    def stop_generation(self):
        """停止生成或下载"""
        print("[操作] 用户请求停止")
        if self.worker:
            print("[操作] 停止生成任务")
            self.worker.stop()
            self.statusBar.showMessage("正在停止生成...")
        if self.download_worker:
            print("[操作] 停止下载任务")
            self.download_worker.stop()
            self.statusBar.showMessage("正在停止下载...")

    def retry_failed(self):
        """重试所有失败的任务"""
        print("[重试] 开始重试失败任务...")

        # 获取所有失败的任务
        failed_tasks = [t for t in self.tasks if t.status == TaskStatus.FAILED]
        print(f"[重试] 失败任务数: {len(failed_tasks)}")

        if not failed_tasks:
            QMessageBox.information(self, "提示", "没有失败的任务需要重试")
            return

        # 重置失败任务的状态
        for task in failed_tasks:
            task.status = TaskStatus.PENDING
            task.progress = 0
            task.video_url = None
            task.error_msg = None
            task.retry_count = 0  # 重置重试计数
            self.update_table_row(task.row_index, TaskStatus.PENDING, 0, "")

        self.update_stats()

        # 禁用按钮
        self.import_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.retry_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.export_btn.setEnabled(False)

        # 创建工作线程
        self.worker = APIWorker(
            tasks=failed_tasks,
            api_url=self.api_url_input.text(),
            api_key=self.api_key_input.text(),
            max_workers=self.thread_count_spin.value(),
            config=self.config
        )

        # 连接信号
        self.worker.task_started.connect(self.on_task_started)
        self.worker.task_progress.connect(self.on_task_progress)
        self.worker.task_completed.connect(self.on_task_completed)
        self.worker.all_completed.connect(self.on_all_completed)

        # 启动
        self.worker.start()
        print(f"[重试] 工作线程已启动，共 {len(failed_tasks)} 个任务")
        self.statusBar.showMessage(f"正在重试 {len(failed_tasks)} 个失败任务...")

    def open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self, self.config)
        if dialog.exec_() == QDialog.Accepted:
            self.config = dialog.get_config()
            self.statusBar.showMessage("设置已保存")
            print("[设置] 配置已更新")

    def start_download(self):
        """开始批量下载"""
        print("[下载] 开始批量下载...")

        # 检查是否有生成URL的任务
        download_tasks = [t for t in self.tasks if t.video_url and t.status == TaskStatus.SUCCESS]
        print(f"[下载] 可下载任务数: {len(download_tasks)}")

        if not download_tasks:
            QMessageBox.warning(self, "警告", "没有已生成的视频可以下载")
            return

        # 选择保存目录
        save_dir = QFileDialog.getExistingDirectory(
            self, "选择保存目录", ""
        )

        if not save_dir:
            print("[下载] 用户取消选择目录")
            return

        print(f"[下载] 保存目录: {save_dir}")

        # 禁用按钮
        self.import_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.download_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

        # 创建下载工作线程
        self.download_worker = DownloadWorker(
            tasks=download_tasks,
            config=self.config,
            save_dir=save_dir,
            max_workers=self.config.get("download_threads", 20)
        )

        # 连接信号
        self.download_worker.task_started.connect(self.on_download_task_started)
        self.download_worker.task_completed.connect(self.on_download_task_completed)
        self.download_worker.all_completed.connect(self.on_download_all_completed)

        # 启动
        self.download_worker.start()
        print("[下载] 工作线程已启动")
        self.statusBar.showMessage("正在下载视频...")

    def on_download_task_started(self, row_index: int, message: str):
        """下载任务开始"""
        print(f"[回调] 下载任务 {row_index}: {message}")
        if row_index < len(self.tasks):
            self.update_table_row(row_index, TaskStatus.PROCESSING, message=message)

    def on_download_task_completed(self, row_index: int, success: bool, result: str):
        """下载任务完成"""
        print(f"[回调] 下载任务 {row_index} 完成 - 成功: {success}")
        if row_index < len(self.tasks):
            task = self.tasks[row_index]
            if success:
                # 更新下载路径信息到错误消息字段
                task.error_msg = f"已下载: {result}"
                self.update_table_row(row_index, TaskStatus.SUCCESS, 100, message="已下载")
            else:
                task.error_msg = result
                self.update_table_row(row_index, TaskStatus.FAILED, 0, message=result)

    def on_download_all_completed(self, success_count: int, fail_count: int):
        """所有下载任务完成"""
        print(f"[回调] 所有下载任务处理完成 - 成功: {success_count}, 失败: {fail_count}")

        self.import_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.download_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True)

        # 检查是否有失败的生成任务，启用重试按钮
        failed_gen = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        self.retry_btn.setEnabled(failed_gen > 0)

        self.statusBar.showMessage(f"下载完成! 成功: {success_count}, 失败: {fail_count}")

        QMessageBox.information(
            self, "下载完成",
            f"视频下载完成!\n成功: {success_count}\n失败: {fail_count}"
        )

    def on_task_started(self, row_index: int):
        """任务开始"""
        print(f"[回调] 任务 {row_index} 开始处理")
        if row_index < len(self.tasks):
            self.tasks[row_index].status = TaskStatus.PROCESSING
            self.update_table_row(row_index, TaskStatus.PROCESSING)
            self.update_stats()

    def on_task_progress(self, row_index: int, progress: int, message: str):
        """任务进度更新"""
        if row_index < len(self.tasks):
            self.tasks[row_index].progress = progress
            self.update_table_row(row_index, TaskStatus.PROCESSING, progress, message=message)

    def on_task_completed(self, row_index: int, success: bool, result: str):
        """任务完成"""
        print(f"[回调] 任务 {row_index} 完成 - 成功: {success}")
        if row_index < len(self.tasks):
            task = self.tasks[row_index]
            if success:
                task.status = TaskStatus.SUCCESS
                task.video_url = result
                task.progress = 100
                self.update_table_row(row_index, TaskStatus.SUCCESS, 100, result)

                # 检查是否需要自动下载
                if self.auto_download_dir:
                    self.auto_download_single_task(task)
            else:
                task.status = TaskStatus.FAILED
                task.error_msg = result
                self.update_table_row(row_index, TaskStatus.FAILED, 0, result)

            self.update_stats()

    def auto_download_single_task(self, task: VideoTask):
        """自动下载单个任务"""
        if not task.video_url:
            print(f"[自动下载] 任务 {task.row_index} 没有视频URL，跳过")
            return

        # 确定下载目录
        if self.auto_download_dir == "use_task_path":
            # 使用任务自己的下载路径
            if task.download_path:
                save_dir = os.path.dirname(task.download_path)
                if not save_dir:
                    save_dir = task.download_path
            else:
                print(f"[自动下载] 任务 {task.row_index} 没有指定下载路径，跳过")
                return
        else:
            save_dir = self.auto_download_dir

        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                print(f"[自动下载] 创建目录失败: {e}")
                return

        print(f"[自动下载] 任务 {task.row_index} 开始下载到: {save_dir}")

        # 使用线程进行下载，避免阻塞UI
        download_thread = threading.Thread(
            target=self._do_auto_download,
            args=(task, save_dir),
            daemon=True
        )
        download_thread.start()

    def _do_auto_download(self, task: VideoTask, save_dir: str):
        """执行自动下载（在后台线程中执行）"""
        try:
            # 生成标题
            api_key = self.config.get("llm_api_key", "")
            api_proxy = self.config.get("llm_api_proxy", "https://api.openai.com/v1")
            model = self.config.get("llm_model", "gpt-3.5-turbo")
            style = self.config.get("title_style", "简洁")
            custom_prompt = self.config.get("title_prompt", "")

            title = self._generate_title_for_auto_download(
                task.prompt, api_key, api_proxy, model, style, custom_prompt
            )
            print(f"[自动下载] 任务 {task.row_index} 生成标题: {title}")

            # 下载视频
            response = requests.get(task.video_url, stream=True, timeout=300)
            response.raise_for_status()

            # 保存文件
            file_ext = ".mp4"
            file_name = f"{title}{file_ext}"
            file_path = os.path.join(save_dir, file_name)

            # 如果文件已存在，添加序号
            counter = 1
            while os.path.exists(file_path):
                file_name = f"{title}_{counter}{file_ext}"
                file_path = os.path.join(save_dir, file_name)
                counter += 1

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 更新任务的下载路径
            task.download_path = file_path
            print(f"[自动下载] 任务 {task.row_index} 下载完成: {file_path}")

            # 使用信号更新UI (在主线程中更新)
            QMetaObject.invokeMethod(
                self, "_update_download_path_slot",
                Qt.QueuedConnection,
                Q_ARG(int, task.row_index),
                Q_ARG(str, file_path)
            )

        except Exception as e:
            print(f"[自动下载] 任务 {task.row_index} 下载失败: {e}")

    @pyqtSlot(int, str)
    def _update_download_path_slot(self, row_index: int, file_path: str):
        """更新下载路径的槽函数（在主线程中执行）"""
        if row_index < len(self.tasks):
            self.tasks[row_index].download_path = file_path
            self.update_table_row(row_index, TaskStatus.SUCCESS, 100, download_path=file_path)

    def _generate_title_for_auto_download(self, prompt: str, api_key: str, api_proxy: str,
                                          model: str, style: str, custom_prompt: str) -> str:
        """为自动下载生成标题"""
        if not api_key:
            fallback = prompt[:20].replace("/", "_").replace("\\", "_")
            return fallback

        # 风格提示词
        style_prompts = {
            "简洁": "生成一个简洁的视频名称，不超过10个字",
            "创意": "生成一个有创意、吸引眼球的视频名称",
            "专业": "生成一个专业正式的视频名称",
            "情感": "生成一个能引起情感共鸣的视频名称",
            "悬念": "生成一个带有悬念感的视频名称",
        }
        style_hint = style_prompts.get(style, style_prompts["简洁"])

        system_prompt = f"""你是一个视频命名专家。{style_hint}。
{custom_prompt}

要求：
1. 直接输出视频名称，必须有内容
2. 不要思考、分析或推理，直接给出名称
3. 不要使用引号包裹
4. 名称要与内容相关
5. 不要有特殊字符如 / \\ : * ? " < > |"""

        user_message = f"视频内容：{prompt}"

        try:
            url = f"{api_proxy.rstrip('/')}/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.8,
                "max_tokens": 8196,
                "top_p": 1.0
            }

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                title = result['choices'][0]['message']['content'].strip()
                if title:
                    # 清理特殊字符
                    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']:
                        title = title.replace(char, '_')
                    return title.strip('"\'""''')
        except Exception as e:
            print(f"[自动下载] 生成标题失败: {e}")

        # 失败时使用prompt前20个字符
        fallback = prompt[:20]
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']:
            fallback = fallback.replace(char, '_')
        return fallback

    def on_all_completed(self):
        """所有任务完成"""
        print("[回调] 所有任务处理完成")
        self.import_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.download_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        success = sum(1 for t in self.tasks if t.status == TaskStatus.SUCCESS)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)

        # 如果有失败的任务，启用重试按钮
        self.retry_btn.setEnabled(failed > 0)

        print(f"[统计] 成功: {success}, 失败: {failed}")
        self.statusBar.showMessage(f"生成完成! 成功: {success}, 失败: {failed}")

        QMessageBox.information(
            self, "完成",
            f"视频生成完成!\n成功: {success}\n失败: {failed}"
        )

    def export_results(self):
        """导出结果"""
        print("[导出] 开始导出结果...")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "result.csv", "CSV 文件 (*.csv)"
        )

        if not file_path:
            print("[导出] 用户取消导出")
            return

        print(f"[导出] 保存路径: {file_path}")

        try:
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_path', 'prompt', 'resolution', 'duration',
                               'status', 'video_url', 'error_msg', 'download_path'])

                for task in self.tasks:
                    writer.writerow([
                        task.image_path,
                        task.prompt,
                        task.resolution,
                        task.duration,
                        task.status.value,
                        task.video_url or '',
                        task.error_msg or '',
                        task.download_path or ''
                    ])

            print(f"[导出] 成功导出 {len(self.tasks)} 条记录")
            self.statusBar.showMessage(f"结果已导出到: {file_path}")
            QMessageBox.information(self, "导出成功", f"结果已保存到:\n{file_path}")

        except Exception as e:
            print(f"[导出] 导出错误: {str(e)}")
            QMessageBox.critical(self, "导出错误", f"无法保存文件:\n{str(e)}")

    def closeEvent(self, event):
        """关闭窗口事件"""
        print("[MainWindow] 收到关闭窗口事件")
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "任务正在运行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                print("[MainWindow] 用户确认退出，正在停止任务...")
                self.worker.stop()
                self.worker.wait(5000)
                print("[MainWindow] 任务已停止，退出程序")
                event.accept()
            else:
                print("[MainWindow] 用户取消退出")
                event.ignore()
        else:
            print("[MainWindow] 程序退出")
            event.accept()


def main():
    print("=" * 50)
    print("[启动] Sora 视频批量生成工具")
    print("=" * 50)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用 Fusion 风格

    window = MainWindow()
    window.show()

    print("[启动] 应用程序启动完成")
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
