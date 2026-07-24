"""
LLM 任务规划模块

使用 DeepSeek API 将用户的自然语言指令转换为结构化的机器人任务序列。
通过精心设计的 system prompt 约束输出格式为固定 JSON schema。
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from .example_retriever import get_retriever

# ═══════════════════════════════════════════════════════════
# System Prompt 模板（不含示例，示例在运行时动态注入）
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """你是一个水上机器人任务规划专家。用户会用自然语言描述一个水上作业任务，你要将指令拆解为结构化的子任务序列。

## 输出格式

你必须只输出一个 JSON 对象，不要有任何其他内容。格式如下：

```json
{{
  "tasks": [
    {{
      "action": "动作名",
      "target": "目标类型",
      "reason": "为什么要做这个动作（中文）"
    }}
  ],
  "summary": "一句话概述整个任务（中文）"
}}
```

## 动作类型
- detect: 检测/发现某类物体（trash/obstacle/buoy/boat/anomaly）
- avoid: 避开某类物体（obstacle/hazard）
- navigate: 导航到某个位置或沿某条路径行进
- collect: 收集/清理某个目标（trash/object）
- inspect: 检查/巡检某个目标（buoy/area/boat）
- report: 生成报告/标记异常

## 目标类型
- trash: 漂浮垃圾
- obstacle: 障碍物/礁石/水草
- buoy: 浮标/航标
- boat: 其他船只
- anomaly: 异常物体/可疑目标
- dock: 码头/基地/港口
- area: 区域/水域

{examples}

## 当前任务

请为以下用户指令生成任务规划：

"{user_instruction}"
"""

# ═══════════════════════════════════════════════════════════
# Task Planner 类
# ═══════════════════════════════════════════════════════════

class TaskPlanner:
    """基于 LLM 的水上机器人任务规划器"""

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = DEEPSEEK_MODEL,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        self._initialized = False

    def _ensure_client(self):
        """懒初始化 OpenAI 客户端"""
        if self._initialized:
            return
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self._initialized = True

    def plan(self, user_instruction: str, use_fewshot: bool = True) -> Dict:
        """
        调用 LLM 将自然语言指令拆解为任务序列

        Args:
            user_instruction: 用户的自然语言指令（中文）
            use_fewshot: 是否使用动态 few-shot 示例（默认开启）

        Returns:
            结构化任务 JSON

        Raises:
            RuntimeError: LLM 调用失败或响应解析失败
        """
        self._ensure_client()

        # 构建动态 system prompt
        if use_fewshot:
            try:
                retriever = get_retriever()
                example_text = retriever.build_prompt_examples(user_instruction, top_k=3)
            except Exception:
                example_text = "（示例加载失败，请直接根据规则生成任务规划）"
        else:
            example_text = ""

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            examples=example_text,
            user_instruction=user_instruction,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.choices[0].message.content.strip()
            return self._parse_response(content)

        except Exception as e:
            raise RuntimeError(f"LLM 调用失败: {e}")

    def _parse_response(self, content: str) -> Dict:
        """
        解析 LLM 响应，提取 JSON

        支持两种格式：
        1. 纯 JSON 字符串
        2. 被 ```json ``` 代码块包裹的 JSON
        """
        # 尝试提取代码块中的 JSON
        code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if code_block_match:
            content = code_block_match.group(1)

        # 解析 JSON
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # 尝试查找 JSON 对象的起止
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                try:
                    result = json.loads(content[start:end + 1])
                except json.JSONDecodeError:
                    raise RuntimeError(f"无法解析 LLM 响应为 JSON: {content[:200]}...")
            else:
                raise RuntimeError(f"LLM 响应中未找到 JSON: {content[:200]}...")

        # 验证必要字段
        if "tasks" not in result:
            raise RuntimeError("LLM 响应缺少 'tasks' 字段")
        if not isinstance(result["tasks"], list):
            raise RuntimeError("'tasks' 字段必须是数组")

        return result

    def format_plan_display(self, plan: Dict) -> str:
        """
        将任务规划格式化为人可读的展示文本

        Args:
            plan: plan() 方法返回的结构化任务 JSON

        Returns:
            格式化的多行文本
        """
        lines = []
        lines.append(f"[规划] {plan.get('summary', '任务规划')}")
        lines.append("")

        for i, task in enumerate(plan.get("tasks", []), 1):
            action_label = {
                "detect": "[检测]",
                "avoid": "[避障]",
                "navigate": "[导航]",
                "collect": "[收集]",
                "inspect": "[巡检]",
                "report": "[报告]",
            }.get(task.get("action", ""), "[执行]")

            target_name = {
                "trash": "垃圾",
                "obstacle": "障碍物",
                "buoy": "浮标",
                "boat": "船只",
                "anomaly": "异常物体",
                "dock": "码头",
            }.get(task.get("target", ""), task.get("target", "未知"))

            lines.append(f"  {action_label} 步骤{i}: {task.get('action', '?')} -> {target_name}")
            lines.append(f"     {task.get('reason', '')}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 离线模式（当无法访问 API 时使用规则匹配）
# ═══════════════════════════════════════════════════════════

def rule_based_plan(user_instruction: str) -> Dict:
    """
    基于规则的简单任务规划（离线模式）

    当 DeepSeek API 不可用时，用关键词匹配生成基本任务序列。
    覆盖：清理、巡检、巡逻、运输、救援、复合任务
    """
    instruction = user_instruction.lower()
    tasks = []
    has_return = any(w in instruction for w in ["返回", "回去", "码头", "回港", "回到"])

    # 清理类
    has_cleanup = any(w in instruction for w in ["清理", "收集", "垃圾", "清除", "捞", "打捞"])
    # 巡检类
    has_inspect = any(w in instruction for w in ["巡检", "检查", "排查", "查看", "看看", "检修"])
    # 巡逻类
    has_patrol = any(w in instruction for w in ["巡逻", "巡视", "巡航", "侦察"])
    # 救援类
    has_rescue = any(w in instruction for w in ["救", "搜救", "救援", "落水", "失踪", "被困"])
    # 运输类
    has_transport = any(w in instruction for w in ["运送", "运输", "送", "搬到", "搬上", "拖", "护送", "投送"])
    # 避障
    has_avoid = any(w in instruction for w in ["避开", "障碍", "绕开", "绕过", "绕"])
    # 报告
    has_report = any(w in instruction for w in ["报告", "上报", "标记", "记录", "汇报"])

    # 救援优先
    if has_rescue:
        tasks.append({
            "action": "detect", "target": "anomaly",
            "reason": "紧急扫描定位搜救目标",
        })
        tasks.append({
            "action": "navigate", "target": "anomaly",
            "reason": "全速驶向目标位置",
        })
        tasks.append({
            "action": "report", "target": "anomaly",
            "reason": "发现目标后立即发送位置信息",
        })

    # 清理任务
    if has_cleanup:
        tasks.append({
            "action": "detect", "target": "trash",
            "reason": "扫描水域，定位所有垃圾/漂浮物",
        })
        if has_avoid:
            tasks.append({
                "action": "avoid", "target": "obstacle",
                "reason": "规划路径时避开障碍物",
            })
        tasks.append({
            "action": "collect", "target": "trash",
            "reason": "依次前往每个位置进行收集/清理",
        })

    # 巡检任务
    if has_inspect:
        if not has_cleanup:
            tasks.append({
                "action": "navigate", "target": "buoy" if "浮标" in instruction else "area",
                "reason": "前往巡检目标位置",
            })
        tasks.append({
            "action": "inspect",
            "target": "buoy" if "浮标" in instruction else "area",
            "reason": "对目标进行状态检查",
        })
        if any(w in instruction for w in ["管道", "堤坝", "裂缝", "设备"]):
            tasks.append({
                "action": "inspect", "target": "area",
                "reason": "近距离详细检查指定设施",
            })
        if has_report:
            tasks.append({
                "action": "report", "target": "anomaly",
                "reason": "记录巡检结果并生成报告",
            })

    # 巡逻任务
    if has_patrol:
        tasks.append({
            "action": "navigate", "target": "area",
            "reason": "按照预设/指定航线巡航水域",
        })
        tasks.append({
            "action": "detect", "target": "anomaly",
            "reason": "巡航中持续检测异常物体或可疑活动",
        })
        if "可疑" in instruction or "非法" in instruction:
            tasks.append({
                "action": "inspect", "target": "boat",
                "reason": "靠近可疑目标进行观察确认",
            })
        tasks.append({
            "action": "report", "target": "anomaly",
            "reason": "生成巡逻报告，标记所有发现",
        })

    # 运输任务
    if has_transport:
        tasks.append({
            "action": "navigate", "target": "dock" if "码头" in instruction else "area",
            "reason": "前往取货/装载地点",
        })
        if any(w in instruction for w in ["浮标", "送", "每个"]):
            tasks.append({
                "action": "navigate", "target": "buoy",
                "reason": "依次前往各配送目标位置",
            })
        tasks.append({
            "action": "navigate",
            "target": "dock" if has_return else "area",
            "reason": "返回指定位置" if has_return else "前往目标交付地点",
        })

    # 返回码头
    if has_return and not has_transport:
        tasks.append({
            "action": "navigate", "target": "dock",
            "reason": "任务完成，返回码头/基地",
        })

    # 默认回退
    if not tasks:
        tasks.append({
            "action": "navigate", "target": "area",
            "reason": "按指令执行水域导航任务",
        })
        tasks.append({
            "action": "detect", "target": "anomaly",
            "reason": "航行中持续监控水面情况",
        })

    return {
        "tasks": tasks,
        "summary": f"根据指令生成的任务规划（规则模式）：{user_instruction}",
    }
