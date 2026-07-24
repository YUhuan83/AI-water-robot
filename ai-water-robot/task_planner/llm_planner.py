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

# ═══════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个水上机器人任务规划专家。用户会用自然语言描述一个水上作业任务，你要将指令拆解为结构化的子任务序列。

## 输出格式

你必须只输出一个 JSON 对象，不要有任何其他内容。格式如下：

```json
{
  "tasks": [
    {
      "action": "动作名",
      "target": "目标类型",
      "reason": "为什么要做这个动作（中文）"
    }
  ],
  "summary": "一句话概述整个任务（中文）"
}
```

## 动作类型
- detect: 检测/发现某类物体（trash/obstacle/buoy/boat/anomaly）
- avoid: 避开某类物体（obstacle/hazard）
- navigate: 导航到某个位置或沿某条路径行进
- collect: 收集/清理某个目标（trash/object）
- inspect: 检查/巡检某个目标（buoy/area）
- report: 生成报告/标记异常

## 目标类型
- trash: 漂浮垃圾
- obstacle: 障碍物/礁石
- buoy: 浮标
- boat: 其他船只
- anomaly: 异常物体
- dock: 码头/基地

## 示例

用户指令："清理河道里的所有漂浮垃圾，避开障碍物"

输出：
```json
{
  "tasks": [
    {"action": "detect", "target": "trash", "reason": "扫描水域，找到所有漂浮垃圾的位置"},
    {"action": "avoid", "target": "obstacle", "reason": "规划路径时需要绕开礁石等障碍物"},
    {"action": "collect", "target": "trash", "reason": "依次前往每个垃圾位置进行收集"}
  ],
  "summary": "清理任务：检测所有漂浮垃圾，规划避开障碍物的最优路径，逐一收集"
}
```

用户指令："从码头出发，依次检查1号和3号浮标，然后返回码头"

输出：
```json
{
  "tasks": [
    {"action": "navigate", "target": "buoy", "reason": "从码头出发前往1号浮标"},
    {"action": "inspect", "target": "buoy", "reason": "到达1号浮标进行巡检"},
    {"action": "navigate", "target": "buoy", "reason": "从1号浮标航行到3号浮标"},
    {"action": "inspect", "target": "buoy", "reason": "到达3号浮标进行巡检"},
    {"action": "navigate", "target": "dock", "reason": "巡检完毕，返回码头"}
  ],
  "summary": "巡检任务：依次检查1号和3号浮标的状态，完成后返回码头"
}
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

    def plan(self, user_instruction: str) -> Dict:
        """
        调用 LLM 将自然语言指令拆解为任务序列

        Args:
            user_instruction: 用户的自然语言指令（中文）

        Returns:
            结构化任务 JSON

        Raises:
            RuntimeError: LLM 调用失败或响应解析失败
        """
        self._ensure_client()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_instruction},
                ],
                temperature=0.3,  # 低温度保证输出稳定
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
    用于开发调试和没有网络的演示场景。
    """
    instruction = user_instruction.lower()

    tasks = []

    # 清理类任务
    if any(w in instruction for w in ["清理", "收集", "垃圾", "清除"]):
        tasks.append({
            "action": "detect",
            "target": "trash",
            "reason": "扫描水域，定位所有漂浮垃圾",
        })
        if any(w in instruction for w in ["避开", "障碍", "绕开"]):
            tasks.append({
                "action": "avoid",
                "target": "obstacle",
                "reason": "规划路径时避开障碍物",
            })
        tasks.append({
            "action": "collect",
            "target": "trash",
            "reason": "按最优路径依次收集所有垃圾",
        })

    # 巡检类任务
    if any(w in instruction for w in ["巡检", "检查", "浮标"]):
        tasks.append({
            "action": "navigate",
            "target": "buoy",
            "reason": "从当前位置出发前往目标浮标",
        })
        tasks.append({
            "action": "inspect",
            "target": "buoy",
            "reason": "对浮标进行状态检查",
        })
        if any(w in instruction for w in ["返回", "回去", "码头"]):
            tasks.append({
                "action": "navigate",
                "target": "dock",
                "reason": "完成巡检，返回码头",
            })

    # 巡逻类任务
    if any(w in instruction for w in ["巡逻", "巡视", "侦察"]):
        tasks.append({
            "action": "navigate",
            "target": "area",
            "reason": "按预定路线巡逻整个水域",
        })
        tasks.append({
            "action": "detect",
            "target": "anomaly",
            "reason": "巡逻中持续扫描，发现异常物体",
        })
        tasks.append({
            "action": "report",
            "target": "anomaly",
            "reason": "标记异常物体位置并生成报告",
        })

    # 默认：导航到目标
    if not tasks:
        tasks.append({
            "action": "navigate",
            "target": "area",
            "reason": "按指令执行水域导航任务",
        })
        tasks.append({
            "action": "detect",
            "target": "anomaly",
            "reason": "航行中持续监控水面情况",
        })

    return {
        "tasks": tasks,
        "summary": "根据指令 \"" + user_instruction + "\" 生成的任务规划（规则模式）",
    }
