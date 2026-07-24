"""
示例检索器 — 基于关键词匹配从示例库中找到最相关的 few-shot 示例

不依赖任何 ML 库，纯 Python 实现。通过关键词重叠度 + 类别分类来匹配。
"""

import json
import os
from typing import List, Dict


class ExampleRetriever:
    """基于关键词的示例检索器"""

    def __init__(self, examples_path: str = None):
        if examples_path is None:
            examples_path = os.path.join(os.path.dirname(__file__), "examples.json")
        with open(examples_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        self._build_category_keywords()

    def _build_category_keywords(self):
        """为每个类别建立关键词特征"""
        self.category_keywords = {
            "cleanup": ["清理", "垃圾", "清除", "捞", "收", "打扫", "捡", "油污",
                         "废弃物", "塑料袋", "漂浮物", "杂物", "塑料瓶", "死鱼"],
            "inspection": ["检查", "巡检", "巡查", "查看", "排查", "看", "检修",
                           "记录", "裂缝", "松动", "损坏", "漏油", "管道"],
            "patrol": ["巡逻", "巡视", "巡航", "警戒", "监控", "侦察"],
            "transport": ["运送", "运输", "送", "搬", "拿", "拖", "移", "带",
                          "搬上", "护送", "接", "投送"],
            "rescue": ["救", "搜救", "救援", "打捞", "落水", "失踪", "搁浅",
                       "被困", "伤员", "紧急", "洪灾"],
            "multi_step": ["然后", "顺便", "再", "先", "之后", "同时", "接着",
                           "完了", "统一", "兼顾"],
        }

    def retrieve(self, instruction: str, top_k: int = 3) -> List[Dict]:
        """
        检索与指令最相关的 top_k 个示例

        Args:
            instruction: 用户输入的自然语言指令
            top_k: 返回的示例数量

        Returns:
            最相关的示例列表，每个元素包含 instruction, plan, score
        """
        scored = []
        for ex in self.examples:
            score = self._score(instruction, ex)
            scored.append((score, ex))

        # 按分数降序排列
        scored.sort(key=lambda x: x[0], reverse=True)

        # 返回 top_k，确保至少覆盖不同类别
        results = []
        seen_categories = set()
        for score, ex in scored:
            if len(results) >= top_k:
                break
            # 优先选择高分示例，但也保证类别多样性
            cat = ex.get("category", "")
            if len(results) < top_k and (score > 1 or cat not in seen_categories or len(results) < 2):
                results.append({"instruction": ex["instruction"],
                                "plan": ex["plan"],
                                "score": score,
                                "category": cat})
                seen_categories.add(cat)

        # 如果不够 top_k，补充最高分示例
        if len(results) < top_k:
            for score, ex in scored:
                if len(results) >= top_k:
                    break
                if {"instruction": ex["instruction"], "plan": ex["plan"],
                    "score": score, "category": ex.get("category", "")} not in results:
                    results.append({"instruction": ex["instruction"],
                                    "plan": ex["plan"],
                                    "score": score,
                                    "category": ex.get("category", "")})

        return results

    def _score(self, instruction: str, example: Dict) -> float:
        """
        计算指令与示例的匹配分数

        分数组成：
        - 关键词重叠数（每个重叠 +2）
        - 类别匹配（猜测指令类别与示例类别相同 +3）
        """
        score = 0.0
        instr_lower = instruction.lower()

        # 关键词重叠
        for kw in example.get("keywords", []):
            if kw.lower() in instr_lower:
                score += 2.0

        # 类别匹配
        guessed_cat = self._guess_category(instruction)
        if guessed_cat == example.get("category", ""):
            score += 3.0

        # 如果猜出是多步任务，multi_step 示例额外加分
        if guessed_cat == "multi_step" and example.get("category") == "multi_step":
            score += 1.0

        return score

    def _guess_category(self, instruction: str) -> str:
        """根据关键词猜指令属于哪个类别"""
        best_cat = "patrol"
        best_score = 0
        for cat, keywords in self.category_keywords.items():
            score = sum(1 for kw in keywords if kw in instruction)
            if score > best_score:
                best_score = score
                best_cat = cat
        return best_cat

    def build_prompt_examples(self, instruction: str, top_k: int = 3) -> str:
        """
        构建用于注入 system prompt 的 few-shot 示例文本

        Args:
            instruction: 用户指令
            top_k: 包含的示例数

        Returns:
            格式化的示例文本块
        """
        examples = self.retrieve(instruction, top_k)
        lines = ["以下是一些任务规划的参考示例：\n"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"示例{i}：")
            lines.append(f"用户指令：\"{ex['instruction']}\"")
            lines.append(f"输出：")
            lines.append(json.dumps(ex["plan"], ensure_ascii=False, indent=2))
            lines.append("")
        return "\n".join(lines)


# 全局单例
_retriever: ExampleRetriever = None


def get_retriever() -> ExampleRetriever:
    """获取全局检索器单例"""
    global _retriever
    if _retriever is None:
        _retriever = ExampleRetriever()
    return _retriever
