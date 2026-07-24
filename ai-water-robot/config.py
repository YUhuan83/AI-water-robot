"""
项目配置文件
存放 API 密钥、模型路径等全局配置项
"""

import os

# ── DeepSeek API 配置 ──
# 从环境变量读取 API Key，如未设置则使用默认值
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "sk-your-api-key-here"  # 请替换为你的真实 key
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # deepseek-chat (V3) 或 deepseek-reasoner (R1)

# ── YOLO 模型配置 ──
YOLO_MODEL_NAME = "yolov8n.pt"  # 首次运行时会自动下载

# ── 水面网格默认参数 ──
DEFAULT_GRID_SIZE = 20  # 默认网格大小 (N×N)

# ── 输出路径 ──
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
GIF_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "simulation.gif")
