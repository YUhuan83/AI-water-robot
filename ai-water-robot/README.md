# 水域智能感知与任务理解系统

**AI-Powered Water Robot Intelligence Demo**

一个端到端的 AI + 水上机器人仿真演示系统。用户输入自然语言指令，系统通过 **LLM 任务拆解 -> YOLO 目标检测 -> A* 路径规划 -> 仿真动画** 的完整管线，展示 AI 在水上机器人领域的综合应用。

## 演示场景

| 场景 | 用户指令 | 展示重点 |
|------|---------|---------|
| 垃圾清理 | "清理河道里所有漂浮垃圾，避开障碍物" | YOLO检测 -> A*避障 -> 收集动画 |
| 浮标巡检 | "从码头出发，依次检查所有浮标，然后返回" | 多点巡回路径规划 |
| 水域巡逻 | "巡逻整个水域，发现并标记所有异常物体" | LLM理解意图 + 检测+标记 |

## 系统架构

```
用户自然语言指令
     |
     v
+---------------+    +---------------+    +---------------+    +---------------+
|  LLM 任务     |--->|  YOLOv8       |--->|  A* 路径      |--->| 仿真动画      |
|  规划器       |    |  目标检测      |    |  规划         |    |  GIF 生成     |
| DeepSeek API  |    | ultralytics   |    |  8方向寻路     |    | matplotlib    |
+---------------+    +---------------+    +---------------+    +---------------+
     |                     |                    |                    |
     v                     v                    v                    v
  任务 JSON            检测标注图           路径预览图           执行 GIF
```

## 快速开始

### 环境要求

- Python 3.10+
- （可选）DeepSeek API Key（不设置则使用规则模式）

### 安装

```bash
# 进入项目目录
cd ai-water-robot

# 安装核心依赖
pip install -r requirements.txt

# （可选）如需 YOLO 目标检测功能
pip install ultralytics
```

### 设置 API Key（可选）

```bash
# Windows PowerShell
$env:DEEPSEEK_API_KEY = "sk-your-api-key-here"

# Windows CMD
set DEEPSEEK_API_KEY=sk-your-api-key-here
```

不设置 API Key 也可以运行——系统会自动使用规则模式进行任务规划。

### 启动

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860` 即可使用。

如需生成临时公网链接供远程访问（72 小时有效）：

```python
# 修改 app.py 最后一行
demo.launch(share=True, ...)
```

## 项目结构

```
ai-water-robot/
├── app.py                       # Gradio 主入口（四 Tab UI）
├── config.py                    # API key、模型路径配置
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
│
├── task_planner/                # LLM 任务规划
│   ├── __init__.py
│   └── llm_planner.py           # DeepSeek API + 规则模式
│
├── environment/                 # 水面网格环境
│   ├── __init__.py
│   └── water_grid.py            # 网格类 + 预设场景 + PIL 渲染
│
├── perception/                  # 目标检测
│   ├── __init__.py
│   └── detector.py              # YOLOv8 封装
│
├── planning/                    # 路径规划
│   ├── __init__.py
│   └── astar.py                 # A* 算法（8方向 + 多点巡回）
│
├── visualization/               # 可视化渲染
│   ├── __init__.py
│   └── renderer.py              # matplotlib 动画 -> GIF
│
└── output/                      # 输出文件（自动生成）
    ├── path_preview.png
    └── simulation.gif
```

## 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| Web UI | **Gradio 6** | Blocks API + 四 Tab 布局 |
| LLM 规划 | **DeepSeek API** (`deepseek-chat`) | 自然语言 -> 结构化任务 JSON |
| 目标检测 | **YOLOv8n** (ultralytics) | COCO 预训练权重 |
| 路径规划 | **A* 算法** | 8 方向移动，欧几里得启发式 |
| 可视化 | **matplotlib** FuncAnimation | 帧动画 -> GIF |
| 图像处理 | **Pillow** | 场景渲染 |

## 网格编码说明

| 编码 | 类型 | 颜色 | 说明 |
|------|------|------|------|
| 0 | 水域 | 深蓝 | 可通行 |
| 1 | 障碍物 | 棕色 | 不可通行 |
| 2 | 浮标 | 绿色 | 巡检目标 |
| 3 | 垃圾 | 黄色 | 待清理 |
| 4 | 机器人 | 亮蓝 | 当前位置 |

## 开发调试

### 模块独立测试

```bash
# 测试水面网格
python -c "from environment.water_grid import demo_scene_trash_cleanup; print(demo_scene_trash_cleanup(10))"

# 测试 A* 路径规划
python -c "from planning.astar import astar; import numpy as np; g=np.zeros((10,10)); g[5,3:8]=1; print(astar(g,(0,0),(9,9)))"

# 测试任务规划（离线模式）
python -c "from task_planner.llm_planner import rule_based_plan; print(rule_based_plan('清理垃圾'))"

# 测试可视化
python -c "
from environment.water_grid import demo_scene_trash_cleanup
from planning.astar import plan_multi_point_route
from visualization.renderer import SimulationRenderer
s=demo_scene_trash_cleanup(15)
p=plan_multi_point_route(s.get_obstacle_grid(),s.robot_pos,s.get_object_positions(3))
r=SimulationRenderer(s)
r.render_gif(p,'output/test.gif',collect_targets=s.get_object_positions(3))
"
```

## 项目定位

这是一个 AI + 水上机器人的练手项目，旨在串联以下技术点：

1. **AI + 机器人综合能力** -- 不是单独跑一个模型，而是端到端的系统集成
2. **LLM 在机器人领域的应用** -- 自然语言任务理解与拆解
3. **CV + 规划 + 控制 全链路** -- 计算机视觉 + 路径规划 + 仿真执行
4. **工程能力** -- 模块化设计、清晰的代码结构、可演示的完整系统

---

*Built with DeepSeek LLM / YOLOv8 / A* / Gradio / Matplotlib*
