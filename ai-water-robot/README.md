<p align="center">
 <h1 align="center"> 水域机器人 3D 智能决策平台</h1>
 <p align="center"><strong>Water Robot 3D Intelligent Decision Platform</strong></p>
</p>

<p align="center">
 <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
 <img src="https://img.shields.io/badge/GUI-PySide6-green.svg" alt="PySide6">
 <img src="https://img.shields.io/badge/3D-PyVista-orange.svg" alt="PyVista">
 <img src="https://img.shields.io/badge/ROS2-ready-brightgreen.svg" alt="ROS2">
 <img src="https://img.shields.io/badge/lines-8,690-purple.svg" alt="Lines">
 <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

---

## 项目简介

一个端到端的**水域机器人路径规划与仿真系统**。9种环境因子驱动的水流感知路径规划，支持3D交互式操作、ROS2远程控制和LLM智能决策。

> **核心亮点**：多物理场环境建模 × 多策略路径规划 × 实时3D动画 × ROS2桥接 × LLM决策

---

## 快速演示

```bash
# 一键数据展示 — 4场景策略对比 + 算法性能 + 任务模式多样性
python demo_showcase.py

# 3D桌面应用 — 交互式场景编辑 + 船模动画播放
python desktop3d_app.py

# 算法基准测试 — A* vs Dijkstra 32项全对比
python benchmark.py
```

<details>
<summary><b>点击展开演示输出示例</b></summary>

```
================== 四策略路径规划对比 ==================

> 沿海水域 30×30×8 | 障碍物: 251 | 途经点: 4
 策略 距离(m) 能耗(kJ) 时间(min)
 -------------------------------------------------
 均衡 4,034 3,435 22.4
 安全优先 4,057 3,544 22.5
 速度优先 4,034 3,435 22.4
 节能优先 4,034 3,435 22.4

> 海上风电场 28×28×6 | 障碍物: 190 | 途经点: 13
 策略 距离(m) 能耗(kJ) 时间(min)
 -------------------------------------------------
 均衡 4,068 2,774 22.6
 安全优先 4,788 3,434 26.6 ← 绕行距离+18%
 速度优先 4,068 2,774 22.6
 节能优先 4,068 2,774 22.6

基准测试: 32/32 全部通过, 成功率 100%
```
</details>

---

## 系统架构

```
 ┌──────────────────────────┐
 │ 用户交互层 (多通道输入) │
 │ 鼠标3D点选 │ 坐标输入 │ ROS2│
 └──────────┬───────────────┘
 │
 ┌────────────────────┼────────────────────┐
 ▼ ▼ ▼
 ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
 │ LLM 决策层 │ │ 路径规划引擎 │ │ ROS2 桥接层 │
 │ DeepSeek/GPT-4 │ │ A*3D / Dijkstra │ │ 发布/订阅/服务 │
 │ 场景分析→策略 │ │ 2-opt / TSP排序 │ │ TF坐标变换 │
 └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
 │ │ │
 └───────────────────┼────────────────────┘
 ▼
 ┌───────────────────────────────┐
 │ 3D 水体环境引擎 │
 │ 水深 │ 3层水流 │ 障碍物 │ 水压 │
 │ 风浪 │ 水温 │ 能见度 │ 潮汐 │ 漩涡│
 └───────────────┬───────────────┘
 ▼
 ┌───────────────────────────────┐
 │ 3D 可视化 + 动画仿真 │
 │ PyVista渲染 │ 船模动画 │ HUD │
 │ 尾迹特效 │ FOV视野 │ 途经点标记 │
 └───────────────────────────────┘
```

---

## 安装运行

### 环境要求
- Python 3.10+
- Windows / Linux / macOS

### 安装

```bash
git clone https://github.com/YUhuan83/AI-water-robot.git
cd AI-water-robot/ai-water-robot

# 核心依赖
pip install -r requirements.txt

# ROS2桥接 (可选)
pip install rclpy geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs tf2_ros
```

### 运行

```bash
# 3D桌面应用 (主要)
python desktop3d_app.py

# Web版 (Gradio)
python app.py

# 一键展示脚本
python demo_showcase.py
python demo_showcase.py --screenshots # 生成对比截图
python demo_showcase.py --scene windfarm # 只展示风电场景

# 基准测试
python benchmark.py
python benchmark.py --csv # 导出CSV数据
```

---

## 操作指南

### 鼠标交互 (3D视图)

| 操作 | 功能 | 修饰键 |
|------|------|--------|
| 左键拖拽 | 旋转视角 | — |
| Shift+左键 | 平移 | Shift |
| 右键点击 | 设置起点 / 添加途经点 | — |
| Shift+右键 | 设置终点 | Shift |
| Ctrl+右键 | 放置/移除障碍物 | Ctrl |
| 滚轮 | 缩放 | — |
| 空格键 | 播放/暂停动画 | — |

### 工具栏功能

`[沿海Demo] [河道Demo] [港口Demo] [风电场Demo]` — 加载预设场景
`[patrol v] [5 v] [随机任务]` — 任务模式 + 途经点数量 + 一键生成
`[重新规划] [播放] [暂停] [停止]` — 路径规划 + 动画控制
`[1x v] [跟随]` — 动画速度(0.5~4x) + 镜头跟随开关
`[撤销] [清除途经点] [清除障碍物] [清起点] [清终点]` — 编辑操作
`[保存任务] [加载任务]` — 任务配置存取
`[重置场景]` — 清空所有

### 面板功能

- **场景信息** — 实时显示网格/障碍物/水深/水压/天气/水温/能见度/潮汐
- **路径规划结果** — 距离/水流代价/深度范围/能耗/时间/电量
- **精确坐标输入** — X/Y/Z输入框 + 操作选择(设起点/加途经点/设终点/障碍物)
- **途经点管理** — 列表显示所有途经点，双击删除
- **智能决策** — 自然语言指令 → AI分析 → 自动应用策略
- **放置深度** — 选择目标深度层(表层~z=7)

---

## Demo 场景

<table>
<tr>
 <th>场景</th><th>规模</th><th>水深</th><th>障碍物</th><th>环境特征</th>
</tr>
<tr>
 <td><b>沿海水域</b></td>
 <td>30×30×8</td>
 <td>2~30m</td>
 <td>251格</td>
 <td>渐变水深 | 暗礁群 | 沉船遗址 | 海底管道 | 沿岸流 | 外海漩涡</td>
</tr>
<tr>
 <td><b>内河航道</b></td>
 <td>40×15×5</td>
 <td>0~14m</td>
 <td>73格</td>
 <td>S型蜿蜒河道 | 桥墩群 | 浅滩沙洲 | 变速流 | 弯道回流</td>
</tr>
<tr>
 <td><b>港口码头</b></td>
 <td>25×25×6</td>
 <td>3~57m</td>
 <td>74格</td>
 <td>码头泊位 | 防波堤 | 进出港航道 | 施工浮台 | 潮汐流</td>
</tr>
<tr>
 <td><b>海上风电场</b></td>
 <td>28×28×6</td>
 <td>19~25m</td>
 <td>190格</td>
 <td>12台风机 | 海上升压站 | 海底电缆 | 尾流效应</td>
</tr>
</table>

---

## 路径规划算法

### 代价函数 (9维)

```
总代价 = 基础距离
 + 水流代价 (顺流奖励/逆流惩罚, 含漩涡+风生流+潮汐放大)
 + 深度变化代价
 + 浅水奖励
 + 水压代价 (>200kPa ~ 10m深)
 + 风浪代价 (逆风+大浪, 仅表层)
 + 水温代价 (<5°C极冷|>30°C极热)
 + 能见度代价 (<5m低能见度)
 + 障碍物接近惩罚 (距离倒数加权)
```

### 算法对比

| 算法 | 方向数 | 启发式 | 最优性 | 速度 | 适用场景 |
|------|--------|--------|--------|------|----------|
| **A*3D** | 26 | 欧几里得 | 保证最优 | 快 | 实时规划 |
| **Dijkstra3D** | 26 | 无 | 保证最优 | 慢 | 短距离精确规划 |
| **2-opt** | — | — | 局部最优 | 中 | 路径后处理 |
| **TSP排序** | — | — | ≤5全排列 | 快 | 途经点顺序优化 |

> 基准测试：32/32项通过(100%)，A*与Dijkstra路径质量完全一致，A*平均快1%

### 四种策略

| 策略 | 安全距离 | 水流权重 | 水压权重 | 天气权重 | 适用场景 |
|------|----------|----------|----------|----------|----------|
| **均衡** | 2格 | 2.0 | 0.3 | 0.5 | 通用巡检 |
| **安全优先** | 3格 | 4.0 | 0.5 | 1.0 | 危险水域/载人 |
| **速度优先** | 1格 | 1.0 | 0.1 | 0.3 | 紧急任务 |
| **节能优先** | 2格 | 3.0 | 1.0 | 0.8 | 长航程/电池有限 |

---

## ROS2 接口

ROS2桥接作为可选模块，启用后自动在独立线程运行。

### 发布 Topic (10Hz)

| Topic | 消息类型 | 内容 |
|-------|----------|------|
| `/water_robot/pose` | `PoseStamped` | 机器人位姿 (x, y, z, 四元数朝向) |
| `/water_robot/odom` | `Odometry` | 里程计 (线速度 m/s) |
| `/water_robot/battery` | `BatteryState` | 剩余电量百分比 |
| `/water_robot/sensor` | `FluidPressure` | 水压(kPa) + 水温(variance) |
| `/water_robot/path` | `Path` | 当前规划路径 |
| `/water_robot/status` | `String` | 系统状态文字 |

### 订阅 Topic + Service

```bash
# 远程设置终点
ros2 topic pub /water_robot/cmd_goal geometry_msgs/msg/PoseStamped \
 "{header: {frame_id: 'map'}, pose: {position: {x: 20.0, y: 15.0, z: 0.0}}}"

# 添加途经点
ros2 topic pub /water_robot/cmd_waypoint geometry_msgs/msg/PoseStamped \
 "{header: {frame_id: 'map'}, pose: {position: {x: 8.0, y: 6.0, z: 0.0}}}"

# 远程切换策略
ros2 topic pub /water_robot/cmd_strategy std_msgs/msg/String "{data: 'safe'}"

# 远程启动/停止动画
ros2 service call /water_robot/start_mission std_srvs/srv/SetBool "{data: true}"

# 远程重置场景
ros2 service call /water_robot/clear_all std_srvs/srv/Trigger
```

### TF 树

```
map
└── water_robot/base_link (10Hz广播)
```

---

## 任务模式库

| 模式 | 函数 | 途经点生成逻辑 |
|------|------|---------------|
| **巡逻** `patrol` | `generate_patrol_waypoints()` | 矩形区域 lawnmower 来回扫描线 |
| **螺旋** `spiral` | `generate_spiral_waypoints()` | 从中心向外逐圈扩展，每圈偏移角度 |
| **之字** `zigzag` | `generate_zigzag_waypoints()` | 两点间S形正弦偏移 |
| **散点** `scattered` | `generate_scattered_waypoints()` | 可航行水域随机采样，保证最小间距 |
| **多簇** `cluster` | `generate_cluster_waypoints()` | 围绕多个巡检中心生成点群 |
| **环绕** `perimeter` | `generate_perimeter_waypoints()` | 沿可航行区域边界一周 |

---

## 基准测试数据

```
====================================================================================================
 算法 策略 距离(m) 能耗(kJ) 时间(min) 水流代价 路径点
-----------------------------------------------------------------------------------------------------
 海上风电场 (190个障碍物, 13个途经点)
-----------------------------------------------------------------------------------------------------
 A*3D 均衡 4,068 2,774 22.6 634 32
 A*3D 安全 4,788 3,434 26.6 631 39 ← 绕行+18%
 A*3D 快速 4,068 2,774 22.6 634 32
 A*3D 节能 4,068 2,774 22.6 634 32
-----------------------------------------------------------------------------------------------------
 汇总
-----------------------------------------------------------------------------------------------------
 均衡 平均距离=2,619m | 平均能耗=1,984kJ
 安全 平均距离=2,816m | 平均能耗=2,212kJ ← 安全策略: 距离+7.5%, 能耗+11.5%
 快速 平均距离=2,620m | 平均能耗=1,971kJ ← 快速策略: 最省能耗
 节能 平均距离=2,621m | 平均能耗=1,982kJ

 总测试: 32 | 成功: 32 | 成功率: 100.0%
```

---

## 项目结构

```
ai-water-robot/ 8,690行 Python
├── desktop3d_app.py 3D桌面主程序 (2,092行)
├── environment/
│ └── water_3d.py 3D水体环境 (925行)
├── planning/
│ └── astar3d.py 路径规划 (833行)
├── ros2_bridge.py ROS2桥接 (355行)
├── task_planner/
│ ├── mission_patterns.py 任务模式库 (283行)
│ └── llm_planner.py LLM任务规划器 (416行)
├── data/
│ └── water_adapter.py 多格式数据加载 (368行)
├── visualization/
│ └── render3d.py matplotlib 3D渲染 (154行)
├── demo_showcase.py 一键展示脚本 (230行)
├── benchmark.py 算法基准测试 (200行)
├── app.py Gradio Web入口 (607行)
├── desktop_app.py Tkinter桌面版 (589行)
└── output/ 输出文件目录
```

---

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 3D渲染 | **PyVista + VTK** | 高性能三维可视化 |
| GUI | **PySide6 (Qt)** | 跨平台桌面框架 |
| 数值计算 | **NumPy** | 矩阵运算/插值 |
| 路径规划 | **A* / Dijkstra / 2-opt / TSP** | 自研多策略引擎 |
| 动画 | **QTimer + 线性插值** | 60fps平滑动画 |
| LLM | **OpenAI兼容API** | DeepSeek/GPT-4/GPT-4o |
| 机器人 | **ROS2 (rclpy)** | 可选，标准化机器人通信 |
| 可视化 | **matplotlib** | 截图/图表/报告 |

---

## 设计特点

1. **9维环境因子** — 水深、3层水流、水压、风浪、水温、能见度、潮汐、漩涡、障碍物
2. **水流感知导航** — 顺流奖励、逆流惩罚，动态规划节能航线
3. **多策略对比** — 同屏4色路径，量化距离/能耗/时间差异
4. **TSP智能排序** — ≤5途经点全排列枚举，>5最近邻+轮转启发式
5. **60fps船模动画** — 位置插值、朝向跟随、波浪起伏、尾迹衰减
6. **三通道输入** — 鼠标3D点选 + 精确坐标输入 + ROS2远程指令
7. **ROS2标准化** — 发布位姿/里程计/电池/传感器，订阅指令
8. **LLM决策** — 自然语言→场景分析→自动切换策略→重新规划

---

*Built with PySide6 / PyVista / NumPy / ROS2 / DeepSeek | 2024-2026*
