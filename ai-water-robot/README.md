# 水域机器人 3D 智能决策平台

**Water Robot 3D Intelligent Decision Platform** — 端到端的水域机器人路径规划与仿真系统

8,260 行 Python | PySide6 + PyVista 3D | A* / Dijkstra / 2-opt | ROS2 桥接 | LLM 集成

---

## 功能总览

| 模块 | 功能 |
|------|------|
| 🗺️ **3D场景** | 4个真实水域场景（沿海/河道/港口/风电场），水深/水流/障碍物/水压/风浪/漩涡/水温/能见度/潮汐 |
| 🧭 **路径规划** | A*3D (26方向) + Dijkstra3D + 2-opt优化 + 平滑 + TSP途经点排序 |
| 🎯 **4种策略** | balanced(均衡) / safe(安全优先) / fast(速度优先) / energy(节能优先) |
| 🚤 **3D动画** | 船形机器人 + 60fps平滑插值 + 波浪起伏 + 尾迹 + FOV视野锥 |
| 🖱️ **交互方式** | 3D鼠标点击 / 精确坐标输入 / ROS2远程指令 |
| 🤖 **LLM决策** | DeepSeek/OpenAI API 接口，AI自动分析场景建议策略 |
| 📊 **分析导出** | 策略对比可视化 / 路径统计 / JSON报告 / 截图 |
| 🔗 **ROS2桥接** | 10Hz发布位姿/里程计/电池/传感器，订阅指令，TF广播 |
| 🎲 **任务多样性** | 6种任务模式(巡逻/螺旋/之字/散点/多簇/环绕) × 4种复杂度 |

---

## 快速开始

### 安装

```bash
cd ai-water-robot
pip install -r requirements.txt

# ROS2支持 (可选)
pip install rclpy geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs tf2_ros
```

### 启动 3D 桌面应用

```bash
python desktop3d_app.py
```

### 一键展示脚本

```bash
python demo_showcase.py                    # 控制台输出关键指标
python demo_showcase.py --screenshots      # 同时生成对比截图
python demo_showcase.py --scene windfarm   # 展示指定场景
```

### 算法基准测试

```bash
python benchmark.py           # A* vs Dijkstra 全场景对比
python benchmark.py --csv     # 同时导出CSV
```

---

## 界面操作

| 操作 | 功能 |
|------|------|
| 左键拖拽 | 旋转视角 |
| Shift+左键 | 平移 |
| 滚轮 | 缩放 |
| 右键点击 | 设置起点 / 添加途经点 |
| Shift+右键 | 设置终点 |
| Ctrl+右键 | 放置/移除障碍物 |
| 空格键 | 播放/暂停动画 |

工具栏提供：Demo加载 / 策略选择 / 播放控制 / 速度调节 / 镜头跟随 / 随机任务 / 撤销 / 导出

---

## 项目结构

```
ai-water-robot/
├── desktop3d_app.py          # 🔴 3D桌面主程序 (2,092行)
├── environment/
│   └── water_3d.py           # 3D水体环境 (925行) — 水深/水流/水压/天气/水温/能见度/潮汐/漩涡
├── planning/
│   └── astar3d.py            # 路径规划 (833行) — A*3D/Dijkstra/2-opt/TSP/平滑/能耗
├── ros2_bridge.py            # ROS2桥接 (355行) — 发布/订阅/服务/TF
├── task_planner/
│   ├── mission_patterns.py   # 任务模式库 (283行) — 巡逻/螺旋/之字/散点/多簇/环绕
│   └── llm_planner.py        # LLM任务规划器 (416行)
├── data/
│   └── water_adapter.py      # 多格式数据加载 (368行)
├── visualization/
│   └── render3d.py           # matplotlib 3D渲染 (154行)
├── demo_showcase.py          # 一键展示脚本 — 4场景策略对比
├── benchmark.py              # 算法基准测试 — A* vs Dijkstra
├── app.py                    # Gradio Web入口 (607行)
├── desktop_app.py            # Tkinter桌面版 (589行)
└── output/                   # 输出文件目录
```

---

## DEMO 场景

| 场景 | 规模 | 水深 | 特征 |
|------|------|------|------|
| **沿海水域** | 30×30×8 | 2~30m | 渐变水深/暗礁群/沉船/管道/沿岸流 |
| **内河航道** | 40×15×5 | 2~12m | S型蜿蜒/桥墩群/浅滩/变速流/回流 |
| **港口码头** | 25×25×6 | 3~18m | 泊位/防波堤/航道/浮标/潮汐流 |
| **海上风电场** | 28×28×6 | 20~25m | 12台风机/升压站/电缆/尾流效应 |

---

## 路径规划算法

- **代价函数**: 距离 + 水流(顺流奖励/逆流惩罚) + 深度变化 + 水压 + 风浪 + 水温 + 能见度 + 障碍物接近
- **A*3D**: 26方向移动 × 欧几里得启发式，适用于实时规划
- **Dijkstra3D**: 无启发式保证最优解，适用于短距离精确规划
- **2-opt**: 路径局部优化，消除冗余交叉
- **平滑**: 保护途经点，去除冗余中间格点
- **TSP**: 途经点最优排序 (≤5个全排列枚举，>5个最近邻+轮转)
- **密度保证**: 每4格至少1个路径点，线性插值补密

---

## ROS2 接口

```bash
# 查看机器人位姿
ros2 topic echo /water_robot/pose

# 远程设置终点
ros2 topic pub /water_robot/cmd_goal geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 20.0, y: 15.0, z: 0.0}}}"

# 切换策略
ros2 topic pub /water_robot/cmd_strategy std_msgs/msg/String "{data: 'safe'}"

# 启动动画
ros2 service call /water_robot/start_mission std_srvs/srv/SetBool "{data: true}"
```

---

## 技术栈

| 层 | 技术 |
|----|------|
| 3D渲染 | PyVista + VTK |
| GUI | PySide6 (Qt) |
| 数值计算 | NumPy |
| 路径规划 | A* / Dijkstra / 2-opt / TSP (自研) |
| 动画 | QTimer 60fps + 线性插值 |
| LLM | OpenAI 兼容 API (DeepSeek/GPT) |
| ROS2 | rclpy (可选) |
| 可视化 | matplotlib (截图/图表) |

---

## 项目定位

这是一个 **AI + 机器人** 综合演示项目，串联以下技术点：

1. **3D环境建模** — 多物理场（水流/水压/天气/水温/能见度/潮汐）统一建模
2. **多策略规划** — 4种策略 × 2种算法 × 2种启发式，水流感知代价函数
3. **人机交互** — 鼠标3D点选 + 精确坐标 + ROS2远程指令三通道
4. **实时仿真** — 60fps动画 + HUD遥测 + 镜头跟随 + 途经点特效
5. **ROS2集成** — 标准化机器人通信接口，可接入真实机器人
6. **LLM决策** — 自然语言→场景分析→策略建议→自动应用

---

*Built with PySide6 / PyVista / NumPy / ROS2 / DeepSeek*
