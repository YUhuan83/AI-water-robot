"""
水域机器人 3D 智能决策平台 — PySide6 + PyVista
"""

import os, sys, math, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QDockWidget, QToolBar,
    QStatusBar, QMenuBar, QMenu, QMessageBox, QFrame,
    QGroupBox, QGridLayout, QSizePolicy, QSplitter, QTextEdit,
    QLineEdit, QDialog, QFormLayout, QDialogButtonBox, QComboBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont, QColor, QPalette, QIcon, QKeySequence

from pyvistaqt import QtInteractor
import pyvista as pv

from environment.water_3d import Water3DGrid, demo_3d_coastal, demo_3d_river, demo_3d_harbor
from planning.astar3d import (
    plan_tsp_3d, compute_3d_path_cost, compute_energy_estimate,
    compare_strategies, compare_and_select_best, STRATEGY_WEIGHTS,
)
from data.water_adapter import load_water_data
from task_planner.llm_planner import TaskPlanner, rule_based_plan

# ── 配色 ──
C = {
    "bg":          "#f5f6f8",
    "panel":       "#ffffff",
    "border":      "#d8dce0",
    "text":        "#2c3e50",
    "text_sec":    "#7f8c8d",
    "accent":      "#2980b9",
    "accent_hover":"#3498db",
    "danger":      "#c0392b",
    "success":     "#27ae60",
    "warning":     "#e67e22",
    "path":        "#e8590c",
    "start":       "#2ecc71",
    "wp":          "#f39c12",
    "end":         "#e74c3c",
}

QSS = f"""
QMainWindow {{ background: {C['bg']}; }}
QMenuBar {{ background: {C['panel']}; border-bottom: 1px solid {C['border']}; padding: 2px 8px; font-size: 13px; color: #000; }}
QMenuBar::item:selected {{ background: #d0d8e0; border-radius: 4px; }}
QMenu {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 4px; }}
QMenu::item {{ padding: 6px 28px; border-radius: 4px; color: #000; font-size: 13px; }}
QMenu::item:selected {{ background: #d0d8e0; }}
QToolBar {{ background: {C['panel']}; border-bottom: 1px solid {C['border']}; spacing: 6px; padding: 4px 8px; }}
QPushButton {{
    background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px;
    padding: 8px 16px; font-size: 13px; color: {C['text']};
}}
QPushButton:hover {{ background: #e8f0f8; border-color: {C['accent']}; }}
QPushButton:pressed {{ background: #d0dce8; }}
QPushButton#btnPrimary {{
    background: {C['accent']}; color: white; border: none; font-weight: bold;
}}
QPushButton#btnPrimary:hover {{ background: {C['accent_hover']}; }}
QPushButton#btnDanger {{ background: {C['danger']}; color: white; border: none; }}
QPushButton#btnDanger:hover {{ background: #e74c3c; }}
QGroupBox {{ font-weight: bold; border: 2px solid #b0b8c0; border-radius: 8px; margin-top: 14px; padding-top: 20px; background: {C['panel']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 2px 8px; color: #000; font-size: 14px; font-weight: bold; }}
QTextEdit {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px; font-size: 12px; color: {C['text']}; }}
QLabel {{ color: #111; font-size: 13px; }}
QLineEdit {{ color: #111; background: white; border: 1px solid #aab5c0; border-radius: 6px; padding: 8px 10px; font-size: 13px; font-weight: bold; }}
QStatusBar {{ background: {C['panel']}; border-top: 1px solid {C['border']}; font-size: 12px; }}
QDockWidget {{ font-size: 13px; color: {C['text']}; }}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("水域机器人 3D 智能决策平台")
        self.resize(1280, 800)
        self.grid: Water3DGrid = None
        self.path3d = None
        self.path_actor = None
        self.ctrl_held = False
        self.shift_held = False
        self.wave_timer = None
        self.wave_time = 0.0

        # 动画系统
        self.anim_timer = None
        self.anim_frame = 0
        self.anim_playing = False
        self.anim_speed = 1.0
        self.camera_follow = False   # 镜头跟随开关
        self.anim_robot = None       # 船身主体 actor
        self.anim_robot_cabin = None # 船舱 actor
        self._boat_bow_actor = None  # 船头三角 actor
        self._boat_templates = None  # (hull, cabin, bow) 模板网格
        self.fov_actor = None
        self.sensor_actor = None
        self.boat_heading = 0.0      # 船头朝向（弧度）
        self.boat_speed_kn = 0.0     # 当前航速（节）
        self.last_boat_pos = None    # 上帧位置（用于速度计算）
        self._frame_count = 0        # 帧计数器（用于降频更新）

        # HUD 文字
        self.hud_texts = []

        # 途经点到达特效
        self.pulse_rings = []        # [(actor, remaining_frames)]
        self.visited_waypoints = set()  # 已到达途经点索引
        self.arrival_log = []        # 到达日志 [(index, time_str)]

        # 电量/能耗
        self.battery_pct = 100.0     # 剩余电量百分比
        self.total_energy_kj = 0     # 路径总能耗
        self.energy_per_frame = 0    # 每帧能耗

        # 罗盘
        self.compass_actors = []

        # 当前策略
        self.current_strategy = "balanced"

        # 撤销
        self._undo_stack = []

        # LLM 配置
        self.llm_enabled = False
        self.llm_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
        self.llm_model = os.environ.get("LLM_MODEL", "deepseek-chat")
        self.planner: TaskPlanner = None

        self._build_ui()
        self._setup_3d()
        self._init_empty()

    # ═══════════════════ UI ═══════════════════

    def _build_ui(self):
        # 菜单栏
        mb = self.menuBar()
        file_menu = mb.addMenu("文件(&F)")
        file_menu.addAction("打开 JSON...", self._open_json, "Ctrl+O")
        file_menu.addAction("打开原始数据...", self._open_raw, "Ctrl+Shift+O")
        file_menu.addSeparator()
        file_menu.addAction("导出 JSON...", self._export_json)
        file_menu.addAction("导出截图...", self._export_screenshot)
        file_menu.addAction("导出报告...", self._export_report)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close, "Alt+F4")

        ai_menu = mb.addMenu("决策(&D)")
        ai_menu.addAction("大模型设置...", self._open_llm_settings)
        self.ai_toggle_action = QAction("启用智能决策", self)
        self.ai_toggle_action.setCheckable(True)
        self.ai_toggle_action.toggled.connect(self._toggle_llm)
        ai_menu.addAction(self.ai_toggle_action)

        demo_menu = mb.addMenu("演示(&M)")
        demo_menu.addAction("沿海水域", self._load_coastal)
        demo_menu.addAction("内河航道", self._load_river)
        demo_menu.addAction("港口码头", self._load_harbor)

        view_menu = mb.addMenu("视图(&V)")
        view_menu.addAction("重置视角", self._reset_view, "R")
        view_menu.addAction("俯视图", lambda: self.plotter.view_xy())
        view_menu.addAction("前视图", lambda: self.plotter.view_xz())
        view_menu.addAction("侧视图", lambda: self.plotter.view_yz())

        # 工具栏
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addWidget(self._btn("打开 JSON", self._open_json))
        tb.addWidget(self._btn("打开原始数据", self._open_raw))
        tb.addSeparator()
        tb.addWidget(self._btn("沿海 Demo", self._load_coastal))
        tb.addWidget(self._btn("河道 Demo", self._load_river))
        tb.addWidget(self._btn("港口 Demo", self._load_harbor))
        tb.addSeparator()
        tb.addWidget(self._btn("重新规划", self._replan, primary=True))
        tb.addSeparator()
        tb.addWidget(self._btn("播放", self._anim_play, primary=True))
        tb.addWidget(self._btn("暂停", self._anim_pause))
        tb.addWidget(self._btn("停止", self._anim_stop))
        # 动画速度
        self.speed_label = QLabel(" 速度:")
        self.speed_label.setStyleSheet("color:#2c3e50; font-weight:bold; font-size:12px;")
        tb.addWidget(self.speed_label)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self.speed_combo.setFixedWidth(65)
        self.speed_combo.setStyleSheet("font-size:12px; padding:2px 4px;")
        tb.addWidget(self.speed_combo)
        # 镜头跟随
        from PySide6.QtWidgets import QCheckBox
        self.chk_camera_follow = QCheckBox("跟随")
        self.chk_camera_follow.setToolTip("镜头跟随机器人移动")
        self.chk_camera_follow.setStyleSheet("font-size:12px; font-weight:bold; color:#2c3e50;")
        self.chk_camera_follow.toggled.connect(self._toggle_camera_follow)
        tb.addWidget(self.chk_camera_follow)
        # 进度
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color:#2980b9; font-weight:bold; font-size:12px; padding-left:8px;")
        tb.addWidget(self.progress_label)
        tb.addSeparator()
        tb.addWidget(self._btn("撤销", self._undo))
        tb.addWidget(self._btn("清除途经点", self._clear_wp))
        tb.addWidget(self._btn("清除障碍物", self._clear_obs))
        tb.addSeparator()
        tb.addWidget(self._btn("保存任务", self._save_mission))
        tb.addWidget(self._btn("加载任务", self._load_mission))
        tb.addWidget(self._btn("清起点", self._clear_start))
        tb.addWidget(self._btn("清终点", self._clear_end))
        tb.addWidget(self._btn("重置场景", self._clear_all, danger=True))

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 右键设置起点, Shift+右键设终点, Ctrl+右键放障碍物")

        # 右侧面板
        dock = QDockWidget("控制面板", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setFixedWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.setWidget(self._build_panel())

    def _build_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # 信息
        g1 = QGroupBox("场景信息")
        l1 = QVBoxLayout(g1)
        self.lbl_info = QTextEdit()
        self.lbl_info.setReadOnly(True)
        self.lbl_info.setMaximumHeight(100)
        self.lbl_info.setPlainText("未加载数据")
        l1.addWidget(self.lbl_info)
        lay.addWidget(g1)

        # 结果
        g2 = QGroupBox("路径规划结果")
        l2 = QGridLayout(g2)
        l2.setSpacing(6)

        # 策略选择行 — LLM启用时隐藏，由AI自动决策
        self._strategy_label = QLabel("策略:")
        l2.addWidget(self._strategy_label, 0, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["balanced — 均衡", "safe — 安全优先", "fast — 速度优先", "energy — 节能优先"])
        self.strategy_combo.setCurrentIndex(0)
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        l2.addWidget(self.strategy_combo, 0, 1)
        self.btn_compare = QPushButton("策略对比")
        self.btn_compare.clicked.connect(self._compare_strategies)
        l2.addWidget(self.btn_compare, 0, 2)
        self._ai_strategy_label = QLabel("AI 自动决策策略")
        self._ai_strategy_label.setStyleSheet("font-size:13px; font-weight:bold; color:#2980b9;")
        self._ai_strategy_label.hide()
        l2.addWidget(self._ai_strategy_label, 0, 0, 1, 3)
        # 保存策略相关控件引用便于显示/隐藏
        self._manual_strategy_widgets = [self._strategy_label, self.strategy_combo, self.btn_compare]

        self.lbl_dist = QLabel("总距离: --"); self.lbl_dist.setStyleSheet("font-size:18px;font-weight:bold;color:#c0392b")
        self.lbl_flow = QLabel("水流代价: --"); self.lbl_flow.setStyleSheet("font-size:15px;font-weight:bold;color:#1a5276")
        self.lbl_depth = QLabel("深度代价: --"); self.lbl_depth.setStyleSheet("font-size:15px;font-weight:bold;color:#1a5c2a")
        self.lbl_energy = QLabel("能耗: --"); self.lbl_energy.setStyleSheet("font-size:15px;font-weight:bold;color:#d35400")
        self.lbl_time = QLabel("预估时间: --"); self.lbl_time.setStyleSheet("font-size:15px;font-weight:bold;color:#1a5c2a")
        self.lbl_battery = QLabel("电量: --"); self.lbl_battery.setStyleSheet("font-size:15px;font-weight:bold;color:#2980b9")
        l2.addWidget(self.lbl_dist, 1, 0, 1, 3)
        l2.addWidget(self.lbl_flow, 2, 0, 1, 3)
        l2.addWidget(self.lbl_depth, 3, 0, 1, 3)
        l2.addWidget(self.lbl_energy, 4, 0, 1, 3)
        l2.addWidget(self.lbl_time, 5, 0, 1, 3)
        l2.addWidget(self.lbl_battery, 6, 0, 1, 3)
        lay.addWidget(g2)

        # 智能决策
        g3 = QGroupBox("智能决策")
        l3 = QVBoxLayout(g3)
        l3.setSpacing(6)
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("输入自然语言指令，如：避开暗礁区优先到达目标")
        self.ai_input.setMinimumHeight(32)
        l3.addWidget(self.ai_input)
        h_ai = QHBoxLayout()
        self.btn_analyze = QPushButton("分析决策")
        self.btn_analyze.setObjectName("btnPrimary")
        self.btn_analyze.clicked.connect(self._ai_analyze)
        h_ai.addWidget(self.btn_analyze)
        self.btn_plan_task = QPushButton("任务规划")
        self.btn_plan_task.clicked.connect(self._ai_task_plan)
        h_ai.addWidget(self.btn_plan_task)
        l3.addLayout(h_ai)
        self.lbl_ai = QLabel("大模型未配置 — 请先在菜单 决策 > 大模型设置 中配置 API Key")
        self.lbl_ai.setStyleSheet("color:#2c3e50; font-size:12px; font-weight:bold; background:#f0f4f8; padding:6px; border-radius:4px;")
        self.lbl_ai.setWordWrap(True)
        l3.addWidget(self.lbl_ai)
        lay.addWidget(g3)

        # 深度选择
        g4 = QGroupBox("放置深度")
        l4 = QVBoxLayout(g4)
        h_depth = QHBoxLayout()
        h_depth.addWidget(QLabel("目标深度层:"))
        self.depth_combo = QComboBox()
        self.depth_combo.addItems([f"表层 (z=0)" if i == 0 else f"z={i}" for i in range(8)])
        self.depth_combo.setCurrentIndex(0)
        h_depth.addWidget(self.depth_combo)
        l4.addLayout(h_depth)
        self.lbl_depth_hint = QLabel("右键/Shift+右键/Ctrl+右键 均使用此深度")
        self.lbl_depth_hint.setStyleSheet("color:#5a6a7a; font-size:11px;")
        l4.addWidget(self.lbl_depth_hint)
        lay.addWidget(g4)

        # 坐标精确输入
        g_coord = QGroupBox("精确坐标输入")
        l_coord = QVBoxLayout(g_coord)
        l_coord.setSpacing(4)
        # 坐标输入行
        h_coord = QHBoxLayout()
        h_coord.addWidget(QLabel("X:"))
        self.input_x = QLineEdit("0")
        self.input_x.setFixedWidth(42); self.input_x.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_x.setToolTip("X 坐标 (东向)")
        h_coord.addWidget(self.input_x)
        h_coord.addWidget(QLabel("Y:"))
        self.input_y = QLineEdit("0")
        self.input_y.setFixedWidth(42); self.input_y.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_y.setToolTip("Y 坐标 (北向)")
        h_coord.addWidget(self.input_y)
        h_coord.addWidget(QLabel("Z:"))
        self.input_z = QLineEdit("0")
        self.input_z.setFixedWidth(42); self.input_z.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_z.setToolTip("Z 坐标 (深度层, 0=表层)")
        h_coord.addWidget(self.input_z)
        l_coord.addLayout(h_coord)
        # 操作选择 + 执行
        h_act = QHBoxLayout()
        self.coord_action = QComboBox()
        self.coord_action.addItems(["设起点", "加途经点", "设终点", "切换障碍物"])
        self.coord_action.setToolTip("选择坐标操作类型")
        h_act.addWidget(self.coord_action)
        self.btn_coord_exec = QPushButton("执行")
        self.btn_coord_exec.setObjectName("btnPrimary")
        self.btn_coord_exec.clicked.connect(self._exec_coord_input)
        h_act.addWidget(self.btn_coord_exec)
        l_coord.addLayout(h_act)
        self.lbl_coord_range = QLabel("范围: X:0~29 Y:0~29 Z:0~7")
        self.lbl_coord_range.setStyleSheet("color:#5a6a7a; font-size:10px;")
        l_coord.addWidget(self.lbl_coord_range)
        lay.addWidget(g_coord)

        # 途经点管理列表
        g_wp = QGroupBox("途经点管理")
        l_wp = QVBoxLayout(g_wp)
        l_wp.setSpacing(4)
        from PySide6.QtWidgets import QListWidget, QListWidgetItem
        self.wp_list = QListWidget()
        self.wp_list.setMaximumHeight(100)
        self.wp_list.setStyleSheet("font-size:11px; color:#2c3e50;")
        self.wp_list.setToolTip("双击途经点删除")
        self.wp_list.itemDoubleClicked.connect(self._delete_waypoint_by_item)
        l_wp.addWidget(self.wp_list)
        h_wp_btns = QHBoxLayout()
        h_wp_btns.addWidget(self._btn("清空途经点", self._clear_wp, danger=False))
        self.btn_del_last_wp = QPushButton("删除最后一个")
        self.btn_del_last_wp.clicked.connect(self._delete_last_waypoint)
        self.btn_del_last_wp.setStyleSheet("font-size:11px; padding:4px 8px;")
        h_wp_btns.addWidget(self.btn_del_last_wp)
        l_wp.addLayout(h_wp_btns)
        lay.addWidget(g_wp)

        # 操作提示
        g5 = QGroupBox("操作提示")
        l5 = QVBoxLayout(g5)
        tips = QLabel(
            "左键拖拽 = 旋转 | Shift+左键 = 平移\n"
            "右键点击 = 设起点 / 添加途经点\n"
            "Shift+右键 = 设置终点\n"
            "Ctrl+右键 = 放置/移除障碍物\n"
            "滚轮 = 缩放 | 中键 = 旋转"
        )
        tips.setStyleSheet("color:#5a6a7a; font-size:12px;")
        l5.addWidget(tips)
        lay.addWidget(g5)

        lay.addStretch()
        return w

    def _btn(self, text, callback, primary=False, danger=False):
        b = QPushButton(text)
        b.clicked.connect(callback)
        if primary:
            b.setObjectName("btnPrimary")
        elif danger:
            b.setObjectName("btnDanger")
        return b

    # ═══════════════════ 3D 场景 ═══════════════════

    def _setup_3d(self):
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter)
        self.plotter.set_background("#dce8f0")
        self.plotter.show_grid(color="#c0c8d0", xlabel="东 X", ylabel="北 Y", zlabel="深 Z")
        self.plotter.add_text("加载数据以开始", position="upper_left", font_size=12, color="#556666")

        # 键盘修饰键 — 用 Qt 事件（更可靠）
        self.plotter.iren.interactor.AddObserver("KeyPressEvent", self._on_key)
        self.plotter.iren.interactor.AddObserver("KeyReleaseEvent", self._on_key_up)
        # 使用 track_click_position 获取点击坐标，不干扰正常导航
        self.plotter.track_click_position(callback=self._on_click, side="right")

    def _on_key(self, obj, event):
        key = obj.GetKeySym()
        if key in ("Control_L", "Control_R"):
            self.ctrl_held = True
        elif key in ("Shift_L", "Shift_R"):
            self.shift_held = True
        elif key == "space":
            # 空格切换播放/暂停
            if self.anim_playing:
                self._anim_pause()
            elif self.path3d:
                self._anim_play()

    def _on_key_up(self, obj, event):
        key = obj.GetKeySym()
        if key in ("Control_L", "Control_R"):
            self.ctrl_held = False
        elif key in ("Shift_L", "Shift_R"):
            self.shift_held = False

    def _on_click(self, position):
        """右键交互: 首次=设起点, 后续=途经点, Shift+右键=终点, Ctrl+右键=障碍物"""
        if self.grid is None or position is None:
            return
        try:
            x, y = int(round(float(position[0]))), int(round(float(position[1])))
        except Exception:
            return
        if not (0 <= x < self.grid.nx and 0 <= y < self.grid.ny):
            return
        z = self.depth_combo.currentIndex()
        z = max(0, min(z, self.grid.nz - 1))

        # 同步到坐标输入面板
        self.input_x.setText(str(x))
        self.input_y.setText(str(y))
        self.input_z.setText(str(z))

        self._push_undo()
        if self.ctrl_held:
            # Ctrl+右键: 切换障碍物
            self.grid.obstacles[z, y, x] = not self.grid.obstacles[z, y, x]
            action = f"障碍物 {'放置' if self.grid.obstacles[z, y, x] else '移除'} ({x}, {y})"
        elif self.shift_held:
            # Shift+右键: 设置终点
            self.grid.mission_end = (x, y, z)
            action = f"终点已设置 ({x}, {y})"
        elif self.grid.mission_start is None:
            # 首次右键（无起点时）: 设置起点
            self.grid.mission_start = (x, y, z)
            action = f"起点已设置 ({x}, {y})"
        else:
            # 普通右键: 添加途经点
            self.grid.mission_waypoints.append((x, y, z))
            action = f"途经点已添加 ({x}, {y}) — 共 {len(self.grid.mission_waypoints)} 个"
        self.plotter.clear()
        self._draw()
        self._refresh_info()
        self._refresh_wp_list()
        self.status_bar.showMessage(action)

    # ═══════════════════ 数据加载 ═══════════════════

    def load(self, grid):
        self.grid = grid
        self.path3d = None
        self.path_actor = None
        self.wave_time = 0.0
        self.plotter.clear()
        self._draw()
        self._start_waves()
        self.plotter.reset_camera()
        self._refresh_info()
        self._update_coord_range()
        self._refresh_wp_list()

    def _open_json(self):
        p, _ = QFileDialog.getOpenFileName(self, "打开 JSON 文件", "", "JSON (*.json);;所有文件 (*)")
        if p:
            try:
                self.load(Water3DGrid.from_json(p))
                self.status_bar.showMessage(f"已加载: {os.path.basename(p)}")
            except Exception as e:
                QMessageBox.critical(self, "加载失败", str(e))

    def _open_raw(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "打开原始数据", "",
            "所有支持格式 (*.csv *.xyz *.txt *.json);;CSV (*.csv);;XYZ (*.xyz);;文本 (*.txt);;所有文件 (*)",
        )
        if not p:
            return
        try:
            cur_path = None
            d = os.path.dirname(p)
            for c in ["currents.txt", "currents.csv", "flow.txt"]:
                cp = os.path.join(d, c)
                if os.path.exists(cp):
                    cur_path = cp; break
            self.load(load_water_data(p, currents_path=cur_path, resolution=100, nz=10))
            self.status_bar.showMessage(f"已加载: {os.path.basename(p)}")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"{e}")

    def _export_json(self):
        if not self.grid:
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出 JSON", "water_data.json", "JSON (*.json)")
        if p:
            self.grid.to_json(p)
            self.status_bar.showMessage(f"已导出: {p}")

    def _init_empty(self):
        """初始空场景"""
        from environment.water_3d import Water3DGrid
        g = Water3DGrid(30, 30, 8, resolution=50)
        g.set_uniform_bathymetry(15.0)
        g.dock_pos = None
        g.mission_start = None
        g.mission_waypoints = []
        g.mission_end = None
        self.load(g)

    def _load_coastal(self):
        self.load(demo_3d_coastal())
    def _load_river(self):
        self.load(demo_3d_river())
    def _load_harbor(self):
        self.load(demo_3d_harbor())

    # ═══════════════════ 渲染 ═══════════════════

    def _draw(self):
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz

        # 海底 — 按深度着色（浅水亮→深水暗）
        dv = np.full((ny, nx), np.nan, dtype=np.float32)
        depth_norm = np.zeros((ny, nx), dtype=np.float32)
        for y in range(ny):
            for x in range(nx):
                if g.depth[y, x] > 0:
                    dv[y, x] = -(g.depth[y, x] / max(1, g.depth.max())) * nz * 0.8
                    depth_norm[y, x] = g.depth[y, x] / max(1, g.depth.max())
        gx, gy = np.meshgrid(range(nx), range(ny))
        seabed = pv.StructuredGrid(gx, gy, dv)
        # 将归一化深度值作为scalars用于着色
        seabed["depth_norm"] = depth_norm.ravel(order="F")
        self.plotter.add_mesh(
            seabed, scalars="depth_norm", cmap="terrain",
            opacity=0.55, name="seabed", show_edges=False, clim=[0, 1],
        )

        # 海底等深线
        try:
            contours = seabed.contour()
            if contours.n_points > 0:
                self.plotter.add_mesh(
                    contours, color="#333333", opacity=0.35, line_width=1.2, name="depth_contours",
                )
        except Exception:
            pass

        # 水面半透明参考网格
        water_grid = pv.Plane(center=(nx/2, ny/2, 0), direction=(0, 0, 1),
                               i_size=nx, j_size=ny, i_resolution=nx, j_resolution=ny)
        self.plotter.add_mesh(
            water_grid, color="#88ccff", opacity=0.12, name="water_surface",
            show_edges=True, edge_color="#aaccdd", style="wireframe",
        )

        # 障碍物
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if g.obstacles[z, y, x]:
                        self.plotter.add_mesh(
                            pv.Cube(center=(x, y, -z - 0.5), x_length=0.85, y_length=0.85, z_length=0.85),
                            color=C["danger"], opacity=0.88, name=f"o.{x}.{y}.{z}",
                        )

        # 水流（使用合成水流 = 基础流 + 漩涡 + 风生流）— 降低密度
        step = max(2, min(nx, ny) // 5)
        for y in range(0, ny, step):
            for x in range(0, nx, step):
                if g.depth[y, x] <= 0:
                    continue
                (dx, dy, _), sp = g.get_total_current_at(x, y, 0)
                if sp > 0.01:
                    self.plotter.add_mesh(
                        pv.Arrow(start=(x, y, 0.3), direction=(dx * sp * 3, dy * sp * 3, 0),
                                  tip_length=0.25, tip_radius=0.08, shaft_radius=0.03),
                        color=C["accent"], opacity=0.5,
                    )

        # 漩涡可视化 (螺旋环)
        for cx, cy, radius, strength in g.eddies:
            # 漩涡中心标记
            self.plotter.add_mesh(
                pv.Sphere(center=(cx, cy, 0.1), radius=0.3),
                color="#9933cc", opacity=0.6, name=f"eddy_c_{cx}_{cy}",
            )
            # 漩涡环 (用箭头标示旋转方向) — 减少箭头数
            n_arrows = 6
            for i in range(n_arrows):
                angle = 2 * math.pi * i / n_arrows
                ax, ay = cx + math.cos(angle) * radius * 0.7, cy + math.sin(angle) * radius * 0.7
                # 切向方向
                tx, ty = -math.sin(angle), math.cos(angle)
                if g.depth[min(int(ay), ny - 1), min(int(ax), nx - 1)] > 0:
                    self.plotter.add_mesh(
                        pv.Arrow(start=(ax, ay, 0.15), direction=(tx * 0.6, ty * 0.6, 0),
                                  tip_length=0.2, tip_radius=0.05, shaft_radius=0.02),
                        color="#aa66dd", opacity=0.4,
                    )

        # 天气指示 (风向标)
        wind_dir, wind_spd, wave_h = g.get_weather_at(nx // 2, ny // 2)
        if wind_spd > 0.1:
            wx, wy = nx - 3, ny - 2
            self.plotter.add_mesh(
                pv.Arrow(start=(wx, wy, 0.8), direction=(wind_dir[0] * 2, wind_dir[1] * 2, 0),
                          tip_length=0.5, tip_radius=0.12, shaft_radius=0.06),
                color="#ffc107", opacity=0.7, name="wind",
            )
            self.plotter.add_point_labels(
                [[wx, wy, 0.8]],
                [f"风向 {wind_spd:.0f}m/s 浪{wave_h:.1f}m"],
                font_size=10, text_color="#ffc107", point_size=1,
            )

        # 任务点
        if g.mission_start:
            s = g.mission_start
            bs = pv.Sphere(center=(s[0], s[1], -s[2]), radius=0.55)
            self.plotter.add_mesh(bs, color=C["start"], pbr=True, name="start")
            self.plotter.add_point_labels([bs.center], ["起点"], font_size=13, text_color=C["start"], point_size=1)
        for wp in g.mission_waypoints:
            bw = pv.Sphere(center=(wp[0], wp[1], -wp[2]), radius=0.35)
            self.plotter.add_mesh(bw, color=C["wp"], name=f"wp.{wp}")
        # 途经点顺序标签（仅在路径规划后显示）
        if self.path3d and g.mission_waypoints:
            ordered_wps = [wp for wp in g.mission_waypoints if wp in self.path3d]
            if ordered_wps:
                wp_labels = []
                for wp in ordered_wps:
                    idx = self.path3d.index(wp) if wp in self.path3d else -1
                    if idx >= 0:
                        order = ordered_wps.index(wp) + 1
                        wp_labels.append((wp, f"WP{order}"))
                # 按途经点在路径中出现顺序显示编号
                for (wp, label) in wp_labels:
                    self.plotter.add_point_labels(
                        [[wp[0], wp[1], -wp[2] + 0.5]],
                        [label],
                        font_size=12, text_color="#f39c12", point_size=1,
                        name=f"wp_label_{wp}",
                    )
        if g.mission_end and g.mission_end != g.mission_start:
            e = g.mission_end
            be = pv.Sphere(center=(e[0], e[1], -e[2]), radius=0.5)
            self.plotter.add_mesh(be, color=C["end"], pbr=True, name="end")
            self.plotter.add_point_labels([be.center], ["终点"], font_size=13, text_color=C["end"], point_size=1)

        # 路径 (支持起点→终点直连，无需途经点)
        if g.mission_start and (g.mission_waypoints or g.mission_end):
            self._plan()

        # 罗盘
        self._add_compass()

    def _plan(self):
        g = self.grid
        if not (g and g.mission_start):
            return
        # 至少需要途经点或终点之一才能规划
        if not g.mission_waypoints and not g.mission_end:
            return
        # 确保终点不同于起点
        if not g.mission_waypoints and g.mission_end == g.mission_start:
            return
        self.path3d = plan_tsp_3d(g, g.mission_start, g.mission_waypoints, g.mission_end,
                                  strategy=self.current_strategy)
        if self.path3d is None:
            self.lbl_dist.setText("总距离: 无可行路径")
            self.lbl_flow.setText("水流代价: --")
            self.lbl_depth.setText("深度代价: --")
            self.lbl_energy.setText("能耗: --")
            self.lbl_time.setText("预估时间: --")
            return
        if self.path_actor:
            self.plotter.remove_actor(self.path_actor)
        pts = np.array([[p[0], p[1], -p[2]] for p in self.path3d], dtype=np.float64)
        if len(pts) >= 2:
            tube = pv.Spline(pts, n_points=len(pts) * 3).tube(radius=0.12)
            self.path_actor = self.plotter.add_mesh(tube, color=C["path"], pbr=True, metallic=0.1, name="path")
        d, f, dc = compute_3d_path_cost(g, self.path3d)
        energy = compute_energy_estimate(g, self.path3d)

        # 详细路径统计
        stats = self._compute_path_stats(self.path3d, g)

        strategy_names = {"balanced": "均衡", "safe": "安全", "fast": "快速", "energy": "节能"}
        sn = strategy_names.get(self.current_strategy, self.current_strategy)
        self.lbl_dist.setText(f"总距离: {d:,.0f} m [{sn}]")
        self.lbl_flow.setText(f"水流代价: {f:,.0f} | 顺流: {stats['downstream_pct']:.0f}%")
        self.lbl_depth.setText(f"深度: {stats['min_depth']:.0f}~{stats['max_depth']:.0f}m | 起伏{stats['depth_range']:.0f}m")
        self.lbl_energy.setText(f"能耗: {energy['energy_consumption_kj']:,.0f} kJ | 急转: {stats['sharp_turns']}次")
        self.lbl_time.setText(f"预估时间: {energy['estimated_time_min']:.1f} min | 均速: {stats['avg_speed_ms']:.1f}m/s")
        self.lbl_battery.setText(f"电池续航: 约 {energy['estimated_time_min']:.0f} min @ 100%")

    def _start_waves(self):
        if self.wave_timer is not None:
            return
        self.wave_timer = QTimer(self)
        self.wave_timer.timeout.connect(self._animate_wave)
        self.wave_timer.start(200)  # 200ms间隔（原80ms），降低CPU占用

    def _animate_wave(self):
        if self.grid is None:
            return
        self.wave_time += 0.20
        # 降低波浪网格分辨率：r取1/3而非1/2
        r = max(2, min(self.grid.nx, self.grid.ny) // 3)
        xx, yy = np.meshgrid(np.linspace(0, self.grid.nx, r * 2), np.linspace(0, self.grid.ny, r * 2))
        t = self.wave_time
        zz = (0.12 * np.sin(xx * 0.6 + t * 2.5) * np.cos(yy * 0.5 + t * 1.8)
              + 0.08 * np.sin(xx * 0.3 - t * 1.3) * np.sin(yy * 0.7 + t * 2.1)
              + 0.05 * np.cos((xx + yy) * 0.4 + t * 3.0))
        self.plotter.add_mesh(pv.StructuredGrid(xx, yy, zz), color="#3399bb",
                               opacity=0.32, name="wave", show_edges=False, smooth_shading=True)

    # ═══════════════════ 坐标输入 ═══════════════════

    def _exec_coord_input(self):
        """执行精确坐标输入操作"""
        if self.grid is None:
            QMessageBox.warning(self, "无数据", "请先加载水况数据")
            return
        g = self.grid
        try:
            x = int(self.input_x.text().strip())
            y = int(self.input_y.text().strip())
            z = int(self.input_z.text().strip())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "X/Y/Z 必须为整数")
            return

        # 范围校验
        if not (0 <= x < g.nx and 0 <= y < g.ny and 0 <= z < g.nz):
            QMessageBox.warning(
                self, "坐标越界",
                f"坐标超出范围!\nX: 0~{g.nx-1}\nY: 0~{g.ny-1}\nZ: 0~{g.nz-1}"
            )
            return

        # 水路校验
        if g.depth[y, x] <= 0:
            QMessageBox.warning(self, "不可通行", f"({x}, {y}) 为陆地，无法放置")
            return

        action = self.coord_action.currentText()
        self._push_undo()

        if action == "设起点":
            g.mission_start = (x, y, z)
            msg = f"起点已设置 ({x}, {y}, {z})"
        elif action == "加途经点":
            if g.mission_start is None:
                QMessageBox.warning(self, "无起点", "请先设置起点再添加途经点")
                self._undo_stack.pop()
                return
            g.mission_waypoints.append((x, y, z))
            msg = f"途经点已添加 ({x}, {y}, {z}) — 共 {len(g.mission_waypoints)} 个"
        elif action == "设终点":
            if g.mission_start is None:
                QMessageBox.warning(self, "无起点", "请先设置起点再设置终点")
                self._undo_stack.pop()
                return
            g.mission_end = (x, y, z)
            msg = f"终点已设置 ({x}, {y}, {z})"
        elif action == "切换障碍物":
            g.obstacles[z, y, x] = not g.obstacles[z, y, x]
            msg = f"障碍物 {'放置' if g.obstacles[z, y, x] else '移除'} ({x}, {y}, {z})"

        self.plotter.clear()
        self._draw()
        self._refresh_info()
        self._refresh_wp_list()
        self.status_bar.showMessage(msg)

    def _update_coord_range(self):
        """更新坐标输入范围提示"""
        if self.grid:
            g = self.grid
            self.lbl_coord_range.setText(
                f"范围: X:0~{g.nx-1}  Y:0~{g.ny-1}  Z:0~{g.nz-1}"
            )

    # ═══════════════════ 交互（见 _on_click 方法）═══════════════════

    def _replan(self):
        if self.grid:
            self.path_actor = None  # 清空旧引用，避免 _plan 中移除已清理的 actor
            self.plotter.clear()
            self._draw()

    def _on_strategy_changed(self, text):
        """策略下拉框切换时自动重新规划"""
        strategy_map = {
            "balanced — 均衡": "balanced",
            "safe — 安全优先": "safe",
            "fast — 速度优先": "fast",
            "energy — 节能优先": "energy",
        }
        self.current_strategy = strategy_map.get(text, "balanced")
        if self.grid and self.grid.mission_start and (self.grid.mission_waypoints or self.grid.mission_end):
            self._replan()
            self.status_bar.showMessage(f"策略已切换: {text}")

    def _compare_strategies(self):
        """多策略对比 — 3D视图同时显示4条路径"""
        g = self.grid
        if not (g and g.mission_start and (g.mission_waypoints or g.mission_end)):
            QMessageBox.warning(self, "无法对比", "请先加载数据并设置起点和途经点/终点")
            return
        self.status_bar.showMessage("正在对比四种策略...")

        # 清除旧的对比路径
        for name in ["cmp_balanced", "cmp_safe", "cmp_fast", "cmp_energy"]:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass

        strategy_colors = {
            "balanced": ("#e8590c", "均衡"),
            "safe":      ("#27ae60", "安全"),
            "fast":      ("#e74c3c", "快速"),
            "energy":    ("#2980b9", "节能"),
        }
        results = {}
        best_dist = float("inf")
        best_energy = float("inf")

        for strategy, (color, cn_name) in strategy_colors.items():
            path = plan_tsp_3d(
                g, g.mission_start, g.mission_waypoints, g.mission_end,
                strategy=strategy,
            )
            if path is None:
                results[strategy] = None
                continue
            d, f, dc = compute_3d_path_cost(g, path)
            energy = compute_energy_estimate(g, path)
            results[strategy] = {
                "path": path, "distance": d, "flow_cost": f,
                "depth_cost": dc, "energy_kj": energy["energy_consumption_kj"],
                "time_min": energy["estimated_time_min"],
            }
            if d < best_dist:
                best_dist = d
            if energy["energy_consumption_kj"] < best_energy:
                best_energy = energy["energy_consumption_kj"]

            # 绘制路径
            pts = np.array([[p[0], p[1], -p[2]] for p in path], dtype=np.float64)
            if len(pts) >= 2:
                tube = pv.Spline(pts, n_points=max(len(pts), 2) * 2).tube(radius=0.08)
                self.plotter.add_mesh(
                    tube, color=color, opacity=0.65, pbr=True, metallic=0.05,
                    name=f"cmp_{strategy}",
                )

        # 图例（用text标注在角落）
        legend_lines = ["═══ 策略对比 ═══"]
        for strategy, (color, cn_name) in strategy_colors.items():
            r = results.get(strategy)
            if r is None:
                legend_lines.append(f"  {cn_name}: 无可行路径")
            else:
                dm = " ★最短" if r["distance"] == best_dist else ""
                em = " ★最省" if r["energy_kj"] == best_energy else ""
                legend_lines.append(
                    f"  {cn_name}: {r['distance']:,.0f}m{dm} | {r['energy_kj']:,.0f}kJ{em}"
                )
        legend_lines.append("提示: 切换下拉框策略查看单条路径")

        self.lbl_ai.setText("\n".join(legend_lines))
        self.status_bar.showMessage(
            f"策略对比完成 — 4条路径已叠加显示 | "
            f"最短: {best_dist:,.0f}m | 最省: {best_energy:,.0f}kJ"
        )

    def _clear_wp(self):
        if self.grid:
            self._push_undo()
            self.grid.mission_waypoints = []
            self.plotter.clear()
            self._draw()
            self._refresh_wp_list()

    def _delete_last_waypoint(self):
        """删除最后一个途经点"""
        if self.grid and self.grid.mission_waypoints:
            self._push_undo()
            removed = self.grid.mission_waypoints.pop()
            self.plotter.clear()
            self._draw()
            self._refresh_wp_list()
            self.status_bar.showMessage(f"已删除途经点: {removed}")

    def _delete_waypoint_by_item(self, item):
        """双击途经点列表项删除"""
        idx = self.wp_list.row(item)
        if self.grid and 0 <= idx < len(self.grid.mission_waypoints):
            self._push_undo()
            removed = self.grid.mission_waypoints.pop(idx)
            self.plotter.clear()
            self._draw()
            self._refresh_wp_list()
            self.status_bar.showMessage(f"已删除途经点[{idx}]: {removed}")

    def _refresh_wp_list(self):
        """刷新途经点列表显示"""
        self.wp_list.clear()
        if self.grid:
            for i, wp in enumerate(self.grid.mission_waypoints):
                item = QListWidgetItem(f"WP{i+1}: ({wp[0]}, {wp[1]}, z={wp[2]})")
                item.setToolTip("双击删除此途经点")
                self.wp_list.addItem(item)

    def _clear_obs(self):
        if self.grid:
            self._push_undo()
            self.grid.obstacles[:] = False
            self.plotter.clear()
            self._draw()

    def _clear_start(self):
        """单独清除起点"""
        if self.grid:
            self._push_undo()
            self.grid.mission_start = None
            self.path3d = None
            self.path_actor = None
            self.plotter.clear()
            self._draw()
            self._refresh_info()
            self.status_bar.showMessage("起点已清除")

    def _clear_end(self):
        """单独清除终点"""
        if self.grid:
            self._push_undo()
            self.grid.mission_end = None
            self.path3d = None
            self.path_actor = None
            self.plotter.clear()
            self._draw()
            self.status_bar.showMessage("终点已清除")

    def _clear_all(self):
        """重置整个场景：清除所有任务点 (起点/途经点/终点) 和障碍物"""
        if self.grid:
            self._push_undo()
            self.grid.mission_start = None
            self.grid.mission_waypoints = []
            self.grid.mission_end = None
            self.grid.obstacles[:] = False
            self.path3d = None
            self.path_actor = None
            self.plotter.clear()
            self._draw()
            self._refresh_info()
            # 停止动画
            self._anim_stop()
            self._refresh_wp_list()
            self.status_bar.showMessage("场景已重置 — 右键设置起点开始规划")

    def _reset_view(self):
        self.plotter.reset_camera()

    # ═══════════════════ 船形机器人模型 ═══════════════════

    def _create_boat_template(self):
        """预创建船模模板网格（原点，无旋转），只调用一次"""
        # 船身
        hull = pv.Cube(center=(0, 0, 0), x_length=1.1, y_length=0.45, z_length=0.25)
        # 船舱
        cabin = pv.Cube(center=(-0.12, 0, 0.22), x_length=0.35, y_length=0.28, z_length=0.18)
        # 船头三角
        bow_tip = np.array([[0.6, 0, 0.05], [0.35, 0.12, 0.05], [0.35, -0.12, 0.05]], dtype=np.float64)
        bow = pv.PolyData(bow_tip, faces=np.array([[3, 0, 1, 2]], dtype=np.int64))
        return hull, cabin, bow

    def _update_boat_position(self, pos, heading_rad):
        """原地更新船模位置和朝向（不重建网格，不重加Actor）"""
        x, y, z = pos
        heading_deg = heading_rad * 180 / math.pi
        for actor in [self.anim_robot, self.anim_robot_cabin, self._boat_bow_actor]:
            if actor is not None:
                try:
                    actor.SetPosition(x, y, z)
                    actor.SetOrientation(0, 0, heading_deg)
                except Exception:
                    pass

    def _add_boat_to_scene(self, pos, heading_rad):
        """首次将船模模板添加到场景（仅在动画开始时调用一次）"""
        x, y, z = pos
        heading_deg = heading_rad * 180 / math.pi
        hull_tpl, cabin_tpl, bow_tpl = self._boat_templates

        # 船身
        hull_copy = hull_tpl.copy()
        hull_copy.rotate_z(heading_deg, point=(0, 0, 0))
        hull_copy.translate((x, y, z))
        self.anim_robot = self.plotter.add_mesh(
            hull_copy, color="#e8590c", pbr=True, metallic=0.15, roughness=0.35,
            name="robot_hull", smooth_shading=True,
        )

        # 船舱
        cabin_copy = cabin_tpl.copy()
        cabin_copy.rotate_z(heading_deg, point=(0, 0, 0))
        cabin_copy.translate((x, y, z))
        self.anim_robot_cabin = self.plotter.add_mesh(
            cabin_copy, color="#ffffff", pbr=True, metallic=0.05, roughness=0.2,
            name="robot_cabin", smooth_shading=True,
        )

        # 船头三角
        bow_copy = bow_tpl.copy()
        bow_copy.rotate_z(heading_deg, point=(0, 0, 0))
        bow_copy.translate((x, y, z))
        self._boat_bow_actor = self.plotter.add_mesh(
            bow_copy, color="#ffcc00", pbr=True, name="robot_bow",
        )

    def _remove_boat_actors(self):
        """清理船形机器人所有部件"""
        for name in ["robot_hull", "robot_cabin", "robot_bow"]:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        self.anim_robot = None
        self.anim_robot_cabin = None
        self._boat_bow_actor = None

    # ═══════════════════ HUD 遥测 ═══════════════════

    def _update_hud(self, pos, heading_rad, speed_kn, depth_m):
        """更新 3D 场景中的 HUD 遥测文字叠加"""
        # 移除旧 HUD
        for t in self.hud_texts:
            try:
                self.plotter.remove_actor(t)
            except Exception:
                pass
        self.hud_texts = []

        heading_deg = heading_rad * 180 / math.pi
        heading_deg = heading_deg % 360

        g = self.grid
        depth_actual = abs(pos[2]) / max(1, g.nz) * g.depth.max() if g else abs(pos[2])

        battery_color = "#ff4444" if self.battery_pct < 25 else ("#ffcc00" if self.battery_pct < 50 else "#00ff88")

        hud_lines = [
            f"航速 {speed_kn:.1f} kn  |  深度 {depth_actual:.1f} m  |  航向 {heading_deg:.0f}°",
            f"坐标 ({pos[0]:.1f}, {pos[1]:.1f})  |  电量 {self.battery_pct:.0f}%",
        ]

        # 上半部分 — 白色遥测
        t1 = self.plotter.add_text(
            hud_lines[0], position="upper_left", font_size=11, color="#ffffff",
            name="hud_telemetry",
        )
        self.hud_texts.append(t1)

        # 下半部分用不同颜色显示电量和坐标
        if self.battery_pct < 25:
            batt_text = f"坐标 ({pos[0]:.1f}, {pos[1]:.1f})  |  ⚡电量 {self.battery_pct:.0f}% — 低电量警告!"
        else:
            batt_text = hud_lines[1]
        t2 = self.plotter.add_text(
            batt_text, position="upper_right", font_size=10, color=battery_color,
            name="hud_battery",
        )
        self.hud_texts.append(t2)

    # ═══════════════════ 途经点到达特效 ═══════════════════

    def _spawn_pulse(self, pos):
        """在指定位置生成脉冲扩散环"""
        x, y, z = pos
        ring = pv.Disc(center=(x, y, z), inner=0.25, outer=0.35, normal=(0, 0, 1),
                        r_res=32, c_res=8)
        actor = self.plotter.add_mesh(
            ring, color="#ffdd44", opacity=0.9, name=f"pulse_{len(self.pulse_rings)}",
        )
        self.pulse_rings.append([actor, 30])  # 30帧 ≈ 0.5秒

    def _update_pulses(self):
        """更新所有脉冲环（扩展+淡出）"""
        surviving = []
        for actor, remaining in self.pulse_rings:
            remaining -= 1
            if remaining <= 0:
                try:
                    self.plotter.remove_actor(actor)
                except Exception:
                    pass
                continue
            # 扩展环
            try:
                prop = actor.GetProperty()
                scale = 1.0 + (30 - remaining) * 0.15
                actor.SetScale(scale, scale, scale)
                prop.SetOpacity(max(0.05, remaining / 30 * 0.8))
            except Exception:
                pass
            surviving.append([actor, remaining])
        self.pulse_rings = surviving

    # ═══════════════════ 罗盘 ═══════════════════

    def _add_compass(self):
        """在场景中放置 3D 罗盘（指北针）"""
        g = self.grid
        cx, cy = g.nx - 2.5, g.ny - 1.5
        # N 方向箭头
        self.plotter.add_mesh(
            pv.Arrow(start=(cx, cy, 0.8), direction=(0, 1.2, 0),
                      tip_length=0.4, tip_radius=0.18, shaft_radius=0.08),
            color="#e74c3c", name="compass_n",
        )
        self.plotter.add_mesh(
            pv.Arrow(start=(cx, cy, 0.8), direction=(0, -0.6, 0),
                      tip_length=0.25, tip_radius=0.12, shaft_radius=0.06),
            color="#aaaaaa", name="compass_s",
        )
        self.plotter.add_point_labels(
            [[cx, cy + 1.8, 0.8]], ["N"],
            font_size=14, text_color="#e74c3c", point_size=1, name="compass_label",
        )

    # ═══════════════════ 路径统计 ═══════════════════

    def _compute_path_stats(self, path, grid):
        """计算路径详细统计"""
        if not path or len(path) < 2:
            return {"min_depth": 0, "max_depth": 0, "depth_range": 0,
                    "sharp_turns": 0, "downstream_pct": 0, "avg_speed_ms": 0}

        # 深度统计
        depths = []
        for p in path:
            d = grid.depth[min(p[1], grid.ny - 1), min(p[0], grid.nx - 1)]
            if d > 0:
                depths.append(d)
        min_d = min(depths) if depths else 0
        max_d = max(depths) if depths else 0

        # 急转弯检测 (>60度方向变化)
        sharp_turns = 0
        downstream_count = 0
        total_segments = len(path) - 1
        for i in range(1, len(path) - 1):
            prev_dx = path[i][0] - path[i - 1][0]
            prev_dy = path[i][1] - path[i - 1][1]
            next_dx = path[i + 1][0] - path[i][0]
            next_dy = path[i + 1][1] - path[i][1]

            prev_angle = math.atan2(prev_dy, prev_dx)
            next_angle = math.atan2(next_dy, next_dx)
            angle_diff = abs(next_angle - prev_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > math.pi / 3:  # >60度
                sharp_turns += 1

            # 顺流检测
            cur_vec, cur_spd = grid.get_current_at(path[i][0], path[i][1], path[i][2])
            if cur_spd > 0.01:
                move_angle = math.atan2(next_dy, next_dx)
                flow_angle = math.atan2(cur_vec[1], cur_vec[0])
                diff = abs(move_angle - flow_angle)
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                if diff < math.pi / 4:  # <45度=顺流
                    downstream_count += 1

        downstream_pct = downstream_count / max(1, total_segments) * 100

        # 均速（节→m/s）
        total_dist_km = sum(
            math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
            * grid.resolution / 1000
            for i in range(1, len(path))
        )
        avg_speed = 5.0  # 默认5节≈2.5m/s

        return {
            "min_depth": min_d,
            "max_depth": max_d,
            "depth_range": max_d - min_d,
            "sharp_turns": sharp_turns,
            "downstream_pct": downstream_pct,
            "avg_speed_ms": avg_speed,
        }

    # ═══════════════════ 动画 ═══════════════════

    def _anim_play(self):
        if not self.path3d or len(self.path3d) < 2:
            self.status_bar.showMessage("请先加载数据并规划路径")
            return
        # 清理旧动画（包括上次残留的尾迹）
        self._anim_cleanup()
        self.anim_trail = []
        self.anim_frame = 0
        self._frame_count = 0
        self.anim_playing = True

        # 初始化途经点追踪
        self.visited_waypoints = set()
        self.arrival_log = []

        # 计算能耗预算
        g = self.grid
        energy_info = compute_energy_estimate(g, self.path3d)
        self.total_energy_kj = energy_info["energy_consumption_kj"]
        self.battery_pct = 100.0
        total_frames = len(self.path3d) / max(0.01, self.anim_speed)
        self.energy_per_frame = self.total_energy_kj / max(1, total_frames) * 2

        # 预创建船模模板（只做一次）
        self._boat_templates = self._create_boat_template()

        # 首次添加船模到场景
        p0 = self.path3d[0]
        x0, y0, z0 = p0[0], p0[1], -p0[2]
        if len(self.path3d) >= 2:
            self.boat_heading = math.atan2(
                self.path3d[1][1] - self.path3d[0][1],
                self.path3d[1][0] - self.path3d[0][0],
            )
        else:
            self.boat_heading = 0.0
        self._add_boat_to_scene((x0, y0, z0), self.boat_heading)
        self.last_boat_pos = (x0, y0, z0)
        self.boat_speed_kn = 0.0

        self._show_fov()
        self._update_hud((x0, y0, z0), self.boat_heading, 0.0, abs(z0))
        if self.anim_timer is None:
            self.anim_timer = QTimer(self)
            self.anim_timer.timeout.connect(self._anim_tick)
        self.anim_timer.start(60)
        self.status_bar.showMessage("路径动画播放中... (拖拽视角可观察)")

    def _anim_pause(self):
        self.anim_playing = False
        if self.anim_timer:
            self.anim_timer.stop()
        self.status_bar.showMessage("动画已暂停 — 点击播放继续")

    def _anim_stop(self):
        self.anim_playing = False
        if self.anim_timer:
            self.anim_timer.stop()
        self._anim_cleanup()
        self.progress_label.setText("")
        self.status_bar.showMessage("动画已停止")

    def _on_speed_changed(self, text):
        speed_map = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
        self.anim_speed = speed_map.get(text, 1.0)
        if self.anim_playing:
            self.status_bar.showMessage(f"动画速度: {text}")

    def _toggle_camera_follow(self, checked):
        self.camera_follow = bool(checked)

    def _anim_cleanup(self):
        self.anim_frame = 0
        # 清理船形机器人
        self._remove_boat_actors()
        # 清理残留尾迹小球和渐显路径
        if hasattr(self, 'anim_trail'):
            for w in self.anim_trail:
                try:
                    self.plotter.remove_actor(w)
                except Exception:
                    pass
        self.anim_trail = []
        # 清理 HUD 文字
        for t in self.hud_texts:
            try:
                self.plotter.remove_actor(t)
            except Exception:
                pass
        self.hud_texts = []
        # 清理脉冲环
        for actor, _ in self.pulse_rings:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.pulse_rings = []
        # 清理到达标签
        for name in ["robot_hull", "robot_cabin", "robot_bow", "fov", "anim_trail"]:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        # 清理途经点到达标签
        for wi in range(len(self.grid.mission_waypoints if self.grid else [])):
            try:
                self.plotter.remove_actor(f"arrival_label_{wi}")
            except Exception:
                pass
        self.anim_robot = None
        self.anim_robot_cabin = None
        self._boat_bow_actor = None
        self._boat_templates = None
        self.fov_actor = None
        self.visited_waypoints = set()
        self.battery_pct = 100.0
        self._frame_count = 0

    def _anim_tick(self):
        if not self.anim_playing or not self.path3d:
            return
        self.anim_frame += self.anim_speed
        self._frame_count += 1
        idx = int(self.anim_frame)
        total = len(self.path3d)

        # 进度显示（每10帧更新）
        if self._frame_count % 10 == 0:
            pct = min(100, int(idx / max(1, total - 1) * 100))
            self.progress_label.setText(f"{pct}%")

        if idx >= total - 1:
            self.anim_playing = False
            if self.anim_timer:
                self.anim_timer.stop()
            self.progress_label.setText("100% ✓")
            wp_count = len(self.visited_waypoints)
            total_wp = len(self.grid.mission_waypoints or [])
            self.status_bar.showMessage(
                f"路径动画完成 ✓ — 途经点 {wp_count}/{total_wp} | 剩余电量 {self.battery_pct:.0f}%"
            )
            return

        # ── 当前位置（线性插值平滑） ──
        p_cur = self.path3d[idx]
        p_nxt = self.path3d[min(idx + 1, len(self.path3d) - 1)]
        frac = self.anim_frame - idx
        x = p_cur[0] + (p_nxt[0] - p_cur[0]) * frac
        y = p_cur[1] + (p_nxt[1] - p_cur[1]) * frac
        z_raw = p_cur[2] + (p_nxt[2] - p_cur[2]) * frac
        wave_z = 0.06 * math.sin(x * 1.2 + self.wave_time * 3) * math.cos(y * 1.0 + self.wave_time * 2.5)
        z = -z_raw + wave_z

        # ── 航向（每2帧更新） ──
        dx_f, dy_f = 1.0, 0.0
        if idx + 1 < len(self.path3d):
            dx_f = p_nxt[0] - p_cur[0]
            dy_f = p_nxt[1] - p_cur[1]
            if self._frame_count % 2 == 0 and (abs(dx_f) > 0.001 or abs(dy_f) > 0.001):
                self.boat_heading = math.atan2(dy_f, dx_f)

        # 速度计算
        if self.last_boat_pos and self._frame_count % 2 == 0:
            px, py, _ = self.last_boat_pos
            dist_m = math.sqrt((x - px)**2 + (y - py)**2) * self.grid.resolution
            speed_ms = dist_m / (0.12 / self.anim_speed)
            self.boat_speed_kn = speed_ms / 0.514
        self.last_boat_pos = (x, y, z)

        # ── 船模位置：原地更新，不重建网格（核心优化） ──
        self._update_boat_position((x, y, z), self.boat_heading)

        # ── 镜头跟随（每5帧） ──
        if self.camera_follow and self._frame_count % 5 == 0:
            try:
                # 相机在机器人后方上空跟随
                follow_dist = 8
                cam_x = x - math.cos(self.boat_heading) * follow_dist
                cam_y = y - math.sin(self.boat_heading) * follow_dist
                cam_z = z + follow_dist * 0.6
                self.plotter.camera_position = [
                    (cam_x, cam_y, cam_z),  # 相机位置
                    (x, y, z),              # 焦点（机器人位置）
                    (0, 0, 1),              # 上方向
                ]
            except Exception:
                pass

        # ── FOV：每5帧更新 ──
        if self._frame_count % 5 == 0:
            self._update_fov((x, y, -z_raw), dx_f, dy_f)

        # ── HUD：每15帧更新（~0.9秒间隔） ──
        if self._frame_count % 15 == 0:
            depth_display = abs(z_raw) / max(1, self.grid.nz) * self.grid.depth.max()
            self._update_hud((x, y, z), self.boat_heading, self.boat_speed_kn, depth_display)

        # ── 途经点到达检测（每5帧） ──
        if self._frame_count % 5 == 0:
            waypoints_all = self.grid.mission_waypoints or []
            for wi, wp in enumerate(waypoints_all):
                if wi in self.visited_waypoints:
                    continue
                dist_to_wp = math.sqrt((x - wp[0])**2 + (y - wp[1])**2 + (z + wp[2])**2)
                if dist_to_wp < 1.2:
                    self.visited_waypoints.add(wi)
                    self._spawn_pulse((wp[0], wp[1], -wp[2]))
                    self.status_bar.showMessage(
                        f"到达途经点 {wi+1}/{len(waypoints_all)} — 坐标 ({wp[0]}, {wp[1]})"
                    )
                    self.plotter.add_point_labels(
                        [[wp[0], wp[1], -wp[2] + 0.8]], [f"✓ WP{wi+1}"],
                        font_size=14, text_color="#ffdd44", point_size=1,
                        name=f"arrival_label_{wi}",
                    )

        # ── 脉冲特效 ──
        self._update_pulses()

        # ── 电量消耗（每2帧） ──
        if self._frame_count % 2 == 0:
            self.battery_pct = max(5, self.battery_pct - self.energy_per_frame * 2 / max(1, self.total_energy_kj) * 100)
            self.lbl_battery.setText(f"电量: {self.battery_pct:.0f}% {'🔴' if self.battery_pct < 25 else '🟡' if self.battery_pct < 50 else '🟢'}")

        # ── 尾迹：每8帧，最多40个 ──
        if self._frame_count % 8 == 0 and len(self.anim_trail) < 40:
            wake_dot = self.plotter.add_mesh(
                pv.Sphere(center=(x, y, z + 0.05), radius=0.08),
                color="#88ccff", opacity=0.5, name=f"wake_{self._frame_count}",
            )
            self.anim_trail.append(wake_dot)
        # 衰减尾迹（每10帧），移除最旧的
        if self._frame_count % 10 == 0:
            for w in self.anim_trail:
                try:
                    w.GetProperty().SetOpacity(max(0.05, w.GetProperty().GetOpacity() - 0.06))
                except Exception:
                    pass
            # 移除透明度为0的尾迹
            while len(self.anim_trail) > 40:
                old = self.anim_trail.pop(0)
                try:
                    self.plotter.remove_actor(old)
                except Exception:
                    pass

        # ── 路径渐显：每30帧 ──
        if self._frame_count % 30 == 0:
            partial = np.array([[q[0], q[1], -q[2]] for q in self.path3d[:idx + 1]], dtype=np.float64)
            if len(partial) >= 2:
                try:
                    tube = pv.Spline(partial, n_points=max(len(partial), 3) * 2).tube(radius=0.08)
                    self.plotter.add_mesh(tube, color="#ff9944", name="anim_trail", opacity=0.7, pbr=True)
                except Exception:
                    pass

    def _show_fov(self):
        if self.fov_actor:
            self.plotter.remove_actor(self.fov_actor)
        if self.path3d:
            p = self.path3d[0]
            cone = pv.Cone(center=(p[0], p[1], -p[2]), direction=(1, 0, 0),
                                 height=3.5, radius=1.8, resolution=20)
            self.fov_actor = self.plotter.add_mesh(cone, color="#ffcc44", opacity=0.22, name="fov")

    def _update_fov(self, pos, dx, dy):
        if self.fov_actor:
            self.plotter.remove_actor(self.fov_actor)
        mag = (dx * dx + dy * dy) ** 0.5 or 1.0
        cone = pv.Cone(center=pos, direction=(dx / mag, dy / mag, 0.05),
                             height=3.5, radius=1.8, resolution=20)
        self.fov_actor = self.plotter.add_mesh(cone, color="#ffcc44", opacity=0.22, name="fov")

    # ═══════════════════ 任务保存/加载 ═══════════════════

    def _save_mission(self):
        """快捷保存当前任务配置（起点/途经点/终点/障碍物）"""
        g = self.grid
        if not g:
            QMessageBox.warning(self, "无数据", "请先加载场景数据")
            return
        p, _ = QFileDialog.getSaveFileName(self, "保存任务配置", "mission_config.json", "JSON (*.json)")
        if not p:
            return
        data = {
            "start": list(g.mission_start) if g.mission_start else None,
            "waypoints": [list(w) for w in g.mission_waypoints],
            "end": list(g.mission_end) if g.mission_end else None,
            "obstacles": [
                [int(z), int(y), int(x)]
                for z in range(g.nz) for y in range(g.ny) for x in range(g.nx)
                if g.obstacles[z, y, x]
            ],
            "strategy": self.current_strategy,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.status_bar.showMessage(f"任务已保存: {os.path.basename(p)}")

    def _load_mission(self):
        """快捷加载任务配置"""
        g = self.grid
        if not g:
            QMessageBox.warning(self, "无数据", "请先加载场景数据")
            return
        p, _ = QFileDialog.getOpenFileName(self, "加载任务配置", "", "JSON (*.json);;所有文件 (*)")
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._push_undo()
            g.mission_start = tuple(data["start"]) if data.get("start") else None
            g.mission_waypoints = [tuple(w) for w in data.get("waypoints", [])]
            g.mission_end = tuple(data["end"]) if data.get("end") else None
            # 清空并重建障碍物
            g.obstacles[:] = False
            for obs in data.get("obstacles", []):
                z, y, x = obs
                if 0 <= x < g.nx and 0 <= y < g.ny and 0 <= z < g.nz:
                    g.obstacles[z, y, x] = True
            # 恢复策略
            strat = data.get("strategy", "balanced")
            if strat in ("balanced", "safe", "fast", "energy"):
                self.current_strategy = strat
                strategy_map = {"balanced": 0, "safe": 1, "fast": 2, "energy": 3}
                self.strategy_combo.setCurrentIndex(strategy_map.get(strat, 0))
            # 重绘
            self.path3d = None
            self.path_actor = None
            self.plotter.clear()
            self._draw()
            self._refresh_info()
            self._refresh_wp_list()
            self.status_bar.showMessage(f"任务已加载: {os.path.basename(p)} — "
                                       f"起点={g.mission_start}, 途经点={len(g.mission_waypoints)}, 终点={g.mission_end}")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法解析任务文件:\n{e}")

    # ═══════════════════ 撤销 ═══════════════════

    def _push_undo(self):
        if self.grid:
            import copy
            state = {
                "waypoints": list(self.grid.mission_waypoints),
                "obstacles": self.grid.obstacles.copy(),
                "start": self.grid.mission_start,
                "end": self.grid.mission_end,
            }
            self._undo_stack.append(state)
            if len(self._undo_stack) > 20:
                self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack or not self.grid:
            return
        state = self._undo_stack.pop()
        self.grid.mission_waypoints = state["waypoints"]
        self.grid.obstacles = state["obstacles"]
        self.grid.mission_start = state.get("start")
        self.grid.mission_end = state.get("end")
        self.plotter.clear()
        self._draw()
        self._refresh_info()
        self._refresh_wp_list()
        self.status_bar.showMessage("已撤销")

    # ═══════════════════ 导出 ═══════════════════

    def _export_screenshot(self):
        p, _ = QFileDialog.getSaveFileName(self, "导出截图", "screenshot.png", "PNG (*.png);;JPEG (*.jpg)")
        if p:
            self.plotter.screenshot(p)
            self.status_bar.showMessage(f"截图已保存: {p}")

    def _export_report(self):
        if not self.grid:
            return
        p, _ = QFileDialog.getSaveFileName(self, "导出报告", "report.txt", "文本 (*.txt);;Markdown (*.md)")
        if not p:
            return
        g = self.grid
        n_obs = int(np.sum(g.obstacles))
        d, f, dc = (0, 0, 0)
        energy_info = None
        if self.path3d:
            d, f, dc = compute_3d_path_cost(g, self.path3d)
            energy_info = compute_energy_estimate(g, self.path3d)
        lines = [
            "=== 水域机器人路径规划报告 ===",
            f"网格: {g.nx} x {g.ny} x {g.nz} @ {g.resolution}m/格",
            f"障碍物数量: {n_obs}",
            f"途经点数量: {len(g.mission_waypoints)}",
            f"起点: {g.mission_start}",
            f"终点: {g.mission_end or '未设定'}",
            "",
            "--- 路径结果 ---",
            f"总距离: {d:,.0f} m",
            f"水流代价: {f:,.0f}",
            f"深度代价: {dc:,.0f}",
        ]
        if energy_info:
            lines += [
                f"能耗估算: {energy_info['energy_consumption_kj']:,.0f} kJ",
                f"预估时间: {energy_info['estimated_time_min']:.1f} min",
                f"路径点数: {energy_info['waypoint_count']}",
                f"水压代价: {energy_info.get('pressure_cost', 0):.0f}",
                f"天气代价: {energy_info.get('weather_cost', 0):.0f}",
                f"水温代价: {energy_info.get('temperature_cost', 0):.0f}",
                f"能见度代价: {energy_info.get('visibility_cost', 0):.0f}",
                f"潮汐放大: {energy_info.get('tidal_amplification', 0):.2f}x",
            ]
        else:
            lines.append(f"路径点数: {len(self.path3d) if self.path3d else 0}")
        lines += [
            "",
            "--- 路径坐标 ---",
        ]
        if self.path3d:
            for i, pt in enumerate(self.path3d):
                if i % 10 == 0:
                    lines.append(f"  [{i:3d}] ({pt[0]:2d}, {pt[1]:2d}, {pt[2]:2d})")
            lines.append(f"  [{len(self.path3d)-1:3d}] {self.path3d[-1]}")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        self.status_bar.showMessage(f"报告已导出: {p}")

    def _refresh_info(self):
        g = self.grid
        if g is None:
            return
        n_obs = int(np.sum(g.obstacles))
        # 平均水压 (表层)
        avg_pressure = g.pressure[0, g.depth > 0].mean() if (g.depth > 0).any() else 0
        # 天气
        wind_dir, wind_spd, wave_h = g.get_weather_at(g.nx // 2, g.ny // 2)
        n_eddies = len(g.eddies)
        # 水温和能见度
        avg_temp = g.temperature[g.depth > 0].mean() if (g.depth > 0).any() else 0
        avg_vis = g.visibility[g.depth > 0].mean() if (g.depth > 0).any() else 0
        tidal_label = {0: "低潮", 0.25: "退潮中", 0.5: "半潮", 0.75: "涨潮中", 1.0: "高潮"}.get(
            round(g.tidal_phase * 4) / 4, f"相位{g.tidal_phase:.2f}"
        )
        info = (
            f"网格: {g.nx} x {g.ny} x {g.nz}  @{g.resolution:.0f}m/格\n"
            f"障碍物: {n_obs}  途经点: {len(g.mission_waypoints or [])}\n"
            f"水深: {g.depth[g.depth>0].min():.0f}~{g.depth.max():.0f}m  "
            f"水压: {avg_pressure:.0f}kPa\n"
            f"天气: 风速{wind_spd:.0f}m/s  浪高{wave_h:.1f}m  漩涡: {n_eddies}个\n"
            f"水温: {avg_temp:.1f}°C  能见度: {avg_vis:.1f}m  潮汐: {tidal_label}\n"
            f"起点: {g.mission_start or '未设定'}"
        )
        self.lbl_info.setPlainText(info)

    # ═══════════════════ 智能决策 ═══════════════════

    def _open_llm_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("大模型 API 设置")
        dlg.setMinimumWidth(420)
        layout = QFormLayout(dlg)
        layout.setSpacing(10)

        api_key = QLineEdit(self.llm_api_key)
        api_key.setEchoMode(QLineEdit.EchoMode.Password)
        api_key.setPlaceholderText("sk-... (支持 OpenAI / DeepSeek / 兼容接口)")
        layout.addRow("API Key:", api_key)

        base_url = QLineEdit(self.llm_base_url)
        base_url.setPlaceholderText("https://api.deepseek.com")
        layout.addRow("Base URL:", base_url)

        model = QComboBox()
        model.setEditable(True)
        model.addItems(["deepseek-chat", "deepseek-reasoner", "gpt-4o", "gpt-4o-mini",
                         "qwen-plus", "glm-4", "moonshot-v1-8k"])
        model.setCurrentText(self.llm_model)
        layout.addRow("模型:", model)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                     QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        layout.addRow(btn_box)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.llm_api_key = api_key.text().strip()
            self.llm_base_url = base_url.text().strip()
            self.llm_model = model.currentText().strip()
            self.llm_enabled = bool(self.llm_api_key)
            if self.llm_enabled:
                self.planner = TaskPlanner(self.llm_api_key, self.llm_base_url, self.llm_model)
                self.lbl_ai.setText(f"大模型已配置: {self.llm_model}")
                self.status_bar.showMessage(f"大模型已连接: {self.llm_model}")
                # 同步菜单勾选状态和UI
                self.ai_toggle_action.setChecked(True)
                for w in self._manual_strategy_widgets:
                    w.hide()
                self._ai_strategy_label.show()
                self._ai_strategy_label.setText(f"AI 自动决策 (当前: {self._strategy_cn()})")
            else:
                self.planner = None
                self.lbl_ai.setText("大模型未配置 — 请先设置 API Key")
                self.ai_toggle_action.setChecked(False)
                for w in self._manual_strategy_widgets:
                    w.show()
                self._ai_strategy_label.hide()

    def _toggle_llm(self, checked):
        self.llm_enabled = bool(checked) and bool(self.llm_api_key)
        if self.llm_enabled:
            self.planner = TaskPlanner(self.llm_api_key, self.llm_base_url, self.llm_model)
            self.lbl_ai.setText(f"智能决策已启用: {self.llm_model}")
            # 隐藏手动策略选择，显示AI自动决策标签
            for w in self._manual_strategy_widgets:
                w.hide()
            self._ai_strategy_label.show()
            self._ai_strategy_label.setText(f"AI 自动决策 (当前: {self._strategy_cn()})")
        else:
            self.ai_toggle_action.setChecked(False)
            self.planner = None
            self.lbl_ai.setText("智能决策已禁用")
            # 恢复手动策略选择
            for w in self._manual_strategy_widgets:
                w.show()
            self._ai_strategy_label.hide()

    def _strategy_cn(self):
        """当前策略中文名"""
        m = {"balanced": "均衡", "safe": "安全优先", "fast": "速度优先", "energy": "节能优先"}
        return m.get(self.current_strategy, self.current_strategy)

    def _ai_analyze(self):
        if not self.llm_enabled or self.planner is None:
            QMessageBox.warning(self, "大模型未配置", "请先在菜单 决策 > 大模型设置 中配置 API Key")
            return
        if self.grid is None:
            QMessageBox.warning(self, "无数据", "请先加载水况数据")
            return

        instruction = self.ai_input.text().strip() or "分析当前场景，给出最优路径建议"
        n_obs = int(np.sum(self.grid.obstacles))
        n_wp = len(self.grid.mission_waypoints)

        context = (
            f"网格: {self.grid.nx}x{self.grid.ny}x{self.grid.nz}, 精度: {self.grid.resolution}m/格\n"
            f"障碍物: {n_obs}个, 途经点: {n_wp}个\n"
            f"起点: {self.grid.mission_start}, 终点: {self.grid.mission_end or '未设定'}\n"
            f"水深范围: {self.grid.depth[self.grid.depth>0].min():.1f}~{self.grid.depth.max():.1f}m\n"
            f"水流: 表层 {self.grid.current_speeds['surface'].mean():.2f} m/s"
        )

        self.status_bar.showMessage("智能分析中...")
        try:
            result = self.planner.analyze_scene(instruction, context)
            suggested = result.get('suggested_strategy', 'balanced')
            # 应用AI建议的策略
            if suggested in ("balanced", "safe", "fast", "energy"):
                self.current_strategy = suggested
                self._ai_strategy_label.setText(f"AI 自动决策 (当前: {self._strategy_cn()})")
                # 自动重新规划
                if self.grid and self.grid.mission_start and (self.grid.mission_waypoints or self.grid.mission_end):
                    self._replan()
            self.lbl_ai.setText(
                f"{result.get('recommendation', '')}\n"
                f"策略: {suggested} | "
                f"风险: {result.get('risk_assessment', 'N/A')}"
            )
            self.status_bar.showMessage(f"AI决策完成: 策略={suggested} 已自动应用")
        except Exception as e:
            self.lbl_ai.setText(f"调用失败: {e}")
            plan = rule_based_plan(instruction)
            self.lbl_ai.setText(f"[规则模式] {plan.get('summary', '')}")
            self.status_bar.showMessage("调用失败，已使用规则模式")

    def _ai_task_plan(self):
        if not self.llm_enabled or self.planner is None:
            QMessageBox.warning(self, "大模型未配置", "请先在菜单 决策 > 大模型设置 中配置 API Key")
            return
        if self.grid is None:
            QMessageBox.warning(self, "无数据", "请先加载水况数据")
            return

        instruction = self.ai_input.text().strip()
        if not instruction:
            instruction = "从起点出发，经过所有途经点，最后返回码头"
        self.ai_input.setText(instruction)

        self.status_bar.showMessage("任务规划中...")
        try:
            plan = self.planner.plan(instruction)
            result = self.planner.format_plan_display(plan)
            self.lbl_ai.setText(result)
            self.lbl_info.setPlainText(self.lbl_info.toPlainText() + "\n\n" + result)
            self.status_bar.showMessage("任务规划完成")
        except Exception as e:
            plan = rule_based_plan(instruction)
            self.lbl_ai.setText(f"[规则模式] {plan.get('summary', '')}")
            self.status_bar.showMessage(f"调用失败: {e}，已使用规则模式")

    def closeEvent(self, event):
        if self.wave_timer:
            self.wave_timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 绕过 Windows 原生风格，让 QSS 生效
    app.setStyleSheet(QSS)
    app.setFont(QFont("Microsoft YaHei", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
