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

from environment.water_3d import Water3DGrid, demo_3d_coastal, demo_3d_river
from planning.astar3d import plan_tsp_3d, compute_3d_path_cost
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
QMenuBar {{ background: {C['panel']}; border-bottom: 1px solid {C['border']}; padding: 2px 8px; font-size: 13px; }}
QMenuBar::item:selected {{ background: #e0e8f0; border-radius: 4px; }}
QMenu {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px; padding: 4px; }}
QMenu::item {{ padding: 6px 28px; border-radius: 4px; }}
QMenu::item:selected {{ background: #dce8f4; }}
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
QGroupBox {{ font-weight: bold; border: 1px solid {C['border']}; border-radius: 8px; margin-top: 12px; padding-top: 16px; background: {C['panel']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {C['text']}; }}
QTextEdit {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px; font-size: 12px; color: {C['text']}; }}
QLabel {{ color: {C['text']}; }}
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

        # LLM 配置
        self.llm_enabled = False
        self.llm_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.llm_base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
        self.llm_model = os.environ.get("LLM_MODEL", "deepseek-chat")
        self.planner: TaskPlanner = None

        self._build_ui()
        self._setup_3d()
        self._load_coastal()

    # ═══════════════════ UI ═══════════════════

    def _build_ui(self):
        # 菜单栏
        mb = self.menuBar()
        file_menu = mb.addMenu("文件(&F)")
        file_menu.addAction("打开 JSON...", self._open_json, "Ctrl+O")
        file_menu.addAction("打开原始数据...", self._open_raw, "Ctrl+Shift+O")
        file_menu.addSeparator()
        file_menu.addAction("导出 JSON...", self._export_json)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close, "Alt+F4")

        ai_menu = mb.addMenu("AI(&A)")
        ai_menu.addAction("LLM 设置...", self._open_llm_settings)
        self.ai_toggle_action = QAction("启用 AI 决策", self)
        self.ai_toggle_action.setCheckable(True)
        self.ai_toggle_action.toggled.connect(self._toggle_llm)
        ai_menu.addAction(self.ai_toggle_action)

        demo_menu = mb.addMenu("演示(&D)")
        demo_menu.addAction("沿海水域", self._load_coastal)
        demo_menu.addAction("内河航道", self._load_river)

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
        tb.addSeparator()
        tb.addWidget(self._btn("重新规划", self._replan, primary=True))
        tb.addWidget(self._btn("清除途经点", self._clear_wp, danger=True))
        tb.addWidget(self._btn("清除障碍物", self._clear_obs, danger=True))

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 — 使用 Ctrl+点击放置障碍物, Shift+点击添加途经点")

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
        self.lbl_dist = QLabel("总距离: --"); self.lbl_dist.setStyleSheet("font-size:16px;font-weight:bold;color:#e8590c")
        self.lbl_flow = QLabel("水流代价: --"); self.lbl_flow.setStyleSheet("font-size:14px;color:#2980b9")
        self.lbl_depth = QLabel("深度代价: --"); self.lbl_depth.setStyleSheet("font-size:14px;color:#27ae60")
        l2.addWidget(self.lbl_dist, 0, 0)
        l2.addWidget(self.lbl_flow, 1, 0)
        l2.addWidget(self.lbl_depth, 2, 0)
        lay.addWidget(g2)

        # AI 决策
        g3 = QGroupBox("AI 决策")
        l3 = QVBoxLayout(g3)
        l3.setSpacing(6)
        self.ai_input = QLineEdit()
        self.ai_input.setPlaceholderText("输入自然语言指令，如：避开暗礁区优先到达目标")
        self.ai_input.setMinimumHeight(32)
        l3.addWidget(self.ai_input)
        h_ai = QHBoxLayout()
        self.btn_analyze = QPushButton("AI 分析决策")
        self.btn_analyze.setObjectName("btnPrimary")
        self.btn_analyze.clicked.connect(self._ai_analyze)
        h_ai.addWidget(self.btn_analyze)
        self.btn_plan_task = QPushButton("AI 任务规划")
        self.btn_plan_task.clicked.connect(self._ai_task_plan)
        h_ai.addWidget(self.btn_plan_task)
        l3.addLayout(h_ai)
        self.lbl_ai = QLabel("LLM 未配置 — 请先设置 API")
        self.lbl_ai.setStyleSheet("color:#7f8c8d; font-size:11px;")
        self.lbl_ai.setWordWrap(True)
        l3.addWidget(self.lbl_ai)
        lay.addWidget(g3)

        # 操作提示
        g4 = QGroupBox("操作提示")
        l4 = QVBoxLayout(g4)
        tips = QLabel(
            "Ctrl + 点击 = 放置/移除障碍物\n"
            "Shift + 点击 = 添加途经点\n"
            "左键拖拽 = 旋转 | 右键 = 平移\n"
            "滚轮 = 缩放"
        )
        tips.setStyleSheet("color:#7f8c8d; font-size:12px;")
        l4.addWidget(tips)
        lay.addWidget(g4)

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

        self.plotter.enable_point_picking(
            callback=self._on_pick, show_point=True,
            show_message="点击场景选取坐标 | Ctrl=障碍物 Shift=途经点", font_size=12,
        )
        # 键盘修饰键 — 用 Qt 事件
        self.plotter.iren.interactor.AddObserver("KeyPressEvent", self._on_key)
        self.plotter.iren.interactor.AddObserver("KeyReleaseEvent", self._on_key_up)

    def _on_key(self, obj, event):
        key = obj.GetKeySym()
        if key == "Control_L" or key == "Control_R":
            self.ctrl_held = True
        elif key == "Shift_L" or key == "Shift_R":
            self.shift_held = True

    def _on_key_up(self, obj, event):
        key = obj.GetKeySym()
        if key == "Control_L" or key == "Control_R":
            self.ctrl_held = False
        elif key == "Shift_L" or key == "Shift_R":
            self.shift_held = False

    # ═══════════════════ 数据加载 ═══════════════════

    def load(self, grid):
        self.grid = grid
        self.path3d = None
        self.wave_time = 0.0
        self.plotter.clear()
        self._draw()
        self._start_waves()
        self.plotter.reset_camera()
        self._refresh_info()

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

    def _load_coastal(self):
        self.load(demo_3d_coastal())
    def _load_river(self):
        self.load(demo_3d_river())

    # ═══════════════════ 渲染 ═══════════════════

    def _draw(self):
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz

        # 海底
        dv = np.full((ny, nx), np.nan, dtype=np.float32)
        for y in range(ny):
            for x in range(nx):
                if g.depth[y, x] > 0:
                    dv[y, x] = -(g.depth[y, x] / max(1, g.depth.max())) * nz * 0.8
        gx, gy = np.meshgrid(range(nx), range(ny))
        self.plotter.add_mesh(pv.StructuredGrid(gx, gy, dv), color="#889966", opacity=0.45, name="seabed", show_edges=False)

        # 障碍物
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if g.obstacles[z, y, x]:
                        self.plotter.add_mesh(
                            pv.Cube(center=(x, y, -z - 0.5), x_length=0.85, y_length=0.85, z_length=0.85),
                            color=C["danger"], opacity=0.88, name=f"o.{x}.{y}.{z}",
                        )

        # 水流
        step = max(1, min(nx, ny) // 10)
        for y in range(0, ny, step):
            for x in range(0, nx, step):
                if g.depth[y, x] <= 0:
                    continue
                (dx, dy, _), sp = g.get_current_at(x, y, 0)
                if sp > 0.01:
                    self.plotter.add_mesh(
                        pv.Arrow(start=(x, y, 0.3), direction=(dx * sp * 3, dy * sp * 3, 0),
                                  tip_length=0.25, tip_radius=0.08, shaft_radius=0.03),
                        color=C["accent"], opacity=0.5,
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
        if g.mission_end and g.mission_end != g.mission_start:
            e = g.mission_end
            be = pv.Sphere(center=(e[0], e[1], -e[2]), radius=0.5)
            self.plotter.add_mesh(be, color=C["end"], pbr=True, name="end")

        # 路径
        if g.mission_start and g.mission_waypoints:
            self._plan()

    def _plan(self):
        g = self.grid
        if not (g and g.mission_start and g.mission_waypoints):
            return
        self.path3d = plan_tsp_3d(g, g.mission_start, g.mission_waypoints, g.mission_end)
        if self.path3d is None:
            return
        if self.path_actor:
            self.plotter.remove_actor(self.path_actor)
        pts = np.array([[p[0], p[1], -p[2]] for p in self.path3d], dtype=np.float64)
        if len(pts) >= 2:
            tube = pv.Spline(pts, n_points=len(pts) * 3).tube(radius=0.12)
            self.path_actor = self.plotter.add_mesh(tube, color=C["path"], pbr=True, metallic=0.1, name="path")
        d, f, dc = compute_3d_path_cost(g, self.path3d)
        self.lbl_dist.setText(f"总距离: {d:,.0f} m")
        self.lbl_flow.setText(f"水流代价: {f:,.0f}")
        self.lbl_depth.setText(f"深度代价: {dc:,.0f}")

    def _start_waves(self):
        if self.wave_timer is not None:
            return
        self.wave_timer = QTimer(self)
        self.wave_timer.timeout.connect(self._animate_wave)
        self.wave_timer.start(80)

    def _animate_wave(self):
        if self.grid is None:
            return
        self.wave_time += 0.08
        r = max(2, min(self.grid.nx, self.grid.ny) // 2)
        xx, yy = np.meshgrid(np.linspace(0, self.grid.nx, r * 2), np.linspace(0, self.grid.ny, r * 2))
        t = self.wave_time
        zz = (0.12 * np.sin(xx * 0.6 + t * 2.5) * np.cos(yy * 0.5 + t * 1.8)
              + 0.08 * np.sin(xx * 0.3 - t * 1.3) * np.sin(yy * 0.7 + t * 2.1)
              + 0.05 * np.cos((xx + yy) * 0.4 + t * 3.0))
        self.plotter.add_mesh(pv.StructuredGrid(xx, yy, zz), color="#3399bb",
                               opacity=0.32, name="wave", show_edges=False, smooth_shading=True)

    # ═══════════════════ 交互 ═══════════════════

    def _on_pick(self, point):
        if point is None or self.grid is None:
            return
        try:
            pt = point.points[0] if hasattr(point, "points") else point
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            z = -int(round(float(pt[2])))
        except Exception:
            return
        if not (0 <= x < self.grid.nx and 0 <= y < self.grid.ny and 0 <= z < self.grid.nz):
            return
        if self.ctrl_held:
            self.grid.obstacles[z, y, x] = not self.grid.obstacles[z, y, x]
        elif self.shift_held:
            self.grid.mission_waypoints.append((x, y, z))
        else:
            return
        self.plotter.clear()
        self._draw()
        self._refresh_info()

    def _replan(self):
        if self.grid:
            self._plan()

    def _clear_wp(self):
        if self.grid:
            self.grid.mission_waypoints = []
            self.plotter.clear()
            self._draw()

    def _clear_obs(self):
        if self.grid:
            self.grid.obstacles[:] = False
            self.plotter.clear()
            self._draw()

    def _reset_view(self):
        self.plotter.reset_camera()

    def _refresh_info(self):
        g = self.grid
        if g is None:
            return
        n_obs = int(np.sum(g.obstacles))
        self.lbl_info.setPlainText(
            f"网格: {g.nx} × {g.ny} × {g.nz}\n"
            f"精度: {g.resolution:.0f} m/格\n"
            f"障碍物: {n_obs}  途经点: {len(g.mission_waypoints)}\n"
            f"起点: {g.mission_start or '未设定'}"
        )

    # ═══════════════════ AI 决策 ═══════════════════

    def _open_llm_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("LLM API 设置")
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
                self.lbl_ai.setText(f"LLM 已配置: {self.llm_model}")
                self.status_bar.showMessage(f"LLM 已连接: {self.llm_model}")
            else:
                self.planner = None
                self.lbl_ai.setText("LLM 未配置 — 请先设置 API")

    def _toggle_llm(self, checked):
        self.llm_enabled = bool(checked) and bool(self.llm_api_key)
        if self.llm_enabled:
            self.planner = TaskPlanner(self.llm_api_key, self.llm_base_url, self.llm_model)
            self.lbl_ai.setText(f"LLM 已启用: {self.llm_model}")
        else:
            self.ai_toggle_action.setChecked(False)
            self.planner = None
            self.lbl_ai.setText("LLM 已禁用")

    def _ai_analyze(self):
        if not self.llm_enabled or self.planner is None:
            QMessageBox.warning(self, "LLM 未配置", "请先在菜单 AI → LLM 设置中配置 API Key")
            return
        if self.grid is None:
            QMessageBox.warning(self, "无数据", "请先加载水况数据")
            return

        instruction = self.ai_input.text().strip() or "分析当前场景，给出最优路径建议"
        n_obs = int(np.sum(self.grid.obstacles))
        n_wp = len(self.grid.mission_waypoints)

        # 构建场景上下文
        context = (
            f"网格: {self.grid.nx}×{self.grid.ny}×{self.grid.nz}, 精度: {self.grid.resolution}m/格\n"
            f"障碍物: {n_obs}个, 途经点: {n_wp}个\n"
            f"起点: {self.grid.mission_start}, 终点: {self.grid.mission_end or '未设定'}\n"
            f"水深范围: {self.grid.depth[self.grid.depth>0].min():.1f}~{self.grid.depth.max():.1f}m\n"
            f"水流: 表层 {self.grid.current_speeds['surface'].mean():.2f} m/s"
        )

        self.status_bar.showMessage("AI 分析中...")
        try:
            result = self.planner.analyze_scene(instruction, context)
            self.lbl_ai.setText(
                f"[AI] {result.get('recommendation', '')}\n"
                f"策略: {result.get('suggested_strategy', 'N/A')} | "
                f"风险: {result.get('risk_assessment', 'N/A')}"
            )
            self.status_bar.showMessage(f"AI 决策完成: {result.get('suggested_strategy', '')}")
        except Exception as e:
            self.lbl_ai.setText(f"AI 调用失败: {e}")
            # 回退到规则模式
            plan = rule_based_plan(instruction)
            self.lbl_ai.setText(f"[规则模式] {plan.get('summary', '')}")
            self.status_bar.showMessage("AI 调用失败，已使用规则模式")

    def _ai_task_plan(self):
        if not self.llm_enabled or self.planner is None:
            QMessageBox.warning(self, "LLM 未配置", "请先在菜单 AI → LLM 设置中配置 API Key")
            return
        if self.grid is None:
            QMessageBox.warning(self, "无数据", "请先加载水况数据")
            return

        instruction = self.ai_input.text().strip()
        if not instruction:
            instruction = "从起点出发，经过所有途经点，最后返回码头"
        self.ai_input.setText(instruction)

        self.status_bar.showMessage("AI 任务规划中...")
        try:
            plan = self.planner.plan(instruction)
            result = self.planner.format_plan_display(plan)
            self.lbl_ai.setText(result)
            self.lbl_info.setPlainText(self.lbl_info.toPlainText() + "\n\n" + result)
            self.status_bar.showMessage("AI 任务规划完成")
        except Exception as e:
            plan = rule_based_plan(instruction)
            self.lbl_ai.setText(f"[规则模式] {plan.get('summary', '')}")
            self.status_bar.showMessage(f"AI 调用失败: {e}，已使用规则模式")

    def closeEvent(self, event):
        if self.wave_timer:
            self.wave_timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setFont(QFont("Microsoft YaHei", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
