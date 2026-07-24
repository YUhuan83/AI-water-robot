"""
水域机器人 3D 智能决策平台

功能: 导入水况数据 → 动态海浪模拟 → 自主路径规划 → 3D 交互式可视化
操作: 鼠标拖拽旋转/缩放 | Ctrl+点击切换障碍物 | Shift+点击添加途经点
"""

import os
import sys
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.water_3d import Water3DGrid, demo_3d_coastal, demo_3d_river
from planning.astar3d import plan_tsp_3d, compute_3d_path_cost
from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 亮色主题配色 ──
BG_VIEW = "#dce8f0"        # 3D 视图背景
BG_UI = "#eef3f6"          # 面板背景
FG_TEXT = "#2a4050"        # 文字色
FG_TITLE = "#1a3040"       # 标题色
WATER_COLOR = "#3388bb"    # 水面
WATER_ALPHA = 0.35
SEABED_COLOR = "#889966"   # 海底
OBS_COLOR = "#d04040"      # 障碍物
PATH_COLOR = "#ff6600"     # 路径
START_COLOR = "#22aa55"    # 起点
WP_COLOR = "#ee8822"       # 途经点
END_COLOR = "#dd3333"      # 终点
CURRENT_COLOR = "#2288cc"  # 水流箭头

FONT_UI = ("Microsoft YaHei", 9)
FONT_BOLD = ("Microsoft YaHei", 9, "bold")
FONT_MONO = ("Consolas", 10)


# ═══════════════════════════════════════════════════════════
# 3D 场景
# ═══════════════════════════════════════════════════════════

class Scene3D:
    def __init__(self, panel):
        import pyvista as pv
        self.pv = pv
        self.panel = panel
        self.grid = None
        self.path3d = None
        self.path_actor = None
        self.wave_mesh = None
        self.wave_time = 0.0
        self.wave_running = False
        self.nx = self.ny = 20

        self.plotter = pv.Plotter(window_size=(820, 640))
        self.plotter.set_background(BG_VIEW)
        self.plotter.add_axes(color="#889999", xlabel="东 X", ylabel="北 Y", zlabel="深 Z")

        self.plotter.add_text(
            "导入水况数据或选择演示场景以开始\n数据加载后将自动生成动态海浪并规划路径",
            position="upper_left", font_size=12, color="#556666",
        )
        self._grid_ref()
        self._init_empty_wave()

        # 点击选取
        self.plotter.enable_point_picking(
            callback=self._on_pick, show_point=True,
            show_message="点击 3D 场景选取坐标 | Ctrl=障碍物  Shift=途经点",
            font_size=11,
        )

    def _grid_ref(self):
        g = self.pv.ImageData(dimensions=(3, 3, 1), spacing=(25, 25, 1))
        self.plotter.add_mesh(g, color="#c8d8e0", opacity=0.3, name="grid")

    def _init_empty_wave(self):
        """初始化一个平坦的水面网格（后续加载数据后替换为动态波浪）"""
        xx, yy = np.meshgrid(np.linspace(0, self.nx, 40), np.linspace(0, self.ny, 40))
        zz = np.zeros_like(xx)
        grid = self.pv.StructuredGrid(xx, yy, zz)
        self.wave_mesh = self.plotter.add_mesh(
            grid, color=WATER_COLOR, opacity=WATER_ALPHA,
            name="wave", show_edges=False, smooth_shading=True,
        )

    def load(self, grid):
        self.grid = grid
        self.nx, self.ny = grid.nx, grid.ny
        self.path3d = None
        self.wave_time = 0.0
        self.plotter.clear()
        self._draw()
        self._start_waves()
        self.plotter.reset_camera()
        self.panel.refresh()

    def _start_waves(self):
        """启动波浪动画"""
        if self.wave_running:
            return
        self.wave_running = True
        self._animate_wave()

    def _animate_wave(self):
        """每 80ms 更新一次波浪"""
        if not self.wave_running or self.grid is None:
            return
        self.wave_time += 0.08
        self._update_wave_surface()
        # 通过 Tkinter after 调度下次更新
        if self.panel.root:
            self.panel.root.after(80, self._animate_wave)

    def _update_wave_surface(self):
        """更新水面网格顶点以模拟波浪"""
        if self.grid is None:
            return
        res = max(2, min(self.nx, self.ny) // 2)
        xx, yy = np.meshgrid(
            np.linspace(0, self.nx, res * 2),
            np.linspace(0, self.ny, res * 2),
        )
        t = self.wave_time
        zz = (0.12 * np.sin(xx * 0.6 + t * 2.5) * np.cos(yy * 0.5 + t * 1.8)
              + 0.08 * np.sin(xx * 0.3 - t * 1.3) * np.sin(yy * 0.7 + t * 2.1)
              + 0.05 * np.cos((xx + yy) * 0.4 + t * 3.0))
        grid = self.pv.StructuredGrid(xx, yy, zz)
        self.plotter.add_mesh(
            grid, color=WATER_COLOR, opacity=WATER_ALPHA,
            name="wave", show_edges=False, smooth_shading=True,
        )

    def _draw(self):
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz

        # 海底地形
        dv = np.full((ny, nx), np.nan, dtype=np.float32)
        for y in range(ny):
            for x in range(nx):
                if g.depth[y, x] > 0:
                    dv[y, x] = -(g.depth[y, x] / max(1, g.depth.max())) * nz * 0.8
        xx, yy = np.meshgrid(range(nx), range(ny))
        self.plotter.add_mesh(
            self.pv.StructuredGrid(xx, yy, dv),
            color=SEABED_COLOR, opacity=0.5, name="seabed", show_edges=False,
        )

        # 障碍物
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if g.obstacles[z, y, x]:
                        self.plotter.add_mesh(
                            self.pv.Cube(center=(x, y, -z - 0.5),
                                          x_length=0.85, y_length=0.85, z_length=0.85),
                            color=OBS_COLOR, opacity=0.9, name=f"obs{x}.{y}.{z}",
                        )

        # 水流箭头
        step = max(1, min(nx, ny) // 10)
        for y in range(0, ny, step):
            for x in range(0, nx, step):
                if g.depth[y, x] <= 0:
                    continue
                (dx, dy, _), spd = g.get_current_at(x, y, 0)
                if spd > 0.01:
                    a = self.pv.Arrow(
                        start=(x, y, 0.3), direction=(dx * spd * 3, dy * spd * 3, 0),
                        tip_length=0.25, tip_radius=0.08, shaft_radius=0.03, scale=1.0,
                    )
                    self.plotter.add_mesh(a, color=CURRENT_COLOR, opacity=0.55)

        # 起点
        if g.mission_start:
            s = g.mission_start
            b = self.pv.Sphere(center=(s[0], s[1], -s[2]), radius=0.5)
            self.plotter.add_mesh(b, color=START_COLOR, pbr=True, name="start")
            self.plotter.add_point_labels(
                [b.center], ["起点"], font_size=11, text_color=START_COLOR, point_size=1,
            )

        # 途经点
        for i, wp in enumerate(g.mission_waypoints):
            b = self.pv.Sphere(center=(wp[0], wp[1], -wp[2]), radius=0.3)
            self.plotter.add_mesh(b, color=WP_COLOR, name=f"wp{i}")

        # 终点
        if g.mission_end and g.mission_end != g.mission_start:
            e = g.mission_end
            b = self.pv.Sphere(center=(e[0], e[1], -e[2]), radius=0.45)
            self.plotter.add_mesh(b, color=END_COLOR, pbr=True, name="end")

        # 自主规划
        if g.mission_start and g.mission_waypoints:
            self._plan()

    def _plan(self):
        if not self.grid or not self.grid.mission_start or not self.grid.mission_waypoints:
            return
        self.path3d = plan_tsp_3d(
            self.grid, self.grid.mission_start,
            self.grid.mission_waypoints, self.grid.mission_end,
        )
        if self.path3d is None:
            return
        if self.path_actor:
            self.plotter.remove_actor(self.path_actor)
        pts = np.array([[p[0], p[1], -p[2]] for p in self.path3d], dtype=np.float64)
        if len(pts) >= 2:
            tube = self.pv.Spline(pts, n_points=len(pts) * 3).tube(radius=0.12)
            self.path_actor = self.plotter.add_mesh(
                tube, color=PATH_COLOR, pbr=True, metallic=0.1, name="path",
            )
        d, f, dc = compute_3d_path_cost(self.grid, self.path3d)
        self.panel.show_result(d, f, dc)

    def _on_pick(self, point):
        if point is None or self.grid is None:
            return
        try:
            pt = point.points[0] if hasattr(point, "points") else point
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            z = -int(round(float(pt[2])))
        except (TypeError, IndexError, ValueError):
            return
        if not (0 <= x < self.grid.nx and 0 <= y < self.grid.ny and 0 <= z < self.grid.nz):
            return

        if self.panel.ctrl.get():
            self.grid.obstacles[z, y, x] = not self.grid.obstacles[z, y, x]
        elif self.panel.shift.get():
            self.grid.mission_waypoints.append((x, y, z))
        else:
            return

        self.plotter.clear()
        self._draw()
        self.panel.refresh()

    def show(self):
        self.plotter.show()

    def stop_waves(self):
        self.wave_running = False


# ═══════════════════════════════════════════════════════════
# 控制面板
# ═══════════════════════════════════════════════════════════

class Panel:
    def __init__(self, root):
        self.root = root
        self.viewer = None
        self.ctrl = tk.BooleanVar(value=False)
        self.shift = tk.BooleanVar(value=False)

        style = ttk.Style()
        style.configure("TFrame", background=BG_UI)
        style.configure("TLabelframe", background=BG_UI, foreground=FG_TITLE)
        style.configure("TLabelframe.Label", background=BG_UI, foreground=FG_TITLE, font=FONT_BOLD)
        style.configure("TLabel", background=BG_UI, foreground=FG_TEXT, font=FONT_UI)
        style.configure("TButton", font=FONT_UI)

        f = ttk.Frame(root, padding=8)
        f.pack(fill=tk.Y, side=tk.RIGHT, padx=2, pady=2)
        w = 22

        # 数据
        g1 = ttk.LabelFrame(f, text="数据加载", padding=6)
        g1.pack(fill=tk.X, pady=3)
        ttk.Button(g1, text="打开 JSON 文件...", width=w, command=self._load_json).pack(fill=tk.X, pady=2)
        ttk.Button(g1, text="演示: 沿海水域", width=w, command=self._load_coastal).pack(fill=tk.X, pady=1)
        ttk.Button(g1, text="演示: 内河航道", width=w, command=self._load_river).pack(fill=tk.X, pady=1)

        # 信息
        g2 = ttk.LabelFrame(f, text="场景信息", padding=6)
        g2.pack(fill=tk.X, pady=3)
        self.txt_info = tk.Text(g2, height=4, width=w, bg="#ffffff", fg=FG_TEXT,
                                 font=FONT_MONO, relief=tk.FLAT, borderwidth=1,
                                 wrap=tk.WORD, state=tk.DISABLED)
        self.txt_info.pack(fill=tk.X)

        # 结果
        g3 = ttk.LabelFrame(f, text="路径规划结果", padding=6)
        g3.pack(fill=tk.X, pady=3)
        self.lbl_d = ttk.Label(g3, text="总距离: --", font=FONT_MONO, foreground="#cc5500")
        self.lbl_d.pack(anchor=tk.W)
        self.lbl_f = ttk.Label(g3, text="水流代价: --", font=FONT_MONO, foreground="#3377aa")
        self.lbl_f.pack(anchor=tk.W)
        self.lbl_z = ttk.Label(g3, text="深度代价: --", font=FONT_MONO, foreground="#558844")
        self.lbl_z.pack(anchor=tk.W)

        # 操作
        g4 = ttk.LabelFrame(f, text="操作", padding=6)
        g4.pack(fill=tk.X, pady=3)
        ttk.Button(g4, text="重新规划路径", width=w, command=self._replan).pack(fill=tk.X, pady=2)
        ttk.Button(g4, text="清除途经点", width=w, command=self._clear_wp).pack(fill=tk.X, pady=1)
        ttk.Button(g4, text="清除障碍物", width=w, command=self._clear_obs).pack(fill=tk.X, pady=1)

        # 提示
        g5 = ttk.LabelFrame(f, text="操作提示", padding=6)
        g5.pack(fill=tk.X, pady=3)
        ttk.Label(g5, text="Ctrl + 点击 = 放置/移除障碍物\nShift + 点击 = 添加途经点\n左键拖拽 = 旋转 | 右键拖拽 = 平移\n滚轮 = 缩放",
                  font=("Microsoft YaHei", 7), foreground="#778888").pack(anchor=tk.W)

    def _load_json(self):
        p = filedialog.askopenfilename(
            title="选择水况数据 JSON 文件",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if p:
            try:
                self.viewer.load(Water3DGrid.from_json(p))
            except Exception as e:
                messagebox.showerror("加载失败", str(e))

    def _load_coastal(self):
        self.viewer.load(demo_3d_coastal())
    def _load_river(self):
        self.viewer.load(demo_3d_river())
    def _clear_wp(self):
        if self.viewer.grid:
            self.viewer.grid.mission_waypoints = []
            self.viewer.plotter.clear()
            self.viewer._draw()
    def _clear_obs(self):
        if self.viewer.grid:
            self.viewer.grid.obstacles[:] = False
            self.viewer.plotter.clear()
            self.viewer._draw()
    def _replan(self):
        if self.viewer.grid:
            self.viewer._plan()

    def refresh(self):
        g = self.viewer.grid
        if g is None:
            return
        n_obs = int(np.sum(g.obstacles))
        t = (
            f"网格: {g.nx} x {g.ny} x {g.nz}\n"
            f"精度: {g.resolution:.0f} m/格\n"
            f"障碍物: {n_obs}  途经点: {len(g.mission_waypoints)}\n"
            f"起点: {g.mission_start or '未设定'}"
        )
        self.txt_info.config(state=tk.NORMAL)
        self.txt_info.delete("1.0", tk.END)
        self.txt_info.insert("1.0", t)
        self.txt_info.config(state=tk.DISABLED)

    def show_result(self, d, f, dc):
        self.lbl_d.config(text=f"总距离: {d:,.0f} m")
        self.lbl_f.config(text=f"水流代价: {f:,.0f}")
        self.lbl_z.config(text=f"深度代价: {dc:,.0f}")


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.title("水域机器人 3D 智能决策平台")
    root.geometry("1100x700")
    root.configure(bg=BG_UI)

    ttk.Label(
        root, text="水域机器人 3D 智能决策平台",
        font=("Microsoft YaHei", 14, "bold"),
        foreground=FG_TITLE, background=BG_UI,
    ).pack(pady=8)

    panel = Panel(root)
    viewer = Scene3D(panel)
    panel.viewer = viewer

    def _down(e):
        if "Control" in e.keysym:
            panel.ctrl.set(True)
        elif "Shift" in e.keysym:
            panel.shift.set(True)

    def _up(e):
        if "Control" in e.keysym:
            panel.ctrl.set(False)
        elif "Shift" in e.keysym:
            panel.shift.set(False)

    root.bind("<KeyPress>", _down)
    root.bind("<KeyRelease>", _up)

    def _on_close():
        viewer.stop_waves()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    viewer.plotter.show(title="水域机器人 3D 智能决策平台", window_size=[820, 640])
    root.mainloop()


if __name__ == "__main__":
    main()
