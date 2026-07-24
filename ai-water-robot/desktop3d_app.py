"""
水域机器人 3D 智能决策平台

功能: 导入水况数据 → 自主路径规划 → 3D 交互式可视化
操作: 鼠标拖拽旋转/缩放 | Ctrl+点击切换障碍物 | Shift+点击添加途经点
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.water_3d import Water3DGrid, demo_3d_coastal, demo_3d_river
from planning.astar3d import plan_tsp_3d, compute_3d_path_cost
from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

FONT = ("Microsoft YaHei", 9)
FONT_BOLD = ("Microsoft YaHei", 9, "bold")
FONT_MONO = ("Consolas", 10)
BG = "#0f1923"
FG = "#c8d8e0"


class Scene3D:
    """3D 场景渲染与交互"""

    def __init__(self, panel):
        import pyvista as pv
        self.pv = pv
        self.panel = panel
        self.grid = None
        self.path3d = None
        self.path_actor = None

        self.plotter = pv.Plotter(window_size=(820, 640))
        self.plotter.set_background(BG)
        self.plotter.add_axes(color="#335555", xlabel="东 X", ylabel="北 Y", zlabel="深 Z")

        self.plotter.add_text(
            "导入水况数据或选择演示场景开始",
            position="upper_left", font_size=11, color="#889999",
        )
        self._grid_ref()

        self.plotter.enable_point_picking(
            callback=self._on_pick, show_point=True,
            show_message="点击 3D 场景选取坐标 | Ctrl=障碍物  Shift=途经点",
            font_size=11,
        )

    def _grid_ref(self):
        g = self.pv.ImageData(dimensions=(3, 3, 1), spacing=(25, 25, 1))
        self.plotter.add_mesh(g, color="#152530", opacity=0.3, name="grid")

    def load(self, grid):
        self.grid = grid
        self.path3d = None
        self.plotter.clear()
        self._draw()
        self.plotter.reset_camera()
        self.panel.refresh()

    def _draw(self):
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz

        # 水面
        surf = self.pv.Plane(center=(nx / 2, ny / 2, 0), direction=(0, 0, 1),
                              i_size=nx, j_size=ny, i_resolution=2, j_resolution=2)
        self.plotter.add_mesh(surf, color="#1a7799", opacity=0.22, name="水面")

        # 地形
        dv = np.full((ny, nx), np.nan, dtype=np.float32)
        for y in range(ny):
            for x in range(nx):
                if g.depth[y, x] > 0:
                    dv[y, x] = -(g.depth[y, x] / max(1, g.depth.max())) * nz * 0.8
        xx, yy = np.meshgrid(range(nx), range(ny))
        self.plotter.add_mesh(
            self.pv.StructuredGrid(xx, yy, dv),
            color="#2a6640", opacity=0.45, name="地形", show_edges=False,
        )

        # 障碍物
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if g.obstacles[z, y, x]:
                        self.plotter.add_mesh(
                            self.pv.Cube(center=(x, y, -z - 0.5),
                                          x_length=0.8, y_length=0.8, z_length=0.8),
                            color="#d03030", opacity=0.85, name=f"obs{x}.{y}.{z}",
                        )

        # 水流
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
                    self.plotter.add_mesh(a, color="#0099cc", opacity=0.45)

        # 起点 / 途经点 / 终点
        if g.mission_start:
            s = g.mission_start
            b = self.pv.Sphere(center=(s[0], s[1], -s[2]), radius=0.5)
            self.plotter.add_mesh(b, color="#00dd77", pbr=True, name="起点")
            self.plotter.add_point_labels([b.center], ["起点"], font_size=11,
                                           text_color="#00dd77", point_size=1)
        for i, wp in enumerate(g.mission_waypoints):
            b = self.pv.Sphere(center=(wp[0], wp[1], -wp[2]), radius=0.3)
            self.plotter.add_mesh(b, color="#eeaa22", name=f"途经点{i}")
        if g.mission_end and g.mission_end != g.mission_start:
            e = g.mission_end
            b = self.pv.Sphere(center=(e[0], e[1], -e[2]), radius=0.45)
            self.plotter.add_mesh(b, color="#ee4444", pbr=True, name="终点")

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
                tube, color="#ffffff", pbr=True, metallic=0.05, name="路径",
            )
        d, f, dc = compute_3d_path_cost(self.grid, self.path3d)
        self.panel.show_result(d, f, dc)

    def _on_pick(self, point):
        if point is None or self.grid is None:
            return
        try:
            pt = point.points[0] if hasattr(point, "points") else point
            x, y, z = int(round(float(pt[0]))), int(round(float(pt[1]))), -int(round(float(pt[2])))
        except (TypeError, IndexError, ValueError):
            return
        if not (0 <= x < self.grid.nx and 0 <= y < self.grid.ny and 0 <= z < self.grid.nz):
            return

        if self.panel.ctrl.get():       # 障碍物
            self.grid.obstacles[z, y, x] = not self.grid.obstacles[z, y, x]
        elif self.panel.shift.get():    # 途经点
            self.grid.mission_waypoints.append((x, y, z))
        else:
            return

        self.plotter.clear()
        self._draw()
        self.panel.refresh()

    def show(self):
        self.plotter.show()


# ═══════════════════════════════════════════════════════════
# 控制面板
# ═══════════════════════════════════════════════════════════

class Panel:
    def __init__(self, root):
        self.root = root
        self.viewer = None
        self.ctrl = tk.BooleanVar(value=False)
        self.shift = tk.BooleanVar(value=False)

        f = ttk.Frame(root, padding=6)
        f.pack(fill=tk.Y, side=tk.RIGHT, padx=2, pady=2)
        w = 22

        # ── 数据 ──
        g1 = ttk.LabelFrame(f, text="数据加载", padding=6)
        g1.pack(fill=tk.X, pady=3)
        ttk.Button(g1, text="打开 JSON 文件...", width=w, command=self._load_json).pack(fill=tk.X, pady=2)
        ttk.Button(g1, text="演示: 沿海水域", width=w, command=self._load_coastal).pack(fill=tk.X, pady=1)
        ttk.Button(g1, text="演示: 内河航道", width=w, command=self._load_river).pack(fill=tk.X, pady=1)

        # ── 场景信息 ──
        g2 = ttk.LabelFrame(f, text="场景信息", padding=6)
        g2.pack(fill=tk.X, pady=3)
        self.lbl_info = tk.Text(g2, height=4, width=w, bg="#0d1620", fg="#9ab8b0",
                                 font=FONT_MONO, relief=tk.FLAT, borderwidth=0,
                                 wrap=tk.WORD, state=tk.DISABLED)
        self.lbl_info.pack(fill=tk.X)

        # ── 路径结果 ──
        g3 = ttk.LabelFrame(f, text="路径规划结果", padding=6)
        g3.pack(fill=tk.X, pady=3)
        self.lbl_dist = ttk.Label(g3, text="总距离: --", font=FONT_MONO, foreground="#ffcc44")
        self.lbl_dist.pack(anchor=tk.W)
        self.lbl_flow = ttk.Label(g3, text="水流代价: --", font=FONT_MONO, foreground="#44bbee")
        self.lbl_flow.pack(anchor=tk.W)
        self.lbl_depth = ttk.Label(g3, text="深度代价: --", font=FONT_MONO, foreground="#66cc88")
        self.lbl_depth.pack(anchor=tk.W)

        # ── 操作 ──
        g4 = ttk.LabelFrame(f, text="操作", padding=6)
        g4.pack(fill=tk.X, pady=3)
        ttk.Button(g4, text="重新规划路径", width=w, command=self._replan).pack(fill=tk.X, pady=2)
        ttk.Button(g4, text="清除途经点", width=w, command=self._clear_wp).pack(fill=tk.X, pady=1)
        ttk.Button(g4, text="清除障碍物", width=w, command=self._clear_obs).pack(fill=tk.X, pady=1)

        # ── 操作提示 ──
        g5 = ttk.LabelFrame(f, text="操作提示", padding=6)
        g5.pack(fill=tk.X, pady=3)
        tips = (
            "Ctrl + 点击 = 放置/移除障碍物\n"
            "Shift + 点击 = 添加途经点\n"
            "左键拖拽 = 旋转 | 右键拖拽 = 平移\n"
            "滚轮 = 缩放"
        )
        ttk.Label(g5, text=tips, font=("Microsoft YaHei", 7),
                  foreground="#667777").pack(anchor=tk.W)

    def _load_json(self):
        p = filedialog.askopenfilename(
            title="选择水况数据 JSON 文件",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not p:
            return
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
        text = (
            f"网格: {g.nx} x {g.ny} x {g.nz}\n"
            f"精度: {g.resolution:.0f} m/格\n"
            f"障碍物: {n_obs}  途经点: {len(g.mission_waypoints)}\n"
            f"起点: {g.mission_start or '未设定'}"
        )
        self.lbl_info.config(state=tk.NORMAL)
        self.lbl_info.delete("1.0", tk.END)
        self.lbl_info.insert("1.0", text)
        self.lbl_info.config(state=tk.DISABLED)

    def show_result(self, dist, flow, dc):
        self.lbl_dist.config(text=f"总距离: {dist:,.0f} m")
        self.lbl_flow.config(text=f"水流代价: {flow:,.0f}")
        self.lbl_depth.config(text=f"深度代价: {dc:,.0f}")


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.title("水域机器人 3D 智能决策平台")
    root.geometry("1100x700")
    root.configure(bg=BG)

    ttk.Label(
        root, text="水域机器人 3D 智能决策平台",
        font=("Microsoft YaHei", 13, "bold"),
        foreground="#c8d8e0", background=BG,
    ).pack(pady=6)

    panel = Panel(root)
    viewer = Scene3D(panel)
    panel.viewer = viewer

    # 键盘修饰键状态
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

    viewer.plotter.show(title="水域机器人 3D 智能决策平台", window_size=[820, 640])
    root.mainloop()


if __name__ == "__main__":
    main()
