"""
PyVista 3D 交互式水上机器人决策平台

功能:
  - 3D 场景: 水面/地形/障碍物/水流/路径
  - 交互: 鼠标旋转/缩放, Ctrl+点击放障碍物, Shift+点击放途经点
  - 自主决策: 加载水况后自动规划, 修改场景即时重规划
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


# ═══════════════════════════════════════════════════════════
# PyVista 3D 场景
# ═══════════════════════════════════════════════════════════

class Water3DViewer:
    """PyVista 3D 水体场景渲染与交互"""

    def __init__(self, control_panel):
        import pyvista as pv
        self.pv = pv
        self.control = control_panel
        self.grid: Water3DGrid = None
        self.path3d = None
        self.obs_actors = []
        self.path_actor = None
        self.wp_actors = []

        # 创建 PyVista Plotter 并嵌入 Tkinter
        self.plotter = pv.Plotter(window_size=(800, 600))
        self.plotter.set_background("#0a1628")
        self.plotter.add_axes(color="#446666")

        # 初始文字
        self.plotter.add_text(
            "PyVista 3D Water Robot Platform\n"
            "Load JSON data or select Demo to start",
            position="upper_left", font_size=10, color="#aabbcc",
        )
        self._add_empty_scene()

        # 键盘事件: Ctrl=障碍物, Shift=途经点
        self.plotter.enable_surface_picking(
            callback=self._on_pick, show_point=True,
            pickable_window=False,  # only fire when Ctrl/Shift held
        )

    def _add_empty_scene(self):
        """空场景的参考网格"""
        g = self.pv.ImageData(dimensions=(3, 3, 1), spacing=(25, 25, 1))
        self.plotter.add_mesh(g, color="#1a3a4a", opacity=0.3, name="ref_grid")

    def load_data(self, grid: Water3DGrid):
        """加载 Water3DGrid 数据并渲染完整场景"""
        self.grid = grid
        self.path3d = None
        self.plotter.clear()
        self._render_scene()
        self.plotter.reset_camera()
        self.control.update_info()

    def _render_scene(self):
        """渲染完整 3D 场景"""
        g = self.grid
        nx, ny, nz = g.nx, g.ny, g.nz

        # ── 1. 水面（蓝色半透明平面）──
        surf = self.pv.Plane(
            center=(nx / 2, ny / 2, 0),
            direction=(0, 0, 1),
            i_size=nx, j_size=ny,
            i_resolution=2, j_resolution=2,
        )
        self.plotter.add_mesh(surf, color="#1a88aa", opacity=0.25,
                               name="surface", show_edges=False)

        # ── 2. 地形曲面 ──
        depth_vis = np.zeros((ny, nx))
        for y in range(ny):
            for x in range(nx):
                d = g.depth[y, x]
                if d > 0:
                    depth_vis[y, x] = -(d / g.resolution) * nz / 10
                else:
                    depth_vis[y, x] = np.nan
        xx, yy = np.meshgrid(range(nx), range(ny))
        terrain = self.pv.StructuredGrid(xx, yy, depth_vis)
        self.plotter.add_mesh(terrain, color="#2a6a3a", opacity=0.5,
                               name="terrain", show_edges=False)

        # ── 3. 障碍物（红色立方体）──
        self.obs_actors = []
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    if g.obstacles[z, y, x]:
                        cube = self.pv.Cube(
                            center=(x, y, -z - 0.5),
                            x_length=0.8, y_length=0.8, z_length=0.8,
                        )
                        actor = self.plotter.add_mesh(
                            cube, color="#cc2222", opacity=0.85,
                            name=f"obs_{x}_{y}_{z}",
                        )
                        self.obs_actors.append(actor)

        # ── 4. 水流箭头 ──
        step = max(1, min(nx, ny) // 8)
        arrow_pts, arrow_vecs = [], []
        for y in range(0, ny, step):
            for x in range(0, nx, step):
                if g.depth[y, x] <= 0:
                    continue
                (dx, dy, _), speed = g.get_current_at(x, y, 0)
                if speed > 0.01:
                    arrow_pts.append([x, y, 0.3])
                    arrow_vecs.append([dx * speed * 3, dy * speed * 3, 0])
        if arrow_pts:
            arrows = self.pv.Arrow()
            for pt, vec in zip(arrow_pts, arrow_vecs):
                arrow = self.pv.Arrow(
                    start=pt, direction=vec,
                    tip_length=0.3, tip_radius=0.1, shaft_radius=0.04,
                    scale=1.0,
                )
                self.plotter.add_mesh(arrow, color="#00aadd", opacity=0.5)

        # ── 5. 任务标记 ──
        if g.mission_start:
            s = g.mission_start
            ball_s = self.pv.Sphere(center=(s[0], s[1], -s[2]), radius=0.5)
            self.plotter.add_mesh(ball_s, color="#00ff88", pbr=True,
                                   metallic=0.3, roughness=0.4, name="start")
            self.plotter.add_point_labels(
                [ball_s.center], ["START"], font_size=12,
                text_color="#00ff88", point_size=1,
            )
        if g.mission_end and g.mission_end != g.mission_start:
            e = g.mission_end
            ball_e = self.pv.Sphere(center=(e[0], e[1], -e[2]), radius=0.5)
            self.plotter.add_mesh(ball_e, color="#ff4444", pbr=True, name="end")

        # 途经点
        self.wp_actors = []
        for i, wp in enumerate(g.mission_waypoints):
            ball = self.pv.Sphere(center=(wp[0], wp[1], -wp[2]), radius=0.35)
            self.plotter.add_mesh(ball, color="#ffaa00", name=f"wp_{i}")
            self.wp_actors.append(ball)

        # ── 6. 自主规划路径 ──
        if g.mission_start and g.mission_waypoints:
            self._auto_plan()

    def _auto_plan(self):
        """自动规划路径并渲染"""
        if self.grid is None:
            return
        g = self.grid
        if not g.mission_start or not g.mission_waypoints:
            return

        self.path3d = plan_tsp_3d(
            g, g.mission_start, g.mission_waypoints, g.mission_end,
        )
        if self.path3d is None:
            return

        # 移除旧路径
        if self.path_actor:
            self.plotter.remove_actor(self.path_actor)

        # 路径管线
        pts = np.array([[p[0], p[1], -p[2]] for p in self.path3d], dtype=np.float64)
        if len(pts) < 2:
            return
        spline = self.pv.Spline(pts, n_points=len(pts) * 3)
        tube = spline.tube(radius=0.15)
        self.path_actor = self.plotter.add_mesh(
            tube, color="#ffffff", pbr=True, metallic=0.1, name="path",
        )

        dist, flow, dc = compute_3d_path_cost(g, self.path3d)
        self.control.update_results(dist, flow, dc)

    def _on_pick(self, point):
        """处理 3D 点击"""
        if point is None:
            return
        x, y, z = int(round(point[0])), int(round(point[1])), -int(round(point[2]))

        if not (0 <= x < self.grid.nx and 0 <= y < self.grid.ny):
            return
        if z < 0 or z >= self.grid.nz:
            return

        # Ctrl+点击: 添加/移除障碍物
        if self.control.ctrl_held.get():
            if self.grid.obstacles[z, y, x]:
                self.grid.obstacles[z, y, x] = False
            else:
                self.grid.obstacles[z, y, x] = True
            self.plotter.clear()
            self._render_scene()
            self.control.update_info()

        # Shift+点击: 添加途经点
        elif self.control.shift_held.get():
            self.grid.mission_waypoints.append((x, y, z))
            self.plotter.clear()
            self._render_scene()
            self.control.update_info()

    def show(self):
        self.plotter.show()


# ═══════════════════════════════════════════════════════════
# 控制面板 (Tkinter)
# ═══════════════════════════════════════════════════════════

class ControlPanel:
    def __init__(self, root: tk.Tk, viewer: Water3DViewer):
        self.root = root
        self.viewer = viewer
        self.ctrl_held = tk.BooleanVar(value=False)
        self.shift_held = tk.BooleanVar(value=False)

        frame = ttk.Frame(root, padding=8)
        frame.pack(fill=tk.Y, side=tk.RIGHT, padx=4, pady=4)

        # ── 数据加载 ──
        data_frame = ttk.LabelFrame(frame, text="Water Data", padding=6)
        data_frame.pack(fill=tk.X, pady=4)
        ttk.Button(data_frame, text="Load JSON...",
                   command=self._on_load_json).pack(fill=tk.X, pady=1)
        ttk.Button(data_frame, text="Demo: Coastal",
                   command=self._on_load_coastal).pack(fill=tk.X, pady=1)
        ttk.Button(data_frame, text="Demo: River",
                   command=self._on_load_river).pack(fill=tk.X, pady=1)

        # ── 交互模式 ──
        interact_frame = ttk.LabelFrame(frame, text="3D Interaction", padding=6)
        interact_frame.pack(fill=tk.X, pady=4)
        ttk.Label(interact_frame, text="Hold Ctrl + Click: Toggle obstacle",
                  font=("", 8)).pack(anchor=tk.W)
        ttk.Label(interact_frame, text="Hold Shift + Click: Add waypoint",
                  font=("", 8)).pack(anchor=tk.W)
        ttk.Label(interact_frame, text="Drag: Rotate | Right-drag: Pan | Scroll: Zoom",
                  font=("", 7), foreground="#888").pack(anchor=tk.W, pady=4)

        # ── 任务编辑 ──
        task_frame = ttk.LabelFrame(frame, text="Mission", padding=6)
        task_frame.pack(fill=tk.X, pady=4)
        ttk.Button(task_frame, text="Clear Waypoints",
                   command=self._clear_waypoints).pack(fill=tk.X, pady=1)
        ttk.Button(task_frame, text="Clear Obstacles",
                   command=self._clear_obstacles).pack(fill=tk.X, pady=1)

        # ── 操作 ──
        action_frame = ttk.LabelFrame(frame, text="Actions", padding=6)
        action_frame.pack(fill=tk.X, pady=4)
        ttk.Button(action_frame, text="Re-Plan Path",
                   command=self._replan).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="Reset View",
                   command=self._reset_view).pack(fill=tk.X, pady=1)

        # ── 信息 ──
        info_frame = ttk.LabelFrame(frame, text="Data Info", padding=6)
        info_frame.pack(fill=tk.X, pady=4)
        self.lbl_grid_info = ttk.Label(info_frame, text="No data loaded",
                                        font=("Consolas", 9), foreground="#aaccbb")
        self.lbl_grid_info.pack(anchor=tk.W)

        result_frame = ttk.LabelFrame(frame, text="Path Result", padding=6)
        result_frame.pack(fill=tk.X, pady=4)
        self.lbl_dist = ttk.Label(result_frame, text="Distance: --",
                                   font=("Consolas", 10), foreground="#ffcc00")
        self.lbl_dist.pack(anchor=tk.W)
        self.lbl_flow = ttk.Label(result_frame, text="Flow cost: --",
                                   font=("Consolas", 10), foreground="#00ccff")
        self.lbl_flow.pack(anchor=tk.W)
        self.lbl_depth = ttk.Label(result_frame, text="Depth cost: --",
                                    font=("Consolas", 10), foreground="#88cc88")
        self.lbl_depth.pack(anchor=tk.W)

        # ── 键盘提示 ──
        hint_frame = ttk.LabelFrame(frame, text="Keyboard", padding=4)
        hint_frame.pack(fill=tk.X, pady=4)
        ttk.Label(hint_frame, text="Ctrl=Obstacle  Shift=Waypoint\n"
                                    "R=Replan  C=Clear view",
                  font=("", 7), foreground="#888").pack(anchor=tk.W)

    def _on_load_json(self):
        path = filedialog.askopenfilename(
            title="Select water data JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            grid = Water3DGrid.from_json(path)
            self.viewer.load_data(grid)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _on_load_coastal(self):
        self.viewer.load_data(demo_3d_coastal())

    def _on_load_river(self):
        self.viewer.load_data(demo_3d_river())

    def _clear_waypoints(self):
        if self.viewer.grid:
            self.viewer.grid.mission_waypoints = []
            self.viewer.plotter.clear()
            self.viewer._render_scene()

    def _clear_obstacles(self):
        if self.viewer.grid:
            self.viewer.grid.obstacles[:] = False
            self.viewer.plotter.clear()
            self.viewer._render_scene()

    def _replan(self):
        if self.viewer.grid:
            self.viewer._auto_plan()

    def _reset_view(self):
        self.viewer.plotter.reset_camera()

    def update_info(self):
        g = self.viewer.grid
        if g is None:
            return
        n_obs = int(np.sum(g.obstacles))
        self.lbl_grid_info.config(
            text=f"Grid: {g.nx}x{g.ny}x{g.nz} @ {g.resolution}m\n"
                 f"Obstacles: {n_obs} | Waypoints: {len(g.mission_waypoints)}\n"
                 f"Start: {g.mission_start or 'N/A'}"
        )

    def update_results(self, dist, flow, dc):
        self.lbl_dist.config(text=f"Distance: {dist:.0f} m")
        self.lbl_flow.config(text=f"Flow cost: {flow:.0f}")
        self.lbl_depth.config(text=f"Depth cost: {dc:.0f}")


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.title("3D Water Robot Intelligence Platform")
    root.geometry("1100x700")
    root.configure(bg="#0a1628")

    # 标题
    header = ttk.Label(
        root,
        text="3D Water Robot Intelligence Platform  |  PyVista + A*3D",
        font=("Consolas", 12, "bold"),
        foreground="#c0d8e0", background="#0a1628",
    )
    header.pack(pady=4)

    # 先创建控制面板
    viewer_holder = {"viewer": None}
    panel = ControlPanel(root, viewer_holder)

    # 创建 3D 视图（嵌入 Tkinter）
    viewer = Water3DViewer(panel)
    viewer_holder["viewer"] = viewer
    panel.viewer = viewer

    # 绑定全局键盘
    def on_key(event):
        if event.keysym == "Control_L" or event.keysym == "Control_R":
            panel.ctrl_held.set(True)
        elif event.keysym == "Shift_L" or event.keysym == "Shift_R":
            panel.shift_held.set(True)

    def on_key_release(event):
        if event.keysym == "Control_L" or event.keysym == "Control_R":
            panel.ctrl_held.set(False)
        elif event.keysym == "Shift_L" or event.keysym == "Shift_R":
            panel.shift_held.set(False)

    root.bind("<KeyPress>", on_key)
    root.bind("<KeyRelease>", on_key_release)

    # 启动 PyVista 的 Tkinter 嵌入
    viewer.plotter.show(title="3D Water Robot", window_size=[800, 650])

    root.mainloop()


if __name__ == "__main__":
    main()
