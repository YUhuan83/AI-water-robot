"""
水域智能感知与任务理解系统 — Tkinter 桌面版

无需浏览器，直接运行。左键点击网格放置物体，右键清除。
复用全部后端模块：water_grid, astar, llm_planner, renderer
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.water_grid import (
    WaterGrid, EditableScene, render_grid_as_image,
    OBSTACLE, BUOY, TRASH, ROBOT, WATER,
    demo_scene_trash_cleanup, demo_scene_buoy_inspection, demo_scene_patrol,
)
from planning.astar import plan_tsp_route, compute_path_length, astar
from task_planner.llm_planner import rule_based_plan
from visualization.renderer import render_static_path
from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 颜色常量 ──
COLORS = {
    WATER: "#0d4f6b",
    OBSTACLE: "#5c3a1e",
    BUOY: "#00cc66",
    TRASH: "#ff9900",
    ROBOT: "#00ccff",
}
CELL_SIZE = 28
GRID_PAD = 2
CANVAS_SIZE = CELL_SIZE * 20 + GRID_PAD * 2


class DesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("水域智能感知与任务理解系统")
        self.root.configure(bg="#0a1628")

        self.scene = EditableScene(20)
        self.brush = tk.StringVar(value="obstacle")
        self.task_instruction = tk.StringVar(
            value="从起点出发，经过所有途经点收集垃圾，然后回到码头"
        )
        self.use_preset = tk.BooleanVar(value=False)
        self.preset_name = tk.StringVar(value="垃圾清理")

        self._build_ui()
        self._draw_grid()

    # ═══════════════════════════════════════════════════════
    # UI 构建
    # ═══════════════════════════════════════════════════════

    def _build_ui(self):
        # 主容器
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── 左侧：网格画布 ──
        left_frame = ttk.Frame(main)
        main.add(left_frame, weight=3)

        self.canvas = tk.Canvas(
            left_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="#0d4f6b", highlightthickness=1,
            highlightbackground="#1a6a8a", cursor="crosshair",
        )
        self.canvas.pack(padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)

        # 图例
        legend_frame = ttk.Frame(left_frame)
        legend_frame.pack(pady=4)
        for label, color in [("障碍物", COLORS[OBSTACLE]), ("起点", COLORS[ROBOT]),
                              ("途经点", COLORS[BUOY]), ("目标", COLORS[TRASH])]:
            f = tk.Frame(legend_frame, bg=color, width=14, height=14)
            f.pack(side=tk.LEFT, padx=(0, 2))
            ttk.Label(legend_frame, text=label, foreground="#aabbcc",
                      background="#0a1628").pack(side=tk.LEFT, padx=(0, 10))

        # ── 右侧：控制面板 ──
        right_frame = ttk.Frame(main)
        main.add(right_frame, weight=2)

        # 画笔模式
        brush_frame = ttk.LabelFrame(right_frame, text="画笔模式", padding=6)
        brush_frame.pack(fill=tk.X, pady=4)
        modes = [
            ("障碍物 obstacle", "obstacle"),
            ("起点 start", "start"),
            ("途经点 waypoint", "waypoint"),
            ("目标 trash", "trash"),
            ("清除 clear", "clear"),
        ]
        for text, value in modes:
            ttk.Radiobutton(
                brush_frame, text=text, variable=self.brush,
                value=value,
            ).pack(anchor=tk.W)

        # 场景
        scene_frame = ttk.LabelFrame(right_frame, text="场景", padding=6)
        scene_frame.pack(fill=tk.X, pady=4)

        ttk.Checkbutton(
            scene_frame, text="使用预设场景", variable=self.use_preset,
            command=self._on_preset_toggle,
        ).pack(anchor=tk.W)

        preset_combo = ttk.Combobox(
            scene_frame, textvariable=self.preset_name,
            values=["垃圾清理", "浮标巡检", "水域巡逻"],
            state="readonly",
        )
        preset_combo.pack(fill=tk.X, pady=2)
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_load)

        btn_row = ttk.Frame(scene_frame)
        btn_row.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row, text="重置场景", command=self._on_reset).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="加载预设", command=self._on_preset_load).pack(
            side=tk.LEFT, padx=2)

        # 任务指令
        task_frame = ttk.LabelFrame(right_frame, text="任务指令", padding=6)
        task_frame.pack(fill=tk.X, pady=4)
        ttk.Entry(task_frame, textvariable=self.task_instruction).pack(
            fill=tk.X)

        ttk.Button(
            right_frame, text="执行任务",
            command=self._on_execute,
        ).pack(fill=tk.X, pady=6)

        # ── 结果展示 ──
        result_frame = ttk.LabelFrame(right_frame, text="结果", padding=4)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: 任务规划
        plan_tab = ttk.Frame(self.notebook)
        self.notebook.add(plan_tab, text="任务规划")
        self.plan_text = tk.Text(plan_tab, bg="#0d1628", fg="#c0d8d0",
                                  font=("Microsoft YaHei", 9),
                                  wrap=tk.WORD, relief=tk.FLAT)
        self.plan_text.pack(fill=tk.BOTH, expand=True)

        # Tab 2: 路径图
        path_tab = ttk.Frame(self.notebook)
        self.notebook.add(path_tab, text="路径图")
        self.path_label = ttk.Label(path_tab)
        self.path_label.pack(fill=tk.BOTH, expand=True)

        # Tab 3: 动画
        anim_tab = ttk.Frame(self.notebook)
        self.notebook.add(anim_tab, text="动画")
        self.anim_label = ttk.Label(anim_tab)
        self.anim_label.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status = ttk.Label(
            self.root, text="左键放置 | 右键清除 | 就绪",
            relief=tk.SUNKEN, anchor=tk.W,
        )
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ═══════════════════════════════════════════════════════
    # 网格绘制
    # ═══════════════════════════════════════════════════════

    def _draw_grid(self):
        self.canvas.delete("all")
        size = self.scene.size

        # 网格线
        for i in range(size + 1):
            x = GRID_PAD + i * CELL_SIZE
            self.canvas.create_line(
                x, GRID_PAD, x, GRID_PAD + size * CELL_SIZE,
                fill="#1a6a8a", width=1,
            )
            y = GRID_PAD + i * CELL_SIZE
            self.canvas.create_line(
                GRID_PAD, y, GRID_PAD + size * CELL_SIZE, y,
                fill="#1a6a8a", width=1,
            )

        # 物体
        for r in range(size):
            for c in range(size):
                val = self.scene.grid[r, c]
                if val == WATER:
                    continue
                self._draw_cell(r, c, val)

        # 起点特殊标记
        if self.scene.robot_pos:
            r, c = self.scene.robot_pos
            self._draw_start(r, c)

        # 途经点编号
        for i, (r, c) in enumerate(self.scene.waypoints):
            x = GRID_PAD + c * CELL_SIZE + CELL_SIZE // 2
            y = GRID_PAD + r * CELL_SIZE + CELL_SIZE // 2
            self.canvas.create_text(
                x, y, text=str(i + 1), fill="white",
                font=("Microsoft YaHei", 8, "bold"),
            )

        # 状态更新
        n_obs = int(self.scene.grid.sum())
        n_wp = len(self.scene.waypoints)
        n_tr = len(self.scene.trash_positions)
        self.status.config(
            text=f"障碍物:{n_obs} 途经点:{n_wp} 目标:{n_tr} "
                 f"起点:{self.scene.robot_pos or '未设'} | "
                 f"左键放置 | 右键清除"
        )

    def _draw_cell(self, r, c, val):
        x1 = GRID_PAD + c * CELL_SIZE + 2
        y1 = GRID_PAD + r * CELL_SIZE + 2
        x2 = x1 + CELL_SIZE - 4
        y2 = y1 + CELL_SIZE - 4
        color = COLORS.get(val, "#fff")
        if val == OBSTACLE:
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color,
                                          outline="#3a1a0a", width=1)
        elif val in (BUOY, TRASH):
            self.canvas.create_oval(x1, y1, x2, y2, fill=color,
                                     outline="#fff", width=1)

    def _draw_start(self, r, c):
        cx = GRID_PAD + c * CELL_SIZE + CELL_SIZE // 2
        cy = GRID_PAD + r * CELL_SIZE + CELL_SIZE // 2
        r_ = CELL_SIZE // 2 - 3
        self.canvas.create_oval(cx - r_, cy - r_, cx + r_, cy + r_,
                                 fill=COLORS[ROBOT], outline="#fff", width=2)
        self.canvas.create_text(cx, cy, text="S", fill="white",
                                 font=("Microsoft YaHei", 9, "bold"))

    # ═══════════════════════════════════════════════════════
    # 交互事件
    # ═══════════════════════════════════════════════════════

    def _to_grid(self, event):
        """像素坐标 → 网格坐标"""
        col = (event.x - GRID_PAD) // CELL_SIZE
        row = (event.y - GRID_PAD) // CELL_SIZE
        if 0 <= row < self.scene.size and 0 <= col < self.scene.size:
            return row, col
        return None

    def _on_left_click(self, event):
        pos = self._to_grid(event)
        if pos is None:
            return
        row, col = pos
        self.scene.set_cell(row, col, self.brush.get())
        self._draw_grid()

    def _on_right_click(self, event):
        pos = self._to_grid(event)
        if pos is None:
            return
        row, col = pos
        self.scene.set_cell(row, col, "clear")
        self._draw_grid()

    def _on_reset(self):
        self.scene = EditableScene(20)
        self.use_preset.set(False)
        self._draw_grid()

    def _on_preset_toggle(self):
        if self.use_preset.get():
            self._on_preset_load()

    def _on_preset_load(self, event=None):
        if not self.use_preset.get():
            return
        name = self.preset_name.get()
        preset_map = {
            "垃圾清理": demo_scene_trash_cleanup,
            "浮标巡检": demo_scene_buoy_inspection,
            "水域巡逻": demo_scene_patrol,
        }
        func = preset_map.get(name)
        if func:
            wg = func(20)
            # 转换为 EditableScene
            self.scene = EditableScene(20)
            for (r, c), tname in wg.objects.items():
                if tname == "障碍物":
                    self.scene.set_cell(r, c, "obstacle")
                elif tname == "浮标":
                    self.scene.set_cell(r, c, "waypoint")
                elif tname == "垃圾":
                    self.scene.set_cell(r, c, "trash")
                elif tname == "机器人":
                    self.scene.set_cell(r, c, "start")
            self._draw_grid()

    # ═══════════════════════════════════════════════════════
    # 执行任务
    # ═══════════════════════════════════════════════════════

    def _on_execute(self):
        self.status.config(text="正在执行...")
        self.root.update()

        try:
            wg = self.scene.to_water_grid()
            obs = wg.get_obstacle_grid()
            start = wg.robot_pos or (0, 0)
            dock = wg.dock_pos

            targets = self.scene.get_targets()
            if not targets:
                messagebox.showwarning("警告", "请先放置至少一个途经点或目标!")
                self.status.config(text="就绪")
                return

            # 可达性检查
            unreachable = [t for t in targets if astar(obs, start, t) is None]
            if unreachable:
                ts = ", ".join([str(t) for t in unreachable[:5]])
                messagebox.showerror(
                    "无法完成任务",
                    f"以下目标被障碍物阻挡:\n{ts}\n\n请移除阻挡的障碍物后重试"
                )
                self.status.config(text=f"不可达目标: {len(unreachable)} 个")
                return

            # TSP 路径规划
            path = plan_tsp_route(obs, start, targets, end=dock)
            if path is None:
                messagebox.showerror(
                    "无法完成任务",
                    "障碍物阻挡导致无法规划完整路径"
                )
                self.status.config(text="路径规划失败")
                return

            dist = compute_path_length(path)

            # ── 任务规划 ──
            instruction = self.task_instruction.get()
            plan = rule_based_plan(instruction)
            plan_display = f"指令: {instruction}\n\n"
            plan_display += f"概述: {plan.get('summary', '')}\n\n"
            for i, t in enumerate(plan.get("tasks", []), 1):
                plan_display += (
                    f"  [{i}] {t.get('action', '?')} -> "
                    f"{t.get('target', '?')}\n"
                    f"      {t.get('reason', '')}\n\n"
                )
            plan_display += (
                f"---\n路径: {len(path)}步, 距离={dist}\n"
                f"起点: {start}  终点: {dock or start}\n"
                f"经过目标: {len(targets)}个"
            )

            self.plan_text.delete("1.0", tk.END)
            self.plan_text.insert("1.0", plan_display)
            self.notebook.select(0)

            # ── 路径图 ──
            path_img_path = os.path.join(OUTPUT_DIR, "desktop_path.png")
            render_static_path(wg, path, path_img_path,
                               title=f"TSP Path (dist={dist}, steps={len(path)})",
                               show_indices=len(path) > 4)
            self._show_image(path_img_path, self.path_label, max_size=350)

            # ── 动画 GIF ──
            from visualization.renderer import SimulationRenderer
            gif_path = os.path.join(OUTPUT_DIR, "desktop_sim.gif")
            renderer = SimulationRenderer(wg, figsize=(7, 6))
            renderer.render_gif(path, gif_path, fps=8, interval=150,
                                collect_targets=targets)
            self._show_image(gif_path, self.anim_label, max_size=350)

            self.status.config(
                text=f"完成! {len(path)}步, 距离={dist}, "
                     f"经过{len(targets)}个目标"
            )

        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.status.config(text=f"错误: {e}")

    def _show_image(self, path, label, max_size=350):
        """在 Label 中显示图片，等比缩放"""
        if not os.path.exists(path):
            return
        img = Image.open(path)
        w, h = img.size
        ratio = min(max_size / w, max_size / h, 1.0)
        if ratio < 1.0:
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo  # 保持引用


def main():
    root = tk.Tk()
    root.geometry("1100x720")
    try:
        root.iconbitmap(default=None)
    except:
        pass
    app = DesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
