"""
可视化渲染模块

使用 matplotlib FuncAnimation 生成水面机器人仿真 GIF 动画。
"""

import os
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 GUI 弹出
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from PIL import Image

from environment.water_grid import (
    WATER, OBSTACLE, BUOY, TRASH, ROBOT, TYPE_COLORS,
)

# ── 中文字体设置 ──
def _setup_chinese_font():
    """尝试设置中文字体，如果不可用则回退到英文"""
    # Windows 常见中文字体
    chinese_fonts = [
        "Microsoft YaHei", "SimHei", "SimSun",
        "WenQuanYi Micro Hei", "Noto Sans CJK SC",
        "Source Han Sans SC", "PingFang SC", "Heiti SC",
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font in chinese_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font
    # 无中文字体，使用英文标签
    return None

_CHINESE_FONT = _setup_chinese_font()

# 根据是否有中文字体决定标签语言
if _CHINESE_FONT:
    LBL_START = "起点"
    LBL_END = "终点"
    LBL_TITLE_WATER = "水上机器人任务执行仿真"
    LBL_TITLE_SUB = "Water Robot Mission Simulation"
    LBL_STEP = "步骤"
    LBL_POS = "位置"
    LBL_COLLECTED = "已收集"
    LBL_TARGETS = "个目标"
    LBL_PATH_PREVIEW = "A* 路径规划 (距离={}, 步数={})"
else:
    LBL_START = "Start"
    LBL_END = "Goal"
    LBL_TITLE_WATER = "Water Robot Mission Simulation"
    LBL_TITLE_SUB = ""
    LBL_STEP = "Step"
    LBL_POS = "Pos"
    LBL_COLLECTED = "Collected"
    LBL_TARGETS = "targets"
    LBL_PATH_PREVIEW = "A* Path (dist={}, steps={})"
from config import GIF_OUTPUT_PATH


class SimulationRenderer:
    """水面机器人仿真动画渲染器"""

    def __init__(
        self,
        grid,
        figsize: Tuple[int, int] = (8, 6),
        dpi: int = 100,
    ):
        """
        Args:
            grid: WaterGrid 实例
            figsize: 图形尺寸（英寸）
            dpi: 分辨率
        """
        self.grid = grid
        self.grid_size = grid.size
        self.figsize = figsize
        self.dpi = dpi

        # 颜色映射
        self.colors = {
            WATER: "#0d4f6b",   # 深蓝水面
            OBSTACLE: "#5c3a1e",  # 深棕
            BUOY: "#00cc66",      # 绿
            TRASH: "#ff9900",     # 橙黄
            ROBOT: "#00ccff",     # 亮蓝
        }

        # 已收集的垃圾位置（用于动画中逐步移除）
        self.collected: set = set()

    def render_gif(
        self,
        path: List[Tuple[int, int]],
        output_path: str = GIF_OUTPUT_PATH,
        fps: int = 15,
        interval: int = 100,
        collect_targets: Optional[List[Tuple[int, int]]] = None,
    ) -> str:
        """
        渲染机器人沿路径移动的 GIF 动画

        Args:
            path: 路径点序列 [(row, col), ...]
            output_path: GIF 输出路径
            fps: 每秒帧数
            interval: 帧间延迟（毫秒）
            collect_targets: 需要收集的目标位置列表（机器人到达后消失）

        Returns:
            生成的 GIF 文件路径
        """
        if not path or len(path) < 2:
            raise ValueError("路径至少需要 2 个点才能生成动画")

        self.collected = set()
        collect_set = set(collect_targets or [])

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        fig.patch.set_facecolor("#0a1628")

        # 设置坐标轴
        margin = 1
        ax.set_xlim(-margin, self.grid_size + margin)
        ax.set_ylim(-margin, self.grid_size + margin)
        ax.set_aspect("equal")
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(colors="#1a3a4a", length=0)

        # 反转 y 轴使 (0,0) 在左上角（与网格坐标一致）
        ax.invert_yaxis()

        # 标题
        title_text = f"{LBL_TITLE_WATER}\n{LBL_TITLE_SUB}" if LBL_TITLE_SUB else LBL_TITLE_WATER
        ax.set_title(
            title_text,
            color="#c0d8e0", fontsize=14, fontweight="bold", pad=15,
        )

        # 绘制静态元素
        self._draw_background(ax)
        static_elements = self._draw_static_objects(ax)
        target_markers = self._draw_targets(ax, collect_set)

        # 路径线
        path_line, = ax.plot([], [], "w--", linewidth=1.2, alpha=0.5, zorder=3)
        # 机器人
        robot_circle = Circle(
            (0, 0), 0.35, facecolor=self.colors[ROBOT],
            edgecolor="#ffffff", linewidth=2, zorder=5,
        )
        ax.add_patch(robot_circle)

        # 移动轨迹
        trail_line, = ax.plot([], [], "c-", linewidth=2, alpha=0.6, zorder=4)

        # 状态文本
        status_text = ax.text(
            0.5, 1.02, "",
            transform=ax.transAxes, ha="center", va="bottom",
            color="#aabbcc", fontsize=9,
        )

        # 进度条
        progress_bg = Rectangle((1, -0.6), self.grid_size - 2, 0.3, linewidth=0,
                                facecolor="#1a3a4a", zorder=6, clip_on=False)
        progress_fg = Rectangle((1, -0.6), 0, 0.3, linewidth=0,
                                 facecolor="#00ccff", zorder=7, clip_on=False)
        ax.add_patch(progress_bg)
        ax.add_patch(progress_fg)

        total_frames = len(path)

        def update(frame):
            r, c = path[frame]
            robot_circle.center = (c, r)  # matplotlib 坐标是 (x, y) = (col, row)

            # 更新路径线（已走过的）
            xx = [p[1] for p in path[:frame + 1]]
            yy = [p[0] for p in path[:frame + 1]]
            path_line.set_data(xx, yy)

            # 更新轨迹线（最近 N 步）
            trail_len = 8
            trail_x = [p[1] for p in path[max(0, frame - trail_len):frame + 1]]
            trail_y = [p[0] for p in path[max(0, frame - trail_len):frame + 1]]
            trail_line.set_data(trail_x, trail_y)

            # 检查是否到达收集点
            pos = (r, c)
            if pos in collect_set and pos not in self.collected:
                self.collected.add(pos)
                # 隐藏/淡化标记
                for marker in target_markers:
                    if hasattr(marker, "target_pos") and marker.target_pos == pos:
                        marker.set_alpha(0.2)

            # 更新进度条
            progress = (frame + 1) / total_frames * (self.grid_size - 2)
            progress_fg.set_width(progress)

            # 状态文本
            status_text.set_text(
                f"{LBL_STEP} {frame + 1}/{total_frames}  |  "
                f"{LBL_POS} ({r}, {c})  |  "
                f"{LBL_COLLECTED} {len(self.collected)}/{len(collect_set)} {LBL_TARGETS}"
            )

            return [robot_circle, path_line, trail_line, status_text, progress_fg]

        # 创建动画
        ani = animation.FuncAnimation(
            fig, update, frames=total_frames,
            interval=interval, blit=False, repeat=True,
        )

        # 保存为 GIF
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = animation.PillowWriter(fps=fps)
        ani.save(output_path, writer=writer, dpi=self.dpi)
        plt.close(fig)

        return output_path

    def _draw_background(self, ax):
        """绘制水面背景"""
        # 深色水面
        ax.set_facecolor(self.colors[WATER])

        # 网格线
        for i in range(self.grid_size + 1):
            ax.axhline(i, color="#1a6a8a", linewidth=0.5, alpha=0.3, zorder=1)
            ax.axvline(i, color="#1a6a8a", linewidth=0.5, alpha=0.3, zorder=1)

        # 水面波纹效果（随机浅色横线）
        np.random.seed(42)
        for _ in range(50):
            r = np.random.uniform(0, self.grid_size)
            c = np.random.uniform(0, self.grid_size)
            length = np.random.uniform(0.3, 1.0)
            ax.plot(
                [c, c + length], [r, r],
                color="#2a8aaa", linewidth=0.5, alpha=0.3, zorder=1,
            )

    def _draw_static_objects(self, ax) -> list:
        """绘制静态物体（障碍物、浮标）"""
        elements = []

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                val = self.grid.grid[r, c]
                if val == OBSTACLE:
                    rect = Rectangle(
                        (c - 0.4, r - 0.4), 0.8, 0.8,
                        facecolor=self.colors[OBSTACLE],
                        edgecolor="#3a1a0a", linewidth=1, zorder=2,
                    )
                    ax.add_patch(rect)
                    elements.append(rect)

                elif val == BUOY:
                    buoy = Circle(
                        (c, r), 0.3,
                        facecolor=self.colors[BUOY],
                        edgecolor="#ffffff", linewidth=1, alpha=0.8, zorder=2,
                    )
                    ax.add_patch(buoy)
                    elements.append(buoy)

        return elements

    def _draw_targets(self, ax, collect_set: set) -> list:
        """绘制需要收集的目标（垃圾）"""
        markers = []

        for r, c in collect_set:
            # 使用闪烁效果的小圆 + X 标记
            marker = Circle(
                (c, r), 0.3,
                facecolor=self.colors[TRASH],
                edgecolor="#ff4400", linewidth=1.5, alpha=0.9, zorder=3,
            )
            marker.target_pos = (r, c)
            ax.add_patch(marker)
            markers.append(marker)

            # 小 X 标记
            ax.plot(
                [c - 0.15, c + 0.15], [r - 0.15, r + 0.15],
                color="#ff4400", linewidth=0.8, zorder=3,
            )
            ax.plot(
                [c - 0.15, c + 0.15], [r + 0.15, r - 0.15],
                color="#ff4400", linewidth=0.8, zorder=3,
            )

        return markers


def render_static_path(
    grid,
    path: List[Tuple[int, int]],
    output_path: str = "output/path_preview.png",
    title: str = "Path Preview",
    show_indices: bool = True,
) -> str:
    """
    渲染静态路径规划图（用于 Gradio 路径规划 Tab）

    Args:
        grid: WaterGrid 实例
        path: 路径点序列
        output_path: 输出图片路径
        title: 图片标题
        show_indices: 是否显示路径点编号

    Returns:
        输出文件路径
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    fig.patch.set_facecolor("#0a1628")
    ax.set_facecolor("#0d4f6b")

    # 坐标轴设置
    margin = 1
    ax.set_xlim(-margin, grid.size + margin)
    ax.set_ylim(-margin, grid.size + margin)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(grid.size))
    ax.set_yticks(range(grid.size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_title(title, color="#c0d8e0", fontsize=13, fontweight="bold")

    # 网格线
    for i in range(grid.size + 1):
        ax.axhline(i, color="#1a6a8a", linewidth=0.3, alpha=0.3)
        ax.axvline(i, color="#1a6a8a", linewidth=0.3, alpha=0.3)

    # 障碍物
    for r, c in grid.get_object_positions(OBSTACLE):
        ax.add_patch(Rectangle(
            (c - 0.45, r - 0.45), 0.9, 0.9,
            facecolor="#5c3a1e", edgecolor="#3a1a0a", linewidth=1, zorder=2,
        ))

    # 浮标
    for r, c in grid.get_object_positions(BUOY):
        ax.add_patch(Circle(
            (c, r), 0.3, facecolor="#00cc66",
            edgecolor="#ffffff", linewidth=1, zorder=2,
        ))

    # 垃圾
    for r, c in grid.get_object_positions(TRASH):
        ax.add_patch(Circle(
            (c, r), 0.25, facecolor="#ff9900",
            edgecolor="#ff4400", linewidth=1, zorder=3,
        ))

    # 路径线
    if path and len(path) > 1:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, "w--", linewidth=1.5, alpha=0.6, zorder=4)

        # 起点标记
        ax.plot(path_x[0], path_y[0], "go", markersize=12, zorder=5)
        ax.annotate(LBL_START, (path_x[0], path_y[0]),
                     textcoords="offset points", xytext=(10, -10),
                     color="#00ff88", fontsize=9, fontweight="bold")

        # 终点标记
        ax.plot(path_x[-1], path_y[-1], "ro", markersize=12, zorder=5)
        ax.annotate(LBL_END, (path_x[-1], path_y[-1]),
                     textcoords="offset points", xytext=(10, -10),
                     color="#ff4444", fontsize=9, fontweight="bold")

        # 路径点编号
        if show_indices and len(path) > 4:
            step = max(1, len(path) // 8)
            for i in range(0, len(path), step):
                ax.annotate(
                    str(i), (path_x[i], path_y[i]),
                    color="#ffffff", fontsize=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="#0a1628", alpha=0.7),
                )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path
