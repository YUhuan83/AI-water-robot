"""
3D 可视化渲染模块 — matplotlib mplot3d

渲染内容:
    - 水体网格（半透明蓝色底面）
    - 障碍物（红色方块）
    - 水流箭头（蓝色，大小表示流速）
    - 路径线（白色渐变 + 起点/途经点/终点标记）
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple, Optional

from config import OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


def render_3d_scene(
    grid,  # Water3DGrid
    path: Optional[List[Tuple[int, int, int]]] = None,
    output_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    view_angle: Tuple[float, float] = (30, -60),
) -> str:
    """
    渲染 3D 水体场景和路径

    Args:
        grid: Water3DGrid 实例
        path: 3D 路径点序列
        output_path: 输出 PNG 路径
        figsize: 图形尺寸
        dpi: 分辨率
        view_angle: (elevation, azimuth) 视角

    Returns:
        输出文件路径
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "scene_3d.png")

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="#0a1628")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d4f6b")

    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # ── 1. 水面（半透明平面） ──
    xx, yy = np.meshgrid(range(nx), range(ny))
    zz = np.zeros_like(xx, dtype=float)
    ax.plot_surface(xx, yy, zz, alpha=0.15, color="#1a8aaa", edgecolor="none")

    # ── 2. 水底地形 ──
    depth_surface = np.zeros((ny, nx))
    for y in range(ny):
        for x in range(nx):
            d = grid.depth[y, x]
            if d > 0:
                depth_surface[y, x] = -(d / grid.resolution) * nz / 10
    ax.plot_surface(xx, yy, depth_surface, alpha=0.3, color="#2a5a3a", edgecolor="none")

    # ── 3. 障碍物（红色立方体） ──
    obs_count = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if grid.obstacles[z, y, x] and obs_count < 200:
                    _draw_cube(ax, x, y, -z, 0.8, color="#8b1a1a", alpha=0.7)
                    obs_count += 1

    # ── 4. 水流箭头（抽样显示） ──
    step = max(1, min(nx, ny) // 8)
    for y in range(0, ny, step):
        for x in range(0, nx, step):
            if grid.depth[y, x] <= 0:
                continue
            (dx, dy, dz), speed = grid.get_current_at(x, y, 0)
            if speed > 0.01:
                arrow_len = speed * 2
                ax.quiver(
                    x, y, 0,
                    dx * arrow_len, dy * arrow_len, dz * arrow_len * 0.5,
                    color="#00ccff", alpha=0.4, linewidth=0.5,
                    arrow_length_ratio=0.3,
                )

    # ── 5. 路径 ──
    if path and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        pz = [-p[2] for p in path]  # 深度向下
        ax.plot(px, py, pz, color="#ffffff", linewidth=2, alpha=0.8, zorder=5)

        # 起点
        ax.scatter(*[px[0]], *[py[0]], *[pz[0]],
                   color="#00ff88", s=80, marker="o", zorder=6, label="Start")
        # 途经点
        for wp in grid.mission_waypoints:
            ax.scatter(*[wp[0]], *[wp[1]], *[-wp[2]],
                       color="#00ccff", s=50, marker="^", zorder=6)
        # 终点
        ax.scatter(*[px[-1]], *[py[-1]], *[pz[-1]],
                   color="#ff4444", s=80, marker="s", zorder=6, label="End")

    # ── 6. 轴标签和视角 ──
    ax.set_xlabel("X (east)", color="#aabbcc", fontsize=9)
    ax.set_ylabel("Y (north)", color="#aabbcc", fontsize=9)
    ax.set_zlabel("Z (depth)", color="#aabbcc", fontsize=9)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(-nz, 1)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    ax.tick_params(colors="#668888", labelsize=7)

    title = "3D Water Robot Path Planning"
    ax.set_title(title, color="#c0d8e0", fontsize=12, pad=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def _draw_cube(ax, x, y, z, size=0.8, color="#ff0000", alpha=0.5):
    """在 3D 坐标处绘制立方体"""
    vertices = [
        [x - size/2, y - size/2, z - size/2],
        [x + size/2, y - size/2, z - size/2],
        [x + size/2, y + size/2, z - size/2],
        [x - size/2, y + size/2, z - size/2],
        [x - size/2, y - size/2, z + size/2],
        [x + size/2, y - size/2, z + size/2],
        [x + size/2, y + size/2, z + size/2],
        [x - size/2, y + size/2, z + size/2],
    ]
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[3], vertices[0], vertices[4], vertices[7]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[3], vertices[2], vertices[1], vertices[0]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                             edgecolor="#440000", linewidth=0.3)
    ax.add_collection3d(poly)
