"""
任务模式生成器 — 生成多样化的水域机器人任务途经点

模式:
  patrol     — 平行线巡逻 (lawnmower), 覆盖矩形区域
  spiral     — 螺旋搜索, 从中心向外扩展
  zigzag     — 之字形河道巡检
  scattered  — 随机散点, 分布在可航行区域
  perimeter  — 边界环绕巡检
  cluster    — 多簇分布, 模拟多个巡检目标群
"""

import math
import random
from typing import List, Tuple, Optional


def generate_patrol_waypoints(
    grid,
    x1: int, y1: int, x2: int, y2: int,
    z: int = 0,
    spacing: int = 4,
) -> List[Tuple[int, int, int]]:
    """
    平行线巡逻 (lawnmower pattern)
    在矩形区域内生成来回扫描线

    ┌──────────┐
    │ →→→→→→→ │
    │ ←←←←←←← │
    │ →→→→→→→ │
    └──────────┘
    """
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(grid.nx - 1, x2); y2 = min(grid.ny - 1, y2)

    waypoints = []
    direction = 1  # 1=向右, -1=向左
    for row_y in range(y1, y2 + 1, spacing):
        if direction == 1:
            waypoints.append((x1, row_y, z))
            if row_y + spacing <= y2:
                # 需要折返时加转角点
                pass
            waypoints.append((x2, row_y, z))
        else:
            waypoints.append((x2, row_y, z))
            waypoints.append((x1, row_y, z))
        direction *= -1

    # 去重连续相同点
    result = [waypoints[0]]
    for wp in waypoints[1:]:
        if wp != result[-1]:
            result.append(wp)
    return result


def generate_spiral_waypoints(
    grid,
    cx: int, cy: int,
    z: int = 0,
    radius: int = 10,
    rings: int = 3,
    points_per_ring: int = 8,
) -> List[Tuple[int, int, int]]:
    """
    螺旋搜索模式 — 从中心向外逐圈扩展
    """
    waypoints = []
    for r in range(1, rings + 1):
        r_actual = radius * r / rings
        for i in range(points_per_ring):
            angle = 2 * math.pi * i / points_per_ring + (r * 0.3)  # 每圈偏移角度
            x = int(cx + r_actual * math.cos(angle))
            y = int(cy + r_actual * math.sin(angle))
            x = max(0, min(grid.nx - 1, x))
            y = max(0, min(grid.ny - 1, y))
            if grid.depth[y, x] > 0:
                waypoints.append((x, y, z))
    return waypoints


def generate_zigzag_waypoints(
    grid,
    x1: int, y1: int, x2: int, y2: int,
    z: int = 0,
    segments: int = 5,
    amplitude: int = 3,
) -> List[Tuple[int, int, int]]:
    """
    之字形河道巡检 — 在两点间生成S型弯曲路径途经点
    """
    waypoints = []
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)

    if length < 1:
        return [(x1, y1, z)]

    # 主方向单位向量
    ux, uy = dx / length, dy / length
    # 垂直方向
    px, py = -uy, ux

    for i in range(segments + 1):
        t = i / segments
        base_x = x1 + dx * t
        base_y = y1 + dy * t

        # S形偏移
        offset = amplitude * math.sin(t * math.pi * 2)
        x = int(base_x + px * offset)
        y = int(base_y + py * offset)
        x = max(0, min(grid.nx - 1, x))
        y = max(0, min(grid.ny - 1, y))
        if grid.depth[y, x] > 0:
            waypoints.append((x, y, z))

    return waypoints


def generate_scattered_waypoints(
    grid,
    count: int = 6,
    z: int = 0,
    min_spacing: int = 3,
) -> List[Tuple[int, int, int]]:
    """
    随机散点模式 — 在可航行水域随机生成途经点
    """
    # 收集可航行位置
    passable = []
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] > 0 and not grid.obstacles[z, y, x]:
                passable.append((x, y, z))

    if not passable:
        return []

    waypoints = []
    attempts = 0
    while len(waypoints) < count and attempts < count * 20:
        attempts += 1
        wp = random.choice(passable)
        # 确保与已有途经点保持最小间距
        too_close = any(
            math.sqrt((wp[0] - w[0])**2 + (wp[1] - w[1])**2) < min_spacing
            for w in waypoints
        )
        if not too_close:
            waypoints.append(wp)

    return waypoints


def generate_cluster_waypoints(
    grid,
    cluster_centers: List[Tuple[int, int, int]],
    points_per_cluster: int = 3,
    cluster_radius: int = 3,
) -> List[Tuple[int, int, int]]:
    """
    多簇分布 — 围绕多个巡检中心生成途经点群
    模拟: 检查多个浮标群/风机群/养殖网箱群
    """
    waypoints = []
    for cx, cy, cz in cluster_centers:
        for _ in range(points_per_cluster):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(1, cluster_radius)
            x = int(cx + dist * math.cos(angle))
            y = int(cy + dist * math.sin(angle))
            x = max(0, min(grid.nx - 1, x))
            y = max(0, min(grid.ny - 1, y))
            if grid.depth[y, x] > 0 and not grid.obstacles[cz, y, x]:
                waypoints.append((x, y, cz))
    return waypoints


def generate_perimeter_waypoints(
    grid,
    margin: int = 2,
    z: int = 0,
    step: int = 5,
) -> List[Tuple[int, int, int]]:
    """
    边界环绕巡检 — 沿可航行区域边界一周
    """
    nx, ny = grid.nx, grid.ny
    waypoints = []

    # 找到可航行区域的边界框
    valid_y, valid_x = (grid.depth > 0).nonzero()
    if len(valid_y) == 0:
        return []

    min_x = max(margin, int(valid_x.min()))
    max_x = min(nx - 1 - margin, int(valid_x.max()))
    min_y = max(margin, int(valid_y.min()))
    max_y = min(ny - 1 - margin, int(valid_y.max()))

    # 顶边 →
    for x in range(min_x, max_x + 1, step):
        if grid.depth[min_y, x] > 0:
            waypoints.append((x, min_y, z))
    # 右边 ↓
    for y in range(min_y, max_y + 1, step):
        if grid.depth[y, max_x] > 0:
            waypoints.append((max_x, y, z))
    # 底边 ←
    for x in range(max_x, min_x - 1, -step):
        if grid.depth[max_y, x] > 0:
            waypoints.append((x, max_y, z))
    # 左边 ↑
    for y in range(max_y, min_y - 1, -step):
        if grid.depth[y, min_x] > 0:
            waypoints.append((min_x, y, z))

    return waypoints


# ═══════════════════ 便捷入口 ═══════════════════

MISSION_PATTERNS = {
    "patrol":    "平行线巡逻 — 矩形区域来回扫描",
    "spiral":    "螺旋搜索 — 从中心向外逐圈扩展",
    "zigzag":    "之字形巡检 — S形弯曲河道巡检",
    "scattered": "随机散点 — 可航行水域随机分布",
    "cluster":   "多簇分布 — 围绕目标群巡检",
    "perimeter": "边界环绕 — 沿水域边界一周",
}


def generate_mission(
    grid,
    pattern: str = "scattered",
    count: int = 6,
    z: int = 0,
) -> Tuple[List[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
    """
    根据模式生成途经点列表和推荐终点

    Returns:
        (waypoints, suggested_end) — 途经点列表和建议终点
    """
    nx, ny = grid.nx, grid.ny
    margin = 3
    cx, cy = nx // 2, ny // 2

    if pattern == "patrol":
        wps = generate_patrol_waypoints(grid, margin, margin, nx - margin, ny - margin, z, spacing=4)
        end = wps[-1] if wps else None
    elif pattern == "spiral":
        wps = generate_spiral_waypoints(grid, cx, cy, z, radius=min(nx, ny)//2, rings=3, points_per_ring=count)
        end = wps[0] if wps else None  # 回到中心
    elif pattern == "zigzag":
        wps = generate_zigzag_waypoints(grid, margin, ny//2, nx - margin, ny//2, z, segments=count, amplitude=4)
        end = wps[-1] if wps else None
    elif pattern == "cluster":
        # 自动生成3个聚类中心
        centers = [
            (nx//4, ny//4, z),
            (3*nx//4, ny//2, z),
            (nx//2, 3*ny//4, z),
        ]
        wps = generate_cluster_waypoints(grid, centers, points_per_cluster=max(2, count//3), cluster_radius=3)
        end = None  # 不设固定终点
    elif pattern == "perimeter":
        wps = generate_perimeter_waypoints(grid, margin=margin, z=z, step=4)
        end = wps[0] if wps else None  # 闭环
    else:  # scattered (default)
        wps = generate_scattered_waypoints(grid, count=count, z=z, min_spacing=3)
        end = None

    # 如果途经点过多，限制数量
    if len(wps) > count * 2:
        step = len(wps) // count
        wps = wps[::max(1, step)][:count]

    return wps, end
