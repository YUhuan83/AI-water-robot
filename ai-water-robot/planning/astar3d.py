"""
A* 3D 路径规划 — 26 方向 + 水流感知代价

代价组成:
    1. 基础移动距离（欧几里得）
    2. 水流惩罚（逆流代价高，顺流代价低）
    3. 深度变化代价（上浮/下潜耗能）
    4. 浅水奖励（水面通信好、光照足）
"""

import heapq
from typing import List, Tuple, Optional, Set
import numpy as np

# ═══════════════════════════════════════════════════════════
# 代价权重（可调参数）
# ═══════════════════════════════════════════════════════════

FLOW_PENALTY_WEIGHT = 2.0     # 水流惩罚系数（越高越倾向于顺流）
DEPTH_CHANGE_WEIGHT = 0.5     # 深度变化代价系数
SHALLOW_BONUS_WEIGHT = 0.3    # 浅水奖励系数

# ═══════════════════════════════════════════════════════════
# 26 方向移动向量
# ═══════════════════════════════════════════════════════════

# 6 面方向 (face-connected)
_FACE = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
# 12 边方向 (edge-connected)
_EDGE = [
    (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
    (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
    (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
]
# 8 角方向 (corner-connected)
_CORNER = [
    (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
    (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
]

ALL_26_DIRECTIONS = [
    (dx, dy, dz, np.sqrt(dx * dx + dy * dy + dz * dz))
    for dx, dy, dz in _FACE + _EDGE + _CORNER
]


def astar3d(
    grid,  # Water3DGrid
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
) -> Optional[List[Tuple[int, int, int]]]:
    """
    3D A* 路径规划，考虑水流和深度

    Args:
        grid: Water3DGrid 实例
        start: 起点 (x, y, z)
        goal: 终点 (x, y, z)

    Returns:
        3D 路径点序列 [(x, y, z), ...]，或 None
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    if not grid.is_passable(*start):
        return None
    if not grid.is_passable(*goal):
        return None

    def heuristic(pos: Tuple[int, int, int]) -> float:
        return np.sqrt(
            (pos[0] - goal[0]) ** 2
            + (pos[1] - goal[1]) ** 2
            + (pos[2] - goal[2]) ** 2
        )

    open_set = [(heuristic(start), 0, start)]
    came_from: dict = {}
    g_score = {start: 0.0}
    closed: Set[Tuple[int, int, int]] = set()
    tiebreaker = 1

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        cx, cy, cz = current
        _, cur_flow_speed = grid.get_current_at(cx, cy, cz)

        for dx, dy, dz, base_cost in ALL_26_DIRECTIONS:
            nx_, ny_, nz_ = cx + dx, cy + dy, cz + dz
            neighbor = (nx_, ny_, nz_)

            if not (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz):
                continue
            if not grid.is_passable(nx_, ny_, nz_):
                continue
            if neighbor in closed:
                continue

            # ── 移动代价计算 ──
            move_cost = base_cost * grid.resolution

            # 水流代价：逆流加罚、顺流减罚
            cur_vec, cur_spd = grid.get_current_at(cx, cy, cz)
            if cur_spd > 0.001 and base_cost > 0.001:
                move_vec = (dx / base_cost, dy / base_cost, dz / base_cost)
                alignment = (
                    move_vec[0] * cur_vec[0]
                    + move_vec[1] * cur_vec[1]
                    + move_vec[2] * cur_vec[2]
                )
                alignment = max(-1.0, min(1.0, alignment))
                flow_penalty = FLOW_PENALTY_WEIGHT * cur_spd * (1.0 - alignment)
                move_cost += flow_penalty * grid.resolution

            # 深度变化代价
            if dz != 0:
                move_cost += DEPTH_CHANGE_WEIGHT * abs(dz) * grid.resolution

            # 浅水奖励（z 越小越靠近水面，通信/光照好）
            depth_frac = nz_ / max(1, nz - 1)
            move_cost -= SHALLOW_BONUS_WEIGHT * (1.0 - depth_frac) * grid.resolution

            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                tiebreaker += 1
                heapq.heappush(
                    open_set,
                    (tentative_g + heuristic(neighbor), tiebreaker, neighbor),
                )

    return None


def plan_tsp_3d(
    grid,
    start: Tuple[int, int, int],
    waypoints: List[Tuple[int, int, int]],
    end: Optional[Tuple[int, int, int]] = None,
) -> Optional[List[Tuple[int, int, int]]]:
    """
    3D 多点巡回路径规划（nearest-neighbor + A*3D）

    Args:
        grid: Water3DGrid
        start: 起点
        waypoints: 途经点列表
        end: 终点（None 则结束在最后一个途经点）

    Returns:
        完整 3D 路径，或 None
    """
    if not waypoints:
        return None

    full_path = []
    remaining = list(waypoints)
    current = start

    while remaining:
        # nearest-neighbor
        nearest_idx = min(
            range(len(remaining)),
            key=lambda i: np.sqrt(
                (current[0] - remaining[i][0]) ** 2
                + (current[1] - remaining[i][1]) ** 2
                + (current[2] - remaining[i][2]) ** 2
            ),
        )
        target = remaining.pop(nearest_idx)
        segment = astar3d(grid, current, target)
        if segment is None:
            return None
        if full_path:
            full_path.extend(segment[1:])
        else:
            full_path = segment
        current = target

    if end is not None and end != current:
        segment = astar3d(grid, current, end)
        if segment is None:
            return None
        full_path.extend(segment[1:])

    return full_path


def compute_3d_path_cost(
    grid, path: List[Tuple[int, int, int]]
) -> Tuple[float, float, float]:
    """
    计算 3D 路径的详细代价

    Returns:
        (总距离_米, 水流代价, 深度变化)
    """
    total_dist = 0.0
    total_flow = 0.0
    total_depth = 0.0

    for i in range(len(path) - 1):
        x1, y1, z1 = path[i]
        x2, y2, z2 = path[i + 1]
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        base = np.sqrt(dx * dx + dy * dy + dz * dz) * grid.resolution
        total_dist += base

        cur_vec, cur_spd = grid.get_current_at(x1, y1, z1)
        if cur_spd > 0.001 and base > 0.001:
            move_vec = (dx / max(0.001, base / grid.resolution),
                         dy / max(0.001, base / grid.resolution),
                         dz / max(0.001, base / grid.resolution))
            alignment = (
                move_vec[0] * cur_vec[0]
                + move_vec[1] * cur_vec[1]
                + move_vec[2] * cur_vec[2]
            )
            total_flow += FLOW_PENALTY_WEIGHT * cur_spd * (1.0 - alignment) * grid.resolution

        total_depth += DEPTH_CHANGE_WEIGHT * abs(dz) * grid.resolution

    return round(total_dist, 1), round(total_flow, 1), round(total_depth, 1)
