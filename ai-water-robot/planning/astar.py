"""
A* 路径规划模块

在 2D 网格上执行 A* 搜索，支持：
- 8 方向移动（含对角线，代价 √2）
- 障碍物规避
- 多点巡回路径规划（按最近邻贪心排序）
"""

import heapq
from typing import List, Tuple, Optional, Set
import numpy as np


def astar(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allow_diagonal: bool = True,
) -> Optional[List[Tuple[int, int]]]:
    """
    A* 寻路算法

    Args:
        grid: 2D numpy 数组，0=可通行，1=障碍物
        start: 起点坐标 (row, col)
        goal: 终点坐标 (row, col)
        allow_diagonal: 是否允许对角线移动

    Returns:
        路径点列表 [(row, col), ...]，包含起点和终点；若无可行路径则返回 None
    """
    rows, cols = grid.shape

    # 边界检查
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
        return None
    if grid[start[0], start[1]] == 1:
        return None
    if grid[goal[0], goal[1]] == 1:
        return None

    # 方向和代价
    if allow_diagonal:
        directions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),       # 四方向
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),  # 对角线
        ]
    else:
        directions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        ]

    def heuristic(pos: Tuple[int, int]) -> float:
        """启发式函数：欧几里得距离"""
        return ((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2) ** 0.5

    # open_set: (f_score, tiebreaker, position)
    open_set = [(heuristic(start), 0, start)]
    came_from: dict = {}
    g_score = {start: 0.0}
    closed_set: Set[Tuple[int, int]] = set()
    tiebreaker = 1

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal:
            # 回溯路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc, move_cost in directions:
            neighbor = (current[0] + dr, current[1] + dc)

            # 边界检查
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            # 障碍物检查
            if grid[neighbor[0], neighbor[1]] == 1:
                continue
            # 已处理
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                tiebreaker += 1
                heapq.heappush(open_set, (f_score, tiebreaker, neighbor))

    return None  # 无可行路径


def plan_multi_point_route(
    obstacle_grid: np.ndarray,
    start: Tuple[int, int],
    waypoints: List[Tuple[int, int]],
    allow_diagonal: bool = True,
) -> Optional[List[Tuple[int, int]]]:
    """
    多点巡回路径规划

    使用最近邻贪心策略：从起点出发，依次前往最近的目标点。
    不是 TSP 最优解，但对 Demo 场景足够用。

    Args:
        obstacle_grid: 障碍物网格（0=可通行，1=障碍物）
        start: 起点坐标
        waypoints: 需要经过的目标点列表
        allow_diagonal: 是否允许对角线移动

    Returns:
        完整路径点列表（从起点到最后一个目标点）；若任何一段无路径则返回 None
    """
    if not waypoints:
        return None

    full_path = [start]
    remaining = list(waypoints)
    current = start

    while remaining:
        # 找最近的目标点
        nearest_idx = 0
        nearest_dist = float("inf")
        for i, wp in enumerate(remaining):
            dist = ((current[0] - wp[0]) ** 2 + (current[1] - wp[1]) ** 2) ** 0.5
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = i

        target = remaining.pop(nearest_idx)

        # A* 寻路到目标
        segment = astar(obstacle_grid, current, target, allow_diagonal)
        if segment is None:
            return None  # 某段无可行路径

        # 合并路径（跳过重复的起点）
        full_path.extend(segment[1:])
        current = target

    return full_path


def compute_path_length(path: List[Tuple[int, int]]) -> float:
    """计算路径总长度（考虑对角线代价）"""
    if not path or len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        dr = abs(path[i][0] - path[i + 1][0])
        dc = abs(path[i][1] - path[i + 1][1])
        if dr > 0 and dc > 0:
            total += 1.414  # 对角线
        else:
            total += 1.0  # 直走
    return round(total, 2)


def path_to_directions(path: List[Tuple[int, int]]) -> List[str]:
    """
    将路径点序列转为方向序列

    Returns:
        方向字符列表: '↑' '↓' '←' '→' '↖' '↗' '↙' '↘'
    """
    directions = []
    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        if dr == -1 and dc == 0:
            directions.append("↑")
        elif dr == 1 and dc == 0:
            directions.append("↓")
        elif dr == 0 and dc == -1:
            directions.append("←")
        elif dr == 0 and dc == 1:
            directions.append("→")
        elif dr == -1 and dc == -1:
            directions.append("↖")
        elif dr == -1 and dc == 1:
            directions.append("↗")
        elif dr == 1 and dc == -1:
            directions.append("↙")
        elif dr == 1 and dc == 1:
            directions.append("↘")
    return directions
