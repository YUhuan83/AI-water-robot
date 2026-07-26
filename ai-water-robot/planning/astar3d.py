"""
A* 3D + Dijkstra + 2-opt 路径规划 — 26 方向 + 水流感知代价 + 多策略

代价组成:
    1. 基础移动距离（欧几里得）
    2. 水流惩罚（逆流代价高，顺流代价低）
    3. 深度变化代价（上浮/下潜耗能）
    4. 浅水奖励（水面通信好、光照足）

策略:
    - balanced:  均衡模式，综合所有代价
    - safe:      安全优先，避开高流速区域，保持安全深度
    - fast:      速度优先，尽量顺流，减少绕路
    - energy:    节能优先，避免深度变化，优先浅水
"""

import heapq
import random
import math
import itertools
from typing import List, Tuple, Optional, Set, Dict
from collections import deque
import numpy as np

# ═══════════════════════════════════════════════════════════
# 代价权重（可调参数）
# ═══════════════════════════════════════════════════════════

FLOW_PENALTY_WEIGHT = 2.0     # 水流惩罚系数（越高越倾向于顺流）
DEPTH_CHANGE_WEIGHT = 0.5     # 深度变化代价系数
SHALLOW_BONUS_WEIGHT = 0.3    # 浅水奖励系数

# 多策略权重配置
# obstacle_margin: 安全距离(格), pressure_weight: 水压代价, weather_weight: 天气代价
STRATEGY_WEIGHTS = {
    "balanced": {
        "flow_penalty": 2.0,
        "depth_change": 0.5,
        "shallow_bonus": 0.3,
        "obstacle_margin": 2,         # 2格安全距离
        "obstacle_penalty": 1.0,      # 障碍物惩罚系数
        "pressure_weight": 0.3,       # 水压代价
        "weather_weight": 0.5,        # 天气/风浪代价
    },
    "safe": {
        "flow_penalty": 4.0,          # 强水流惩罚
        "depth_change": 0.2,          # 少深度变化
        "shallow_bonus": 0.6,         # 偏好浅水
        "obstacle_margin": 3,         # 3格安全距离
        "obstacle_penalty": 2.0,      # 强障碍物规避
        "pressure_weight": 0.5,       # 避免高压深水
        "weather_weight": 1.0,        # 强天气规避
    },
    "fast": {
        "flow_penalty": 1.0,          # 弱水流惩罚（顺流加速）
        "depth_change": 0.8,          # 允许深度变化
        "shallow_bonus": 0.1,         # 不关心深度
        "obstacle_margin": 1,         # 最小安全距离
        "obstacle_penalty": 0.5,      # 弱障碍物规避
        "pressure_weight": 0.1,       # 忽略水压
        "weather_weight": 0.3,        # 忽略天气
    },
    "energy": {
        "flow_penalty": 3.0,          # 利用水流
        "depth_change": 1.5,          # 避免深度变化（耗能大）
        "shallow_bonus": 0.8,         # 强偏好浅水（光照充电）
        "obstacle_margin": 2,
        "obstacle_penalty": 1.5,
        "pressure_weight": 1.0,       # 强水压惩罚（深水耗能）
        "weather_weight": 0.8,        # 避开恶劣天气
    },
}

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


def _compute_move_cost(grid, cx, cy, cz, dx, dy, dz, base_cost, sw):
    """计算单步移动代价 — 含水流/水压/天气/漩涡/障碍物"""
    nx_, ny_, nz_ = cx + dx, cy + dy, cz + dz

    move_cost = base_cost * grid.resolution

    # ── 1) 水流代价 (使用合成水流 = 基础流 + 漩涡 + 风生流) ──
    total_vec, total_spd = grid.get_total_current_at(cx, cy, cz)
    if total_spd > 0.001 and base_cost > 0.001:
        move_vec = (dx / base_cost, dy / base_cost, dz / base_cost)
        alignment = (
            move_vec[0] * total_vec[0]
            + move_vec[1] * total_vec[1]
            + move_vec[2] * total_vec[2]
        )
        alignment = max(-1.0, min(1.0, alignment))
        flow_penalty = sw["flow_penalty"] * total_spd * (1.0 - alignment)
        # 逆流额外惩罚 (alignment < 0 即完全逆流时加倍)
        if alignment < 0:
            flow_penalty *= (1.0 + abs(alignment))
        move_cost += flow_penalty * grid.resolution

    # ── 2) 深度变化代价 ──
    if dz != 0:
        move_cost += sw["depth_change"] * abs(dz) * grid.resolution

    # ── 3) 浅水奖励 ──
    depth_frac = nz_ / max(1, grid.nz - 1)
    move_cost -= sw["shallow_bonus"] * (1.0 - depth_frac) * grid.resolution

    # ── 4) 水压代价 (深层 = 高水压 = 耗能增加) ──
    pressure = grid.get_pressure_at(nx_, ny_, nz_)
    # 水压超过200kPa(≈10m深)开始产生代价
    excess_pressure = max(0.0, pressure - 200.0)
    move_cost += sw["pressure_weight"] * excess_pressure * 0.5

    # ── 5) 天气代价 (仅表层z=0受风浪影响) ──
    if nz_ == 0:
        wind_dir, wind_spd, wave_h = grid.get_weather_at(nx_, ny_)
        if wind_spd > 0.5:
            # 逆风移动代价
            if base_cost > 0.001:
                wind_alignment = (
                    (dx / base_cost) * wind_dir[0]
                    + (dy / base_cost) * wind_dir[1]
                )
                wind_penalty = sw["weather_weight"] * wind_spd * (1.0 - wind_alignment) * 0.5
                move_cost += wind_penalty * grid.resolution
            # 大浪额外代价
            if wave_h > 1.0:
                move_cost += sw["weather_weight"] * wave_h * 15.0

    # ── 6) 障碍物接近惩罚 (栅格化检查，距离越近惩罚越大) ──
    margin = sw.get("obstacle_margin", 1)
    obs_penalty = grid.obstacle_proximity_penalty(nx_, ny_, nz_, max_margin=margin + 1)
    move_cost += sw.get("obstacle_penalty", 1.0) * obs_penalty

    # ── 7) 水温代价 (极冷<5°C或极热>30°C增加能耗) ──
    temp = grid.get_temperature_at(nx_, ny_)
    if temp < 5.0:
        move_cost += (5.0 - temp) * 8.0  # 冷水增加电池消耗
    elif temp > 30.0:
        move_cost += (temp - 30.0) * 5.0  # 热水增加冷却需求

    # ── 8) 能见度代价 (低能见度=高风险，安全策略更受影响) ──
    vis = grid.get_visibility_at(nx_, ny_)
    if vis < 5.0:
        # 能见度越低，安全策略代价越大
        vis_penalty = (5.0 - vis) * 12.0
        move_cost += vis_penalty * sw.get("weather_weight", 0.5)

    # ── 9) 潮汐流附加代价 — 涨落潮时流向变化频繁 ──
    tidal_rate = abs(math.sin(grid.tidal_phase * math.pi * 2))
    if tidal_rate > 0.7:  # 急流期
        move_cost += tidal_rate * 8.0 * sw["flow_penalty"]

    return max(0.0, move_cost)


def astar3d(
    grid,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    strategy: str = "balanced",
) -> Optional[List[Tuple[int, int, int]]]:
    """
    3D A* 路径规划，考虑水流和深度

    Args:
        grid: Water3DGrid 实例
        start: 起点 (x, y, z)
        goal: 终点 (x, y, z)
        strategy: 规划策略 (balanced/safe/fast/energy)

    Returns:
        3D 路径点序列 [(x, y, z), ...]，或 None
    """
    sw = STRATEGY_WEIGHTS.get(strategy, STRATEGY_WEIGHTS["balanced"])
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

        for dx, dy, dz, base_cost in ALL_26_DIRECTIONS:
            nx_, ny_, nz_ = cx + dx, cy + dy, cz + dz
            neighbor = (nx_, ny_, nz_)

            if not (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz):
                continue
            if not grid.is_passable(nx_, ny_, nz_):
                continue
            if neighbor in closed:
                continue

            move_cost = _compute_move_cost(grid, cx, cy, cz, dx, dy, dz, base_cost, sw)
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


def dijkstra3d(
    grid,
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
    strategy: str = "balanced",
) -> Optional[List[Tuple[int, int, int]]]:
    """
    3D Dijkstra 路径规划 — 无启发式，保证最优但更慢
    适用于启发式不准确的高水流变化场景
    """
    sw = STRATEGY_WEIGHTS.get(strategy, STRATEGY_WEIGHTS["balanced"])
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    if not grid.is_passable(*start):
        return None
    if not grid.is_passable(*goal):
        return None

    open_set = [(0.0, start)]
    came_from: dict = {}
    dist = {start: 0.0}
    visited: Set[Tuple[int, int, int]] = set()

    while open_set:
        cur_dist, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        cx, cy, cz = current

        for dx, dy, dz, base_cost in ALL_26_DIRECTIONS:
            nx_, ny_, nz_ = cx + dx, cy + dy, cz + dz
            neighbor = (nx_, ny_, nz_)

            if not (0 <= nx_ < nx and 0 <= ny_ < ny and 0 <= nz_ < nz):
                continue
            if not grid.is_passable(nx_, ny_, nz_):
                continue
            if neighbor in visited:
                continue

            move_cost = _compute_move_cost(grid, cx, cy, cz, dx, dy, dz, base_cost, sw)
            new_dist = cur_dist + move_cost

            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                came_from[neighbor] = current
                heapq.heappush(open_set, (new_dist, neighbor))

    return None


def optimize_2opt(
    path: List[Tuple[int, int, int]],
    grid,
    max_iterations: int = 500,
) -> List[Tuple[int, int, int]]:
    """
    2-opt 路径优化 — 消除路径中的交叉和冗余

    对 3D 路径应用 2-opt 局部搜索，减少总距离和水流代价。
    """
    if len(path) < 4:
        return path

    def segment_cost(p1, p2):
        dx, dy, dz = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
        base = np.sqrt(dx * dx + dy * dy + dz * dz) * grid.resolution
        vec, spd = grid.get_current_at(p1[0], p1[1], p1[2])
        if spd > 0.001 and base > 0.001:
            move_vec = (dx / max(0.001, base / grid.resolution),
                         dy / max(0.001, base / grid.resolution),
                         dz / max(0.001, base / grid.resolution))
            alignment = move_vec[0] * vec[0] + move_vec[1] * vec[1] + move_vec[2] * vec[2]
            base += 2.0 * spd * (1.0 - alignment) * grid.resolution
        return base

    best = list(path)
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(best) - 2):
            for j in range(i + 2, len(best) - 1):
                # 检查交换 i-j 是否改善
                old_cost = segment_cost(best[i - 1], best[i]) + segment_cost(best[j], best[j + 1])
                new_cost = segment_cost(best[i - 1], best[j]) + segment_cost(best[i], best[j + 1])

                if new_cost < old_cost * 0.98:  # 2% 改善阈值
                    # 尝试新路径段是否可通行
                    if _is_path_passable(grid, best[i - 1], best[j]) and \
                       _is_path_passable(grid, best[i], best[j + 1]):
                        # 反转 i..j 段
                        best[i:j + 1] = reversed(best[i:j + 1])
                        improved = True
                        break
            if improved:
                break

    return best


def _is_path_passable(grid, p1, p2) -> bool:
    """
    3D DDA (Digital Differential Analyzer) 线遍历 — 逐格检测直线路径是否可通行

    使用曼哈顿步数确保不遗漏任何对角线上的中间格。
    之前用欧几里得距离导致对角线移动时漏检中间格子，偶现障碍物不阻断路线。
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

    # 曼哈顿步数 — 保证每个经过的格子都被检查
    n_steps = abs(dx) + abs(dy) + abs(dz)
    if n_steps == 0:
        return grid.is_passable(x1, y1, z1)

    for step in range(n_steps + 1):
        t = step / n_steps
        # 使用 banker's rounding 修正: 对 .5 情况向上取整，避免遗漏
        x_raw = x1 + dx * t
        y_raw = y1 + dy * t
        z_raw = z1 + dz * t
        # 使用 floor+0.5 方式确保一致的四舍五入 (0.5→1)
        x = int(math.floor(x_raw + 0.5))
        y = int(math.floor(y_raw + 0.5))
        z = int(math.floor(z_raw + 0.5))
        if not grid.is_passable(x, y, z):
            return False
    return True


def smooth_path(
    path: List[Tuple[int, int, int]],
    grid,
    max_iterations: int = 100,
    max_skip: int = 20,
) -> List[Tuple[int, int, int]]:
    """
    路径平滑 — 移除冗余的中间点
    如果两个非相邻点之间可以直线通行，则删除中间所有点。
    max_skip 限制单次最多跳过多少点，防止误删途经点。
    """
    if len(path) < 3:
        return path

    smoothed = list(path)
    changed = True
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        i = 0
        while i < len(smoothed) - 2:
            j = len(smoothed) - 1
            while j > i + 1:
                # 限制单次跳过的点数
                if j - i > max_skip:
                    j -= 1
                    continue
                # 起点和终点相同时不跳过中间点（回路路径）
                if smoothed[i] == smoothed[j]:
                    j -= 1
                    continue
                if _is_path_passable(grid, smoothed[i], smoothed[j]):
                    # 删除 i+1 到 j-1 的点
                    smoothed = smoothed[:i + 1] + smoothed[j:]
                    changed = True
                    break
                j -= 1
            i += 1

    return smoothed


def plan_tsp_3d(
    grid,
    start: Tuple[int, int, int],
    waypoints: List[Tuple[int, int, int]],
    end: Optional[Tuple[int, int, int]] = None,
    strategy: str = "balanced",
    use_2opt: bool = True,
    use_dijkstra_for_short: bool = True,
) -> Optional[List[Tuple[int, int, int]]]:
    """
    3D 多点巡回路径规划（智能途经点排序 + A*3D/Dijkstra + 2-opt + 分段平滑）

    途经点排序策略 (两阶段):
      阶段一: 用欧几里得距离快速评估所有排列，选出最优顺序
      阶段二: 只为最优顺序运行真实 A* 路径规划

    Args:
        grid: Water3DGrid
        start: 起点
        waypoints: 途经点列表
        end: 终点（None 则结束在最后一个途经点）
        strategy: 规划策略 (balanced/safe/fast/energy)
        use_2opt: 是否使用 2-opt 优化
        use_dijkstra_for_short: 短距离段使用 Dijkstra 保证最优

    Returns:
        完整 3D 路径，或 None
    """
    if not waypoints:
        # 无途经点时: start→end 直连
        if end is not None and end != start:
            seg = astar3d(grid, start, end, strategy)
            if seg is None:
                seg = dijkstra3d(grid, start, end, strategy)
            if seg is not None:
                seg = smooth_path(seg, grid, max_iterations=20, max_skip=5)
            return seg
        return None

    wp_list = list(waypoints)

    def euclidean(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def eval_order(order):
        """用欧几里得距离快速评估排列总长度"""
        total = euclidean(start, order[0])
        for i in range(len(order) - 1):
            total += euclidean(order[i], order[i + 1])
        if end is not None:
            total += euclidean(order[-1], end)
        return total

    def plan_segment(a, b):
        """真实 A* 路径规划"""
        dist = euclidean(a, b)
        if use_dijkstra_for_short and dist < 15:
            seg = dijkstra3d(grid, a, b, strategy)
            if seg is None:
                seg = astar3d(grid, a, b, strategy)
        else:
            seg = astar3d(grid, a, b, strategy)
        return seg

    def build_full_path(order):
        """根据途经点顺序构建并平滑完整路径"""
        segments = []
        cur = start
        for wp in order:
            seg = plan_segment(cur, wp)
            if seg is None:
                return None
            seg = smooth_path(seg, grid, max_iterations=20, max_skip=5)
            segments.append(seg)
            cur = wp
        if end is not None and end != cur:
            seg = plan_segment(cur, end)
            if seg is None:
                return None
            seg = smooth_path(seg, grid, max_iterations=20, max_skip=5)
            segments.append(seg)
        full = list(segments[0])
        for seg in segments[1:]:
            full.extend(seg[1:])
        return full

    # ── 阶段一: 快速评估找最优顺序 ──
    n_wp = len(wp_list)
    if n_wp <= 5:
        # ≤5 个途经点: 枚举所有排列 (最多 120 种)，用欧几里得距离快速评估
        best_order = None
        best_eval = float("inf")
        for perm in itertools.permutations(wp_list):
            d = eval_order(list(perm))
            if d < best_eval:
                best_eval = d
                best_order = list(perm)
    else:
        # >5 个: nearest-neighbor + 轮转起始点
        best_order = None
        best_eval = float("inf")
        for _ in range(min(n_wp, 5)):
            order = []
            remaining = list(wp_list)
            cur = start
            while remaining:
                nearest_idx = min(range(len(remaining)),
                    key=lambda i: euclidean(cur, remaining[i]))
                order.append(remaining.pop(nearest_idx))
                cur = order[-1]
            d = eval_order(order)
            if d < best_eval:
                best_eval = d
                best_order = order
            wp_list = wp_list[1:] + wp_list[:1]  # 轮转

    # ── 阶段二: 只为最优顺序构建真实路径 ──
    full_path = build_full_path(best_order)
    if full_path is None:
        return None

    # 全局 2-opt 优化
    if use_2opt and len(full_path) > 4:
        full_path = optimize_2opt(full_path, grid)

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


def compute_energy_estimate(
    grid, path: List[Tuple[int, int, int]],
    boat_speed_ms: float = 3.0,
) -> Dict:
    """
    估算路径能耗和时间 (含水压/天气/漩涡影响)

    Args:
        grid: 水体网格
        path: 路径
        boat_speed_ms: 船速（米/秒），默认 3 m/s ≈ 5.8 节

    Returns:
        {
            "total_distance_m": 总距离,
            "estimated_time_s": 预估时间,
            "estimated_time_min": 预估时间（分钟）,
            "energy_consumption_kj": 能耗估算（千焦）,
            "avg_flow_assist": 平均水流辅助率（负值=逆流）,
            "depth_variation": 深度变化总量,
            "pressure_cost": 水压代价,
            "weather_cost": 天气代价,
            "waypoint_count": 途经点数量,
        }
    """
    total_dist, total_flow, total_depth = compute_3d_path_cost(grid, path)

    # 水压代价 (路径上各点水压)
    total_pressure_cost = 0.0
    total_weather_cost = 0.0
    total_temp_cost = 0.0
    total_vis_cost = 0.0
    for i, pt in enumerate(path):
        x, y, z = pt
        # 水压
        pressure = grid.get_pressure_at(x, y, z)
        if pressure > 200:
            total_pressure_cost += (pressure - 200) * 0.5
        # 天气 (仅表层)
        if z == 0:
            _, wind_spd, wave_h = grid.get_weather_at(x, y)
            total_weather_cost += wind_spd * 5.0 + wave_h * 15.0
        # 水温
        temp = grid.get_temperature_at(x, y)
        if temp < 5.0:
            total_temp_cost += (5.0 - temp) * 8.0
        elif temp > 30.0:
            total_temp_cost += (temp - 30.0) * 5.0
        # 能见度
        vis = grid.get_visibility_at(x, y)
        if vis < 5.0:
            total_vis_cost += (5.0 - vis) * 12.0

    # 时间估算：距离 / 船速
    estimated_time = total_dist / boat_speed_ms

    # 能耗估算
    base_energy_per_m = 500  # J/m (小型电动船)
    energy = total_dist * base_energy_per_m
    energy += total_flow * 1000          # 水流代价 → J
    energy += total_depth * 2000         # 深度变化能量
    energy += total_pressure_cost * 100  # 水压代价 → J
    energy += total_weather_cost * 50    # 天气代价 → J
    energy += total_temp_cost * 30       # 水温代价 → J
    energy += total_vis_cost * 25        # 能见度代价 → J
    energy_kj = energy / 1000.0

    # 潮汐放大因子
    tidal_amp = 0.6 + 0.8 * abs(math.sin(grid.tidal_phase * math.pi))

    # 平均水流辅助率
    avg_flow = total_flow / max(1, total_dist)

    return {
        "total_distance_m": total_dist,
        "estimated_time_s": round(estimated_time, 1),
        "estimated_time_min": round(estimated_time / 60.0, 1),
        "energy_consumption_kj": round(energy_kj, 1),
        "avg_flow_assist": round(avg_flow, 3),
        "depth_variation": total_depth,
        "pressure_cost": round(total_pressure_cost, 1),
        "weather_cost": round(total_weather_cost, 1),
        "temperature_cost": round(total_temp_cost, 1),
        "visibility_cost": round(total_vis_cost, 1),
        "tidal_amplification": round(tidal_amp, 2),
        "waypoint_count": len(path),
    }


def compare_strategies(
    grid,
    start: Tuple[int, int, int],
    waypoints: List[Tuple[int, int, int]],
    end: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Dict]:
    """
    多策略比较 — 用所有策略规划并返回对比结果

    Returns:
        {strategy_name: {path, costs, energy}}
    """
    results = {}
    for strategy in ["balanced", "safe", "fast", "energy"]:
        path = plan_tsp_3d(grid, start, waypoints, end, strategy=strategy)
        if path:
            dist, flow, depth_cost = compute_3d_path_cost(grid, path)
            energy = compute_energy_estimate(grid, path)
            results[strategy] = {
                "path": path,
                "distance": dist,
                "flow_cost": flow,
                "depth_cost": depth_cost,
                "energy_kj": energy["energy_consumption_kj"],
                "time_min": energy["estimated_time_min"],
                "waypoints_count": energy["waypoint_count"],
            }
        else:
            results[strategy] = None

    return results


def compare_and_select_best(
    grid,
    start: Tuple[int, int, int],
    waypoints: List[Tuple[int, int, int]],
    end: Optional[Tuple[int, int, int]] = None,
    criterion: str = "balanced",
) -> Tuple[Optional[List[Tuple[int, int, int]]], str, Dict]:
    """
    比较多策略结果，按指定标准选择最佳路径

    Args:
        criterion: 选择标准
            - "distance": 最短距离
            - "energy": 最低能耗
            - "time": 最省时间
            - "balanced": 综合评分

    Returns:
        (最佳路径, 使用的策略名, 所有结果字典)
    """
    results = compare_strategies(grid, start, waypoints, end)

    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        return None, "none", results

    if criterion == "distance":
        best = min(valid.items(), key=lambda x: x[1]["distance"])
    elif criterion == "energy":
        best = min(valid.items(), key=lambda x: x[1]["energy_kj"])
    elif criterion == "time":
        best = min(valid.items(), key=lambda x: x[1]["time_min"])
    else:  # balanced — 综合评分
        best = min(valid.items(), key=lambda x: (
            x[1]["distance"] * 0.3
            + x[1]["energy_kj"] * 0.01 * 0.3
            + x[1]["time_min"] * 0.2
            + x[1]["flow_cost"] * 0.2
        ))

    return best[1]["path"], best[0], results
