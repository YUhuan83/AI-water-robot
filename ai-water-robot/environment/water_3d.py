"""
3D 水体网格环境模块

坐标系:
    x — 经度方向（列），0 ~ nx-1
    y — 纬度方向（行），0 ~ ny-1
    z — 深度方向，0=水面，越深 z 越大

每个格子存储:
    - depth: 水深（米），-1 表示不可通行
    - current: (dx, dy, dz) 水流向量
    - current_speed: 水流速率（m/s）
    - obstacle: 是否有障碍物
"""

import json
import os
import math
from typing import List, Tuple, Optional, Dict
import numpy as np


class Water3DGrid:
    """3D 水体网格 — 支持水深、水流、障碍物、水压、天气、漩涡"""

    def __init__(self, nx: int, ny: int, nz: int, resolution: float = 100.0):
        """
        Args:
            nx, ny, nz: x/y/z 方向的网格数
            resolution: 每个格子的实际距离（米）
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.resolution = resolution  # 米/格

        # 核心数据
        self.depth = np.full((ny, nx), -1.0, dtype=np.float32)  # 水深，-1=不可行
        self.obstacles = np.zeros((nz, ny, nx), dtype=bool)     # 障碍物

        # 水流（分三层：表层、中层、底层）
        self.currents: Dict[str, np.ndarray] = {
            "surface": np.zeros((ny, nx, 3), dtype=np.float32),  # (dx, dy, dz)
            "mid":     np.zeros((ny, nx, 3), dtype=np.float32),
            "bottom":  np.zeros((ny, nx, 3), dtype=np.float32),
        }
        self.current_speeds: Dict[str, np.ndarray] = {
            "surface": np.zeros((ny, nx), dtype=np.float32),
            "mid":     np.zeros((ny, nx), dtype=np.float32),
            "bottom":  np.zeros((ny, nx), dtype=np.float32),
        }

        # 水压场 (nz, ny, nx) — 每层的水压 (kPa)
        self.pressure = np.zeros((nz, ny, nx), dtype=np.float32)

        # 天气场 (ny, nx) — 风速风向 + 浪高
        self.wind: Dict[str, np.ndarray] = {
            "dir": np.zeros((ny, nx, 3), dtype=np.float32),   # 风向 (dx,dy,dz)
            "speed": np.zeros((ny, nx), dtype=np.float32),     # 风速 m/s
            "wave_height": np.zeros((ny, nx), dtype=np.float32),  # 浪高 m
        }

        # 漩涡场 — 局部旋转水流中心列表 [(cx, cy, radius, strength)]
        self.eddies: List[Tuple[float, float, float, float]] = []

        # 水温场 (ny, nx) — 摄氏度
        self.temperature = np.full((ny, nx), 15.0, dtype=np.float32)

        # 浑浊度/能见度 (ny, nx) — 米 (越高=越清澈)
        self.visibility = np.full((ny, nx), 10.0, dtype=np.float32)

        # 潮汐时间因子 — 0.0=低潮 0.5=涨潮中 1.0=高潮 (会动态更新)
        self.tidal_phase = 0.5

        # 盐度场 (ny, nx) — PSU (实用盐度单位)
        self.salinity = np.full((ny, nx), 35.0, dtype=np.float32)

        # 元数据
        self.metadata: Dict = {}
        self.mission_start: Optional[Tuple[int, int, int]] = None
        self.mission_waypoints: List[Tuple[int, int, int]] = []
        self.mission_end: Optional[Tuple[int, int, int]] = None

        # 自动计算水压场
        self._compute_pressure_field()

    # ═══════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════

    def is_passable(self, x: int, y: int, z: int) -> bool:
        """检查 (x, y, z) 是否可通行"""
        if not (0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz):
            return False
        if self.depth[y, x] < 0:
            return False  # 陆地
        if z >= self._z_max(y, x):
            return False  # 超过水深
        if self.obstacles[z, y, x]:
            return False  # 障碍物
        return True

    def _z_max(self, y: int, x: int) -> int:
        """
        该位置可通行的最大深度层
        z 层按比例映射：水深 d 对应 z 从 0 到 int(d * nz / max_depth)
        """
        d = self.depth[y, x]
        if d <= 0:
            return 0
        safe_depths = self.depth[self.depth > 0]
        max_d = float(safe_depths.max()) if len(safe_depths) > 0 else 1.0
        return max(1, min(self.nz, int(d * self.nz / max_d)))

    def get_current_at(self, x: int, y: int, z: int) -> Tuple[Tuple[float, float, float], float]:
        """
        获取某位置的水流向量和速率

        Returns:
            ((dx, dy, dz), speed)
        """
        z_frac = z / max(1, self.nz - 1)
        if z_frac < 0.33:
            layer = "surface"
        elif z_frac < 0.66:
            layer = "mid"
        else:
            layer = "bottom"

        vec = self.currents[layer][y, x]
        speed = self.current_speeds[layer][y, x]
        return (float(vec[0]), float(vec[1]), float(vec[2])), float(speed)

    def get_depth_at(self, x: int, y: int) -> float:
        """获取水深（米）"""
        if 0 <= x < self.nx and 0 <= y < self.ny:
            return float(self.depth[y, x])
        return -1.0

    def _compute_pressure_field(self):
        """根据水深自动计算各层水压 (kPa) — 每10m≈100kPa"""
        for z in range(self.nz):
            z_frac = (z + 0.5) / max(1, self.nz)
            for y in range(self.ny):
                for x in range(self.nx):
                    d = self.depth[y, x]
                    if d > 0 and z < self._z_max(y, x):
                        depth_m = d * z_frac  # 当前层的实际深度(米)
                        self.pressure[z, y, x] = 101.3 + depth_m * 9.8  # 大气压 + 水压
                    else:
                        self.pressure[z, y, x] = 0.0

    def get_pressure_at(self, x: int, y: int, z: int) -> float:
        """获取水压 (kPa)"""
        zc = max(0, min(z, self.nz - 1))
        if 0 <= x < self.nx and 0 <= y < self.ny:
            return float(self.pressure[zc, y, x])
        return 101.3

    def get_weather_at(self, x: int, y: int) -> Tuple[Tuple[float, float, float], float, float]:
        """获取天气: (风向, 风速m/s, 浪高m)"""
        if 0 <= x < self.nx and 0 <= y < self.ny:
            wd = self.wind["dir"][y, x]
            return (float(wd[0]), float(wd[1]), float(wd[2])), float(self.wind["speed"][y, x]), float(self.wind["wave_height"][y, x])
        return (0.0, 0.0, 0.0), 0.0, 0.0

    def get_eddy_effect_at(self, x: int, y: int) -> Tuple[float, float, float]:
        """获取所有漩涡在该点的合成效应 (vx, vy, vz)"""
        vx, vy, vz = 0.0, 0.0, 0.0
        for cx, cy, radius, strength in self.eddies:
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < radius and dist > 0.01:
                # 切向速度 (逆时针旋转)
                factor = strength * (1.0 - dist / radius)  # 边缘衰减
                vx += -dy / dist * factor
                vy += dx / dist * factor
                # 中心有微弱下沉流
                vz += -0.02 * factor * (1.0 - dist / radius)
        return (vx, vy, vz)

    def get_total_current_at(self, x: int, y: int, z: int) -> Tuple[Tuple[float, float, float], float]:
        """获取总水流 = (基础水流 + 漩涡效应 + 风生流) × 潮汐放大"""
        base_vec, base_spd = self.get_current_at(x, y, z)
        eddy_vx, eddy_vy, eddy_vz = self.get_eddy_effect_at(x, y)

        # 风生流 (仅表层 z=0: 风速的2%转化为表层流)
        wind_vec, wind_spd, _ = self.get_weather_at(x, y)
        wind_factor = 0.02 if z == 0 else 0.0

        total_vx = base_vec[0] + eddy_vx + wind_vec[0] * wind_factor
        total_vy = base_vec[1] + eddy_vy + wind_vec[1] * wind_factor
        total_vz = base_vec[2] + eddy_vz + wind_vec[2] * wind_factor

        # 潮汐放大: 高潮/低潮时流速最大，半潮时最小
        tidal_amp = 0.6 + 0.8 * abs(math.sin(self.tidal_phase * math.pi))
        total_vx *= tidal_amp
        total_vy *= tidal_amp
        total_vz *= tidal_amp

        total_spd = math.sqrt(total_vx ** 2 + total_vy ** 2 + total_vz ** 2)

        return (total_vx, total_vy, total_vz), total_spd

    def get_temperature_at(self, x: int, y: int) -> float:
        """获取水温 (摄氏度)"""
        if 0 <= x < self.nx and 0 <= y < self.ny:
            return float(self.temperature[y, x])
        return 15.0

    def get_visibility_at(self, x: int, y: int) -> float:
        """获取能见度 (米) — 低值表示浑浊水域"""
        if 0 <= x < self.nx and 0 <= y < self.ny:
            return float(self.visibility[y, x])
        return 10.0

    def set_tidal_phase(self, phase: float):
        """设置潮汐相位 0~1 (0=低潮, 0.5=半潮, 1=高潮)"""
        self.tidal_phase = max(0.0, min(1.0, phase))

    def set_uniform_temperature(self, temp_c: float):
        """设置均匀水温"""
        self.temperature.fill(temp_c)

    def set_uniform_visibility(self, vis_m: float):
        """设置均匀能见度"""
        self.visibility.fill(max(0.1, vis_m))

    def set_uniform_weather(self, dx: float, dy: float, dz: float = 0.0,
                            speed: float = 0.0, wave_height: float = 0.0):
        """设置均匀天气"""
        self.wind["dir"][:, :] = [dx, dy, dz]
        self.wind["speed"][:, :] = speed
        self.wind["wave_height"][:, :] = wave_height

    def add_eddy(self, cx: float, cy: float, radius: float, strength: float):
        """添加漩涡 (cx,cy=中心, radius=半径, strength=强度 m/s)"""
        self.eddies.append((cx, cy, radius, strength))

    def has_obstacle_near(self, x: int, y: int, z: int, margin: int = 1) -> bool:
        """检查 (x,y,z) 周围 margin 格内是否有障碍物"""
        for dz in range(-1, 2):
            nz_ = z + dz
            if not (0 <= nz_ < self.nz):
                continue
            for dy in range(-margin, margin + 1):
                for dx in range(-margin, margin + 1):
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny:
                        if self.obstacles[nz_, ny_, nx_]:
                            return True
        return False

    def obstacle_proximity_penalty(self, x: int, y: int, z: int, max_margin: int = 3) -> float:
        """计算障碍物接近惩罚 (距离越近惩罚越大)"""
        penalty = 0.0
        for margin in range(1, max_margin + 1):
            if self.has_obstacle_near(x, y, z, margin):
                # 距离越近惩罚越大 (1格距离=100, 2格=40, 3格=15)
                penalty += 100.0 / (margin * margin)
        return penalty

    # ═══════════════════════════════════════════════════════
    # 场景构建
    # ═══════════════════════════════════════════════════════

    def set_uniform_bathymetry(self, water_depth: float):
        """设置均匀水深"""
        self.depth.fill(water_depth)

    def set_uniform_current(self, dx: float, dy: float, dz: float = 0.0,
                            speed: float = 0.0, layer: str = "surface"):
        """设置均匀水流"""
        vec = np.array([dx, dy, dz], dtype=np.float32)
        self.currents[layer][:, :] = vec
        self.current_speeds[layer][:, :] = speed

    def add_obstacle(self, x: int, y: int, z_min: int, z_max: int):
        """添加障碍物（占据 z_min 到 z_max 的深度层）"""
        for z in range(z_min, min(z_max + 1, self.nz)):
            if 0 <= x < self.nx and 0 <= y < self.ny:
                self.obstacles[z, y, x] = True

    def set_mission(self, start: Tuple[int, int, int],
                    waypoints: List[Tuple[int, int, int]],
                    end: Optional[Tuple[int, int, int]] = None):
        """设置任务"""
        self.mission_start = start
        self.mission_waypoints = waypoints
        self.mission_end = end

    def get_obstacle_mask_3d(self) -> np.ndarray:
        """返回 3D 布尔障碍物掩码（含水深限制）"""
        mask = np.zeros((self.nz, self.ny, self.nx), dtype=bool)
        for y in range(self.ny):
            for x in range(self.nx):
                z_max = self._z_max(y, x)
                mask[z_max:, y, x] = True  # 超过水深不可通行
                if self.depth[y, x] < 0:
                    mask[:, y, x] = True   # 陆地完全不可通行
        mask |= self.obstacles
        return mask

    # ═══════════════════════════════════════════════════════
    # JSON 导入/导出
    # ═══════════════════════════════════════════════════════

    @classmethod
    def from_json(cls, path: str) -> "Water3DGrid":
        """从 JSON 文件加载 3D 水体数据"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        domain = data["domain"]
        grid = cls(domain["nx"], domain["ny"], domain["nz"],
                   data.get("metadata", {}).get("resolution", 100.0))
        grid.metadata = data.get("metadata", {})

        # 水深数据
        bathy = data.get("bathymetry", {})
        if bathy.get("format") == "2d_array":
            arr = np.array(bathy["data"], dtype=np.float32)
            if arr.shape == (domain["ny"], domain["nx"]):
                grid.depth = arr
            else:
                raise ValueError(f"水深数据尺寸不匹配: {arr.shape} vs ({domain['ny']},{domain['nx']})")

        # 水流数据
        currents = data.get("currents", {})
        if currents.get("format") == "uniform":
            for layer in ["surface", "mid", "bottom"]:
                ld = currents["data"].get(layer, {})
                if ld:
                    grid.set_uniform_current(
                        ld.get("dx", 0), ld.get("dy", 0), ld.get("dz", 0),
                        ld.get("speed", 0), layer,
                    )

        # 障碍物
        for obs in data.get("obstacles", []):
            grid.add_obstacle(obs["x"], obs["y"],
                              obs.get("z_min", 0), obs.get("z_max", 0))

        # 任务
        mission = data.get("mission", {})
        start = mission.get("start")
        if start:
            s = (start["x"], start["y"], start.get("z", 0))
            waypoints = [(w["x"], w["y"], w.get("z", 0))
                         for w in mission.get("waypoints", [])]
            end = mission.get("end")
            e = (end["x"], end["y"], end.get("z", 0)) if end else None
            grid.set_mission(s, waypoints, e)

        return grid

    def to_json(self, path: str):
        """导出为 JSON"""
        data = {
            "metadata": self.metadata,
            "domain": {"nx": self.nx, "ny": self.ny, "nz": self.nz},
            "bathymetry": {
                "format": "2d_array",
                "data": self.depth.tolist(),
            },
            "obstacles": [
                {"x": int(x), "y": int(y), "z": int(z)}
                for z in range(self.nz)
                for y in range(self.ny)
                for x in range(self.nx)
                if self.obstacles[z, y, x]
            ],
            "mission": {
                "start": {"x": self.mission_start[0], "y": self.mission_start[1],
                          "z": self.mission_start[2]} if self.mission_start else None,
                "waypoints": [{"x": w[0], "y": w[1], "z": w[2]}
                              for w in self.mission_waypoints],
                "end": {"x": self.mission_end[0], "y": self.mission_end[1],
                        "z": self.mission_end[2]} if self.mission_end else None,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """可读摘要"""
        passable = int(np.sum(self.depth >= 0))
        n_obs = int(np.sum(self.obstacles))
        return (
            f"3D Water Grid: {self.nx}x{self.ny}x{self.nz} "
            f"({self.resolution}m/cell)\n"
            f"  Passable cells: {passable}/{self.nx * self.ny}\n"
            f"  Obstacles: {n_obs}\n"
            f"  Start: {self.mission_start}\n"
            f"  Waypoints: {len(self.mission_waypoints)}\n"
            f"  End: {self.mission_end}"
        )


# ═══════════════════════════════════════════════════════════
# 预设 3D 场景
# ═══════════════════════════════════════════════════════════

def demo_3d_coastal():
    """Demo: 沿海水域 30x30x8 — 渐变水深/复杂洋流/暗礁群/沉船/管道"""
    grid = Water3DGrid(30, 30, 8, resolution=50.0)  # 50m per cell, 1.5km x 1.5km
    grid.metadata = {
        "name": "沿海水域 — 港口外海",
        "source": "模拟数据",
        "description": "近岸浅水区渐变至深海，含暗礁群、沉船遗址、海底管道。表层有沿岸流，中层有潮汐余流。",
    }

    # ── 渐变水深：近岸(左上)浅→远海(右下)深 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            # 基础：离岸距离决定水深 (左上角为岸，右下为深海)
            dist_offshore = (x + y) / (grid.nx + grid.ny)  # 0=岸, 1=深海
            base_depth = 2.0 + dist_offshore * 28.0

            # 海底地形起伏 (模拟沙波和海沟)
            terrain_noise = (
                3.0 * math.sin(x * 0.3) * math.cos(y * 0.35)
                + 2.0 * math.sin((x + y) * 0.2)
                + 1.5 * math.cos(x * 0.15 - y * 0.25)
            )

            # 沿岸沙洲 (左上角特别浅)
            if x + y < 8:
                terrain_noise -= 3.0

            # 一条深水航道 (对角线方向)
            channel_dist = abs((x - y) / math.sqrt(2))
            if channel_dist < 3:
                terrain_noise += 4.0  # 航道加深

            depth = max(0.5, base_depth + terrain_noise)
            grid.depth[y, x] = float(depth)

    # ── 分层水流 ──
    # 表层：沿岸流 (平行于海岸，大致从左上流向右下)
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] <= 0:
                continue
            # 主流向：东偏南 (平行于海岸线)
            dist = (x + y) / (grid.nx + grid.ny)
            main_dx = 0.6 + dist * 0.2
            main_dy = 0.4 + dist * 0.1
            main_spd = 0.3 + dist * 0.4  # 离岸越远流速越大

            # 局部涡流 (暗礁附近)
            reef_dist = math.sqrt((x - 12.5) ** 2 + (y - 12.5) ** 2)
            if reef_dist < 6:
                eddy_angle = math.atan2(y - 12.5, x - 12.5) + math.pi / 2
                main_dx += math.cos(eddy_angle) * 0.3
                main_dy += math.sin(eddy_angle) * 0.3
                main_spd += 0.2

            mag = math.sqrt(main_dx ** 2 + main_dy ** 2) or 1.0
            grid.currents["surface"][y, x] = [main_dx / mag, main_dy / mag, 0.0]
            grid.current_speeds["surface"][y, x] = float(main_spd)

    # 中层：潮汐余流 (较弱的往复流)
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] <= 0:
                continue
            tide_phase = (x + y) * 0.2
            grid.currents["mid"][y, x] = [math.cos(tide_phase) * 0.5,
                                            math.sin(tide_phase) * 0.3, 0.0]
            grid.current_speeds["mid"][y, x] = 0.2 + abs(math.sin(tide_phase)) * 0.15

    # 底层：极弱流
    grid.set_uniform_current(dx=0.1, dy=0.05, speed=0.05, layer="bottom")

    # ── 障碍物 ──
    # 1) 暗礁群 (不规则形状，占据中层到底层)
    reef_centers = [(12, 12), (13, 11), (11, 13), (14, 12)]
    for cx, cy in reef_centers:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx * dx + dy * dy <= 5:  # 圆形暗礁
                    grid.add_obstacle(cx + dx, cy + dy, 4, 7)

    # 2) 沉船遗址 (细长形状)
    for dx in range(-1, 2):
        grid.add_obstacle(24 + dx, 22, 5, 7)
    grid.add_obstacle(24, 21, 6, 7)
    grid.add_obstacle(24, 23, 6, 7)

    # 3) 海底管道 (线状)
    for t in range(15):
        x, y = 6 + t, 3 + int(t * 0.4)
        if 0 <= x < grid.nx and 0 <= y < grid.ny:
            grid.add_obstacle(x, y, 6, 7)

    # 4) 散落礁石
    for (x, y) in [(5, 8), (8, 22), (18, 5), (27, 12), (20, 28), (3, 18)]:
        grid.add_obstacle(x, y, 5, 7)
        grid.add_obstacle(x, y, 4, 4)  # 向上凸起

    # 5) 近岸防波堤
    for dx in range(4):
        grid.add_obstacle(3 + dx, 3, 0, 4)

    # ── 环境: 天气和漩涡 ──
    # 东北风 6m/s，近海有中等浪
    grid.set_uniform_weather(dx=-0.6, dy=0.4, speed=6.0, wave_height=1.5)
    # 暗礁区附近漩涡
    grid.add_eddy(cx=12.5, cy=12.5, radius=7.0, strength=0.4)
    # 沉船附近小漩涡
    grid.add_eddy(cx=24.0, cy=22.0, radius=3.0, strength=0.25)
    # 外海大漩涡
    grid.add_eddy(cx=20.0, cy=25.0, radius=5.0, strength=0.3)

    # ── 水温: 近岸暖、远海凉 (夏季沿岸) ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] > 0:
                offshore = (x + y) / (grid.nx + grid.ny)
                grid.temperature[y, x] = 26.0 - offshore * 6.0  # 近岸26°C→远海20°C

    # ── 能见度: 近岸浑浊(港口/河流泥沙)，远海清澈 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] > 0:
                offshore = (x + y) / (grid.nx + grid.ny)
                # 暗礁区附近能见度低 (搅动泥沙)
                reef_dist = math.sqrt((x - 12.5)**2 + (y - 12.5)**2)
                if reef_dist < 4:
                    grid.visibility[y, x] = 2.0 + reef_dist * 0.8
                elif x + y < 8:
                    grid.visibility[y, x] = 3.0  # 港口区浑浊
                else:
                    grid.visibility[y, x] = 4.0 + offshore * 12.0  # 远海清澈

    # ── 潮汐: 当前为涨潮中期 ──
    grid.set_tidal_phase(0.65)

    # ── 任务：从港口出发，巡检外海设施，避开暗礁，返回 ──
    grid.set_mission(
        start=(2, 2, 0),                     # 港口
        waypoints=[
            (8, 15, 1),                      # 航道浮标点
            (18, 8, 2),                      # 南侧巡检点 (绕开暗礁)
            (27, 15, 1),                     # 沉船遗址附近 (深水)
            (22, 27, 0),                     # 外海观测点
        ],
        end=(2, 2, 0),                       # 返回港口
    )
    return grid


def demo_3d_river():
    """Demo: 河道 40x15x5 — S型蜿蜒河道/变速水流/桥墩群/浅滩/巡逻"""
    grid = Water3DGrid(40, 15, 5, resolution=20.0)  # 20m per cell, 800m x 300m
    grid.metadata = {
        "name": "内河航道 — 蜿蜒河段",
        "source": "模拟数据",
        "description": "S型弯曲河道，含两座桥梁、浅滩沙洲、水下暗桩。上游流速快，弯道处有回流。",
    }

    # ── S型蜿蜒河道 ──
    # 河道中心线：从 x=0 到 x=39，呈 S 型弯曲
    def river_center(x: float) -> float:
        """河道中心 y 坐标，x→0 时靠近 y=7，中间弯曲"""
        phase = x / 39.0 * math.pi * 2  # 两个完整周期形成 S
        return 7.0 + 3.5 * math.sin(phase)

    def river_width(x: float) -> float:
        """河道半宽，弯道处略宽，直道处窄"""
        phase = x / 39.0 * math.pi * 2
        return 2.8 + 0.6 * abs(math.cos(phase))

    for y in range(grid.ny):
        for x in range(grid.nx):
            center_y = river_center(x)
            half_w = river_width(x)
            dist_from_center = abs(y - center_y)

            if dist_from_center < half_w:
                # 主航道：中间深，两边浅
                depth_frac = dist_from_center / half_w
                # 弯道外侧更深 (冲刷效应)
                outer_deep = 0.0
                phase = x / 39.0 * math.pi * 2
                if (y - center_y) * math.cos(phase) > 0:  # 弯道外侧
                    outer_deep = 2.0
                grid.depth[y, x] = float(2.0 + (1.0 - depth_frac) * 8.0 + outer_deep)
            elif dist_from_center < half_w + 1.2:
                # 河岸缓坡 (浅水区)
                grid.depth[y, x] = float(1.5 - (dist_from_center - half_w))
            else:
                # 陆地
                grid.depth[y, x] = -1.0

    # ── 深潭和浅滩 ──
    # 弯道外侧深潭
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] <= 0:
                continue
            phase = x / 39.0 * math.pi * 2
            center_y = river_center(x)
            if abs(y - (center_y + 3.5 * math.cos(phase))) < 1.0:
                grid.depth[y, x] += 3.0  # 弯道外侧加深
            # 两弯之间的浅滩
            if 15 < x < 22 and abs(y - center_y) > 2:
                grid.depth[y, x] = max(0.8, grid.depth[y, x] - 5.0)

    # ── 变速水流 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] <= 0:
                continue
            phase = x / 39.0 * math.pi * 2
            center_y = river_center(x)

            # 主流向下游 (东)，流速受河宽影响
            half_w = river_width(x)
            narrow_factor = 3.5 / max(2.0, half_w)  # 窄处加速

            # 表层流速
            base_speed = 0.4 + narrow_factor * 0.6
            # 弯道外侧快，内侧慢 (可能有回流)
            if (y - center_y) * math.cos(phase) > 0:  # 外侧
                base_speed += 0.3
            else:
                base_speed -= 0.15

            # 离岸越近越慢 (边界层效应)
            dist_from_center = abs(y - center_y)
            shore_factor = max(0.1, 1.0 - (half_w - dist_from_center) / half_w * 0.7)

            surf_spd = max(0.05, base_speed * shore_factor)
            grid.current_speeds["surface"][y, x] = float(surf_spd)
            # 流向大致向东，但在弯道处偏转
            angle = math.atan2(
                river_center(x + 1) - river_center(x - 1) if 0 < x < grid.nx - 1 else 0.0,
                2.0
            )
            grid.currents["surface"][y, x] = [float(math.cos(angle)), float(math.sin(angle)), 0.0]

    # 中层 (较慢)
    grid.set_uniform_current(dx=0.5, dy=0.0, speed=0.3, layer="mid")
    grid.set_uniform_current(dx=0.0, dy=0.0, speed=0.02, layer="bottom")

    # ── 障碍物 ──
    # 1) 第一座桥 (桥墩群, x≈10)
    for b in [(10, 5), (10, 6), (10, 8), (10, 9)]:
        for dz in range(5):
            grid.add_obstacle(b[0], b[1], dz, dz)

    # 2) 第二座桥 (桥墩群, x≈28)
    for b in [(27, 4), (27, 5), (28, 9), (28, 10)]:
        for dz in range(5):
            grid.add_obstacle(b[0], b[1], dz, dz)

    # 3) 浅滩暗桩 (河中央散落的木桩/石块)
    for (x, y) in [(5, 8), (8, 6), (15, 5), (18, 10), (25, 4), (32, 9), (35, 6)]:
        grid.add_obstacle(x, y, 0, 2)

    # 4) 岸边凸出岩石
    for x in [6, 7, 20, 21, 33, 34]:
        grid.add_obstacle(x, int(river_center(x) + river_width(x) - 0.5), 0, 1)

    # ── 环境: 天气和漩涡 ──
    # 下游风 3m/s，轻浪
    grid.set_uniform_weather(dx=0.2, dy=-0.1, speed=3.0, wave_height=0.6)
    # 弯道外侧漩涡 (x≈19, y≈10)
    grid.add_eddy(cx=19.0, cy=10.0, radius=3.5, strength=0.3)
    # 桥墩后方涡流 (x≈10)
    grid.add_eddy(cx=10.5, cy=7.5, radius=2.0, strength=0.2)
    # 第二座桥下游涡流
    grid.add_eddy(cx=28.5, cy=7.0, radius=2.5, strength=0.25)

    # ── 水温: 上游凉→下游暖 (夏季内河) ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] > 0:
                grid.temperature[y, x] = 18.0 + (x / grid.nx) * 6.0  # 上游18°C→下游24°C

    # ── 能见度: 弯道/浅滩浑浊，主流清澈 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] > 0:
                center_y = river_center(x)
                dist_from_center = abs(y - center_y)
                # 浅滩区泥沙搅动
                if 15 < x < 22 and dist_from_center > 2:
                    grid.visibility[y, x] = 1.5
                elif dist_from_center < 1.5:
                    grid.visibility[y, x] = 8.0  # 主流清澈
                else:
                    grid.visibility[y, x] = 3.0 + dist_from_center * 1.5

    # ── 潮汐: 内河受潮汐影响较弱 ──
    grid.set_tidal_phase(0.5)

    # ── 任务：河道巡逻 ──
    grid.set_mission(
        start=(1, 7, 0),                      # 上游码头 (河道中心)
        waypoints=[
            (8, 9, 0),                        # 第一弯道巡检
            (19, 10, 0),                      # 弯道外侧观测点
            (30, 7, 0),                       # 第二座桥下游 (浅水)
            (38, 7, 0),                       # 终点前检查点
        ],
        end=(38, 7, 0),                       # 下游码头
    )
    return grid


def demo_3d_harbor():
    """Demo: 港口水域 25x25x6 — 码头/航道/浮标/泊位"""
    grid = Water3DGrid(25, 25, 6, resolution=30.0)  # 30m per cell, 750m x 750m
    grid.metadata = {
        "name": "港口水域 — 码头作业区",
        "source": "模拟数据",
        "description": "繁忙港口水域，含码头泊位、进出港航道、锚地。需巡检浮标、避让泊位障碍物。",
    }

    # ── 港口地形 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            # 左上是陆地/码头区
            dist_from_shore = x + y * 0.7
            if dist_from_shore < 5:
                grid.depth[y, x] = -1.0  # 陆地
            elif dist_from_shore < 8:
                grid.depth[y, x] = 3.0 + (dist_from_shore - 5) * 3   # 浅水泊位区
            elif dist_from_shore < 15:
                grid.depth[y, x] = 12.0  # 中等水深航道
            else:
                grid.depth[y, x] = 18.0 + (dist_from_shore - 15) * 1.5  # 深水锚地

    # ── 主航道挖深 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] <= 0:
                continue
            # 对角线主航道
            channel_dist = abs((x - y * 0.8 - 2) / math.sqrt(1 + 0.64))
            if channel_dist < 2.5:
                grid.depth[y, x] = max(grid.depth[y, x], 16.0)

    # ── 水流：潮汐进出港流 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] <= 0:
                continue
            # 沿航道方向的往复流 (近似对角)
            grid.currents["surface"][y, x] = [0.55, -0.35, 0.0]
            grid.current_speeds["surface"][y, x] = 0.3
    grid.set_uniform_current(dx=0.2, dy=-0.1, speed=0.1, layer="mid")
    grid.set_uniform_current(dx=0.0, dy=0.0, speed=0.02, layer="bottom")

    # ── 障碍物 ──
    # 1) 码头泊位 (靠岸区的停泊船只 — 模拟为障碍物)
    for (x, y) in [(3, 1), (4, 2), (2, 3), (5, 3), (3, 4), (6, 2)]:
        if grid.depth[y, x] > 0:
            grid.add_obstacle(x, y, 0, 2)

    # 2) 防波堤
    for t in range(8):
        bx, by = 8 - t, t + 1
        if 0 <= bx < grid.nx and 0 <= by < grid.ny:
            grid.add_obstacle(bx, by, 0, 4)

    # 3) 航道中的施工浮台
    grid.add_obstacle(14, 10, 0, 3)
    grid.add_obstacle(14, 11, 0, 3)
    grid.add_obstacle(15, 10, 0, 3)
    grid.add_obstacle(15, 11, 0, 3)

    # 4) 锚地中停泊的大型船只
    for (x, y) in [(20, 18), (22, 15), (18, 21)]:
        if grid.depth[y, x] > 0:
            grid.add_obstacle(x, y, 0, 1)

    # ── 环境: 天气和漩涡 ──
    # 海风 5m/s，轻到中浪
    grid.set_uniform_weather(dx=-0.3, dy=-0.5, speed=5.0, wave_height=1.0)
    # 防波堤外侧漩涡
    grid.add_eddy(cx=6.0, cy=6.0, radius=3.0, strength=0.2)
    # 航道施工区附近涡流
    grid.add_eddy(cx=15.0, cy=10.5, radius=3.5, strength=0.3)
    # 锚地附近缓流漩涡
    grid.add_eddy(cx=20.0, cy=18.0, radius=4.0, strength=0.15)

    # ── 水温: 码头区受排放影响略高 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] > 0:
                dist_shore = x + y * 0.7
                if dist_shore < 6:
                    grid.temperature[y, x] = 24.0  # 码头水温偏高
                elif dist_shore < 12:
                    grid.temperature[y, x] = 22.0
                else:
                    grid.temperature[y, x] = 20.0

    # ── 能见度: 码头区浑浊(泥沙/油污)，外海较好 ──
    for y in range(grid.ny):
        for x in range(grid.nx):
            if grid.depth[y, x] > 0:
                dist_shore = x + y * 0.7
                # 施工区能见度极低
                if 13 <= x <= 16 and 9 <= y <= 12:
                    grid.visibility[y, x] = 1.0
                elif dist_shore < 6:
                    grid.visibility[y, x] = 2.5
                elif dist_shore < 12:
                    grid.visibility[y, x] = 5.0
                else:
                    grid.visibility[y, x] = 8.0 + (dist_shore - 12) * 0.5

    # ── 潮汐: 港口区受潮汐影响明显 — 退潮中 ──
    grid.set_tidal_phase(0.25)

    # ── 任务：港口浮标巡检 ──
    grid.set_mission(
        start=(5, 2, 0),                      # 码头出发 (航道入口)
        waypoints=[
            (8, 6, 0),                        # 1号浮标 (航道入口)
            (12, 10, 0),                      # 2号浮标 (航道中段)
            (17, 13, 0),                      # 3号浮标 (施工区南侧绕行)
            (20, 17, 0),                      # 4号浮标 (锚地)
            (12, 14, 0),                      # 5号浮标 (返程航道)
        ],
        end=(5, 2, 0),                        # 返回码头
    )
    return grid
