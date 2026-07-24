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
from typing import List, Tuple, Optional, Dict
import numpy as np


class Water3DGrid:
    """3D 水体网格 — 支持水深、水流、障碍物"""

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

        # 元数据
        self.metadata: Dict = {}
        self.mission_start: Optional[Tuple[int, int, int]] = None
        self.mission_waypoints: List[Tuple[int, int, int]] = []
        self.mission_end: Optional[Tuple[int, int, int]] = None

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
    """Demo: 沿海水域 30x30x8，均匀水深，有暗礁，有水流"""
    grid = Water3DGrid(30, 30, 8, resolution=50.0)  # 50m per cell
    grid.metadata = {"name": "沿海 Demo", "source": "模拟数据"}

    # 均匀水深 15m
    grid.set_uniform_bathymetry(15.0)

    # 表层水流向东偏北 0.5 m/s
    grid.set_uniform_current(dx=0.7, dy=-0.3, speed=0.5, layer="surface")
    grid.set_uniform_current(dx=0.3, dy=-0.1, speed=0.2, layer="mid")
    grid.set_uniform_current(dx=0.0, dy=0.0, speed=0.05, layer="bottom")

    # 暗礁区
    for x in range(10, 15):
        for y in range(10, 15):
            grid.add_obstacle(x, y, 3, 7)

    # 沉船
    grid.add_obstacle(22, 22, 4, 6)

    # 几个分散礁石
    for (x, y) in [(5, 8), (7, 20), (18, 5), (25, 15)]:
        grid.add_obstacle(x, y, 5, 7)

    # 任务
    grid.set_mission(
        start=(2, 2, 0),
        waypoints=[(15, 20, 0), (26, 10, 2)],
        end=(2, 2, 0),
    )
    return grid


def demo_3d_river():
    """Demo: 河道 40x15x5，中间深浅不一，有桥墩"""
    grid = Water3DGrid(40, 15, 5, resolution=20.0)
    grid.metadata = {"name": "河道 Demo", "source": "模拟数据"}

    # 河道：两边浅(陆地)，中间深
    for y in range(15):
        if y < 3 or y > 11:
            grid.depth[y, :] = -1  # 岸
        elif y < 5 or y > 9:
            grid.depth[y, :] = 3.0  # 浅水
        else:
            grid.depth[y, :] = 10.0  # 深水航道

    # 向下游的水流
    grid.set_uniform_current(dx=1.0, dy=0.0, speed=0.8, layer="surface")
    grid.set_uniform_current(dx=0.6, dy=0.0, speed=0.4, layer="mid")

    # 桥墩
    for y in range(5, 10):
        grid.add_obstacle(20, y, 0, 4)

    # 任务：从上游到下游
    grid.set_mission(
        start=(1, 7, 0),
        waypoints=[(15, 5, 1), (28, 9, 0), (38, 7, 0)],
        end=None,
    )
    return grid
