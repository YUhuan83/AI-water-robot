"""
水况数据适配层

功能: 将多种原始数据格式转换为 Water3DGrid JSON 标准格式
支持: CSV水深 / XYZ点云 / 纯文本矩阵 / 水流文本 / 自动格式检测

用法:
    from data.water_adapter import load_water_data
    grid = load_water_data("bathymetry.csv")           # 自动检测
    grid = load_water_data("depths.txt", fmt="matrix")  # 指定格式
"""

import os
import json
import csv
import re
from typing import Optional, Dict, List, Tuple
import numpy as np

from environment.water_3d import Water3DGrid


# ═══════════════════════════════════════════════════════════
# 格式检测
# ═══════════════════════════════════════════════════════════

def detect_format(filepath: str) -> str:
    """
    自动检测文件格式

    Returns:
        'json' | 'csv' | 'xyz' | 'matrix' | 'unknown'
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".json":
        return "json"

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        head = "".join(f.readline() for _ in range(5))

    # CSV: .csv 后缀直接判定，或检测逗号分隔
    if ext == ".csv":
        return "csv"
    if "," in head.split("\n")[0] and "\t" not in head.split("\n")[0]:
        for line in head.split("\n")[:5]:
            line = line.strip()
            if line and not line.startswith("#") and line.count(",") >= 2:
                return "csv"

    # XYZ: 每行 "x y z" 或 "x,y,z" 或 "lon lat depth"
    lines = [l.strip() for l in head.split("\n") if l.strip() and not l.startswith("#")]
    if lines:
        parts = re.split(r"[\s,]+", lines[0])
        if len(parts) == 3:
            try:
                float(parts[0]); float(parts[1]); float(parts[2])
                return "xyz"
            except ValueError:
                pass

    # Matrix: 每行都是空格/制表符分隔的数字
    if lines:
        n_cols = len(re.split(r"\s+", lines[0]))
        if n_cols >= 3 and all(
            len(re.split(r"\s+", l)) == n_cols for l in lines[:3]
        ):
            try:
                float(re.split(r"\s+", lines[0])[0])
                return "matrix"
            except ValueError:
                pass

    return "unknown"


# ═══════════════════════════════════════════════════════════
# 各格式解析器
# ═══════════════════════════════════════════════════════════

def parse_xyz(filepath: str) -> np.ndarray:
    """
    解析 XYZ 点云格式 → 2D 水深数组

    文件格式:
        x y depth
        0 0 5.2
        1 0 5.5
        ...
    """
    points = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,]+", line)
            if len(parts) >= 3:
                try:
                    x, y, d = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append((int(round(x)), int(round(y)), d))
                except ValueError:
                    continue

    if not points:
        raise ValueError("XYZ 文件未解析到有效数据")

    xs, ys, ds = zip(*points)
    nx, ny = max(xs) + 1, max(ys) + 1
    depth = np.full((ny, nx), -1.0, dtype=np.float32)
    for x, y, d in points:
        if 0 <= y < ny and 0 <= x < nx:
            depth[y, x] = d
    return depth


def parse_csv(filepath: str) -> np.ndarray:
    """
    解析 CSV 水深数据 → 2D 数组

    文件格式:
        # 可选注释行
        5.0,5.5,6.0,...
        5.2,5.8,6.2,...
    """
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                vals = [float(v) for v in row if v.strip()]
                if vals:
                    rows.append(vals)
            except ValueError:
                continue

    if not rows:
        raise ValueError("CSV 文件未解析到有效数据")

    n_cols = len(rows[0])
    rows = [r for r in rows if len(r) == n_cols]
    return np.array(rows, dtype=np.float32)


def parse_matrix(filepath: str) -> np.ndarray:
    """
    解析纯文本矩阵 → 2D 数组

    文件格式:
        5.0 5.5 6.0 6.5
        5.2 5.8 6.2 6.7
        ...
    """
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            try:
                vals = [float(p) for p in parts if p]
                if vals:
                    rows.append(vals)
            except ValueError:
                continue

    if not rows:
        raise ValueError("矩阵文件未解析到有效数据")

    n_cols = len(rows[0])
    rows = [r for r in rows if len(r) == n_cols]
    return np.array(rows, dtype=np.float32)


def parse_currents_text(filepath: str) -> Dict:
    """
    解析水流文本文件 → 水流字典

    文件格式:
        layer surface dx=0.5 dy=-0.3 speed=0.6
        layer mid dx=0.2 dy=-0.1 speed=0.3
        layer bottom dx=0.0 dy=0.0 speed=0.05
    """
    currents = {}
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(
                r"layer\s+(\w+)\s+dx=([\d.\-]+)\s+dy=([\d.\-]+)\s+(?:dz=([\d.\-]+)\s+)?speed=([\d.\-]+)",
                line,
            )
            if match:
                layer = match.group(1)
                currents[layer] = {
                    "dx": float(match.group(2)),
                    "dy": float(match.group(3)),
                    "dz": float(match.group(4) or 0),
                    "speed": float(match.group(5)),
                }
    return currents


# ═══════════════════════════════════════════════════════════
# 主加载函数
# ═══════════════════════════════════════════════════════════

def load_water_data(
    bathymetry_path: str,
    currents_path: Optional[str] = None,
    obstacles_path: Optional[str] = None,
    mission: Optional[Dict] = None,
    resolution: float = 100.0,
    nz: int = 10,
    fmt: Optional[str] = None,
) -> Water3DGrid:
    """
    从原始数据文件加载 3D 水体网格

    Args:
        bathymetry_path: 水深数据文件 (csv/xyz/matrix/json)
        currents_path:  水流数据文件 (可选)
        obstacles_path: 障碍物数据文件 (可选, JSON)
        mission:        任务定义字典 {"start":(x,y,z), "waypoints":[...], "end":(x,y,z)}
        resolution:     网格精度 (米/格)
        nz:             深度层数
        fmt:            文件格式 (auto/csv/xyz/matrix/json)，默认自动检测

    Returns:
        Water3DGrid 实例

    用法示例:
        # 从 CSV 水深 + 水流文本 加载
        grid = load_water_data("depths.csv", "currents.txt", resolution=50)

        # 从 XYZ 点云 加载
        grid = load_water_data("sonar_data.xyz", fmt="xyz")

        # 从 JSON 加载 (直接透传)
        grid = load_water_data("data.json")

        # 带任务定义
        grid = load_water_data("bathy.csv",
            mission={"start": (5,5,0), "waypoints": [(20,25,0)], "end": (5,5,0)})
    """
    # JSON 直接加载
    ext = os.path.splitext(bathymetry_path)[1].lower()
    if ext == ".json":
        grid = Water3DGrid.from_json(bathymetry_path)
        _apply_mission(grid, mission)
        return grid

    # 检测或使用指定格式
    file_fmt = fmt or detect_format(bathymetry_path)
    if file_fmt == "unknown":
        raise ValueError(
            f"无法识别文件格式: {bathymetry_path}\n"
            f"支持的格式: CSV, XYZ, Matrix, JSON\n"
            f"可用 fmt= 参数手动指定格式"
        )

    # 解析水深
    parsers = {"xyz": parse_xyz, "csv": parse_csv, "matrix": parse_matrix}
    parser = parsers.get(file_fmt)
    if parser is None:
        raise ValueError(f"不支持的格式: {file_fmt}，可用: {list(parsers.keys())}")

    depth = parser(bathymetry_path)
    ny, nx = depth.shape

    # 构建 Water3DGrid
    grid = Water3DGrid(nx, ny, nz, resolution)
    grid.depth = depth.astype(np.float32)
    grid.metadata = {
        "name": os.path.basename(bathymetry_path),
        "source": file_fmt,
        "resolution": resolution,
        "nz": nz,
    }

    # 加载水流
    if currents_path and os.path.exists(currents_path):
        currents = parse_currents_text(currents_path)
        for layer in ["surface", "mid", "bottom"]:
            if layer in currents:
                c = currents[layer]
                grid.set_uniform_current(c["dx"], c["dy"], c.get("dz", 0),
                                          c["speed"], layer)
    else:
        # 默认水流: 轻微表层流
        grid.set_uniform_current(0.2, 0.0, 0.0, 0.1, "surface")

    # 加载障碍物
    if obstacles_path and os.path.exists(obstacles_path):
        with open(obstacles_path, "r") as f:
            obs_data = json.load(f)
        for obs in obs_data:
            grid.add_obstacle(obs["x"], obs["y"],
                              obs.get("z_min", 0), obs.get("z_max", 0))

    _apply_mission(grid, mission)
    return grid


def _apply_mission(grid, mission):
    if mission:
        grid.set_mission(
            mission.get("start"),
            mission.get("waypoints", []),
            mission.get("end"),
        )


# ═══════════════════════════════════════════════════════════
# 导出工具
# ═══════════════════════════════════════════════════════════

def export_to_json(grid: Water3DGrid, output_path: str):
    """将 Water3DGrid 导出为 JSON 文件"""
    grid.to_json(output_path)


def convert_file(input_path: str, output_path: str, **kwargs):
    """
    一步转换: 原始数据 → JSON

    Args:
        input_path: 原始数据文件
        output_path: 输出 JSON 文件路径
        **kwargs: 传递给 load_water_data 的参数
    """
    grid = load_water_data(input_path, **kwargs)
    grid.to_json(output_path)
    print(f"转换完成: {input_path} → {output_path}")
    print(f"  网格: {grid.nx}x{grid.ny}x{grid.nz} @ {grid.resolution}m")
    print(f"  水深范围: {grid.depth[grid.depth>0].min():.1f} ~ {grid.depth.max():.1f} m")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="水况数据适配器 - 转换原始数据为 JSON")
    ap.add_argument("input", help="输入文件")
    ap.add_argument("-o", "--output", default=None, help="输出 JSON 文件 (默认: 输入文件名.json)")
    ap.add_argument("-r", "--resolution", type=float, default=100.0, help="网格精度 (m/格)")
    ap.add_argument("-z", "--nz", type=int, default=10, help="深度层数")
    ap.add_argument("-c", "--currents", default=None, help="水流数据文件")
    ap.add_argument("-f", "--fmt", default=None, help="文件格式 (auto/csv/xyz/matrix)")
    ap.add_argument("--obs", default=None, help="障碍物 JSON 文件")

    args = ap.parse_args()
    out = args.output or os.path.splitext(args.input)[0] + ".json"
    convert_file(
        args.input, out,
        currents_path=args.currents,
        obstacles_path=args.obs,
        resolution=args.resolution,
        nz=args.nz,
        fmt=args.fmt,
    )
