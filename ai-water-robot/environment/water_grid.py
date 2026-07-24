"""
水面网格环境模块

网格编码：
    0 — 水域（可通行）
    1 — 障碍物（岩石、礁石等）
    2 — 浮标（巡检目标）
    3 — 垃圾（待清理的漂浮物）
    4 — 机器人当前位置

支持随机场景生成和三个预设 Demo 场景。
"""

import random

import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, List, Tuple, Optional

# 物体类型常量
WATER = 0
OBSTACLE = 1
BUOY = 2
TRASH = 3
ROBOT = 4

# 类型名称映射
TYPE_NAMES = {
    WATER: "水域",
    OBSTACLE: "障碍物",
    BUOY: "浮标",
    TRASH: "垃圾",
    ROBOT: "机器人",
}

# 物体显示符号（用于打印和可视化）
TYPE_SYMBOLS = {
    WATER: "·",
    OBSTACLE: "█",
    BUOY: "○",
    TRASH: "▲",
    ROBOT: "●",
}

# 物体颜色（用于 matplotlib 渲染）
TYPE_COLORS = {
    WATER: "#1a6b8a",    # 深蓝水面
    OBSTACLE: "#8b4513",  # 棕色
    BUOY: "#00ff88",      # 绿色
    TRASH: "#ffdd00",     # 黄色
    ROBOT: "#ffffff",     # 白色
}


class WaterGrid:
    """水面网格环境"""

    def __init__(self, size: int = 20):
        """
        初始化 N×N 水面网格

        Args:
            size: 网格边长（默认 20）
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.objects: Dict[Tuple[int, int], str] = {}  # {(row, col): type_name}
        self.robot_pos: Optional[Tuple[int, int]] = None
        self.home_pos: Optional[Tuple[int, int]] = None  # 码头/基地位置

    def place_object(self, row: int, col: int, obj_type: int) -> bool:
        """
        在指定位置放置物体

        Args:
            row, col: 坐标
            obj_type: 物体类型 (WATER/OBSTACLE/BUOY/TRASH/ROBOT)

        Returns:
            是否放置成功
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False

        if obj_type == ROBOT:
            # 机器人：先清除旧位置
            if self.robot_pos is not None:
                old_r, old_c = self.robot_pos
                self.grid[old_r, old_c] = WATER
                self.objects.pop((old_r, old_c), None)
            self.robot_pos = (row, col)
            self.home_pos = (row, col)  # 初始位置也是基地

        self.grid[row, col] = obj_type
        self.objects[(row, col)] = TYPE_NAMES[obj_type]
        return True

    def remove_object(self, row: int, col: int):
        """移除指定位置的物体，恢复为水域"""
        if 0 <= row < self.size and 0 <= col < self.size:
            self.grid[row, col] = WATER
            self.objects.pop((row, col), None)
            if self.robot_pos == (row, col):
                self.robot_pos = None

    def get_object_positions(self, obj_type: int) -> List[Tuple[int, int]]:
        """获取网格中所有指定类型物体的坐标列表"""
        positions = np.argwhere(self.grid == obj_type)
        return [(int(r), int(c)) for r, c in positions]

    def is_passable(self, row: int, col: int) -> bool:
        """检查某个位置是否可通行（水域或垃圾等可收集物）"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        # 可通行：水域、垃圾（可收集）、浮标（可接近）
        return self.grid[row, col] in (WATER, TRASH, BUOY)

    def is_obstacle(self, row: int, col: int) -> bool:
        """检查某个位置是否为障碍物"""
        if not (0 <= row < self.size and 0 <= col < self.size):
            return True  # 边界外视为障碍物
        return self.grid[row, col] == OBSTACLE

    def move_robot(self, new_row: int, new_col: int) -> bool:
        """
        移动机器人到新位置

        Returns:
            是否移动成功（目标位置可通行）
        """
        if not self.is_passable(new_row, new_col):
            return False
        if self.robot_pos is not None:
            old_r, old_c = self.robot_pos
            self.grid[old_r, old_c] = WATER
            self.objects.pop((old_r, old_c), None)
        return self.place_object(new_row, new_col, ROBOT)

    def random_scene(
        self,
        n_obstacles: int = 5,
        n_buoys: int = 3,
        n_trash: int = 6,
        robot_start: Tuple[int, int] = (0, 0),
    ):
        """
        随机生成一个水面场景

        Args:
            n_obstacles: 障碍物数量
            n_buoys: 浮标数量
            n_trash: 垃圾数量
            robot_start: 机器人起始位置
        """
        self.reset()

        # 放置机器人
        self.place_object(*robot_start, ROBOT)

        # 收集已占用的位置
        occupied = {robot_start}

        def random_pos():
            """生成未被占用的随机坐标"""
            while True:
                pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if pos not in occupied:
                    occupied.add(pos)
                    return pos

        # 放置障碍物
        for _ in range(n_obstacles):
            r, c = random_pos()
            self.place_object(r, c, OBSTACLE)

        # 放置浮标
        for _ in range(n_buoys):
            r, c = random_pos()
            self.place_object(r, c, BUOY)

        # 放置垃圾
        for _ in range(n_trash):
            r, c = random_pos()
            self.place_object(r, c, TRASH)

        return self

    def reset(self):
        """重置网格为全部水域"""
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.objects.clear()
        self.robot_pos = None
        self.home_pos = None

    def get_obstacle_grid(self) -> np.ndarray:
        """
        返回用于路径规划的障碍物网格
        0 = 可通行，1 = 不可通行（障碍物）

        注意：垃圾和浮标是可通行的（机器人可以接近并收集/检查）
        """
        obstacle_grid = np.zeros_like(self.grid)
        obstacle_grid[self.grid == OBSTACLE] = 1
        return obstacle_grid

    def __repr__(self) -> str:
        """打印网格的可视化表示"""
        lines = []
        for r in range(self.size):
            line = ""
            for c in range(self.size):
                val = self.grid[r, c]
                line += TYPE_SYMBOLS.get(val, "?") + " "
            lines.append(line)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 三个预设 Demo 场景
# ═══════════════════════════════════════════════════════════

def demo_scene_trash_cleanup(size: int = 20) -> WaterGrid:
    """
    Demo 场景 1：垃圾清理
    场景：河道中散落漂浮垃圾，有若干障碍物（礁石），机器人从码头出发清理垃圾。
    """
    grid = WaterGrid(size)

    # 机器人从左上角码头出发
    grid.place_object(1, 1, ROBOT)

    # 河道中间的障碍物（礁石带）
    obstacles = [
        (8, 5), (8, 6), (8, 7), (9, 5), (9, 6),
        (12, 12), (12, 13), (13, 12), (13, 13), (13, 14),
        (5, 15), (5, 16), (6, 15),
        (16, 3), (16, 4), (17, 3),
    ]
    for r, c in obstacles:
        grid.place_object(r, c, OBSTACLE)

    # 漂浮垃圾
    trash_positions = [
        (3, 6), (3, 10), (4, 16),
        (7, 10), (7, 13), (9, 14),
        (11, 3), (11, 8), (14, 8),
        (15, 15), (16, 9), (17, 17),
    ]
    for r, c in trash_positions:
        grid.place_object(r, c, TRASH)

    # 浮标（需要避开的固定设施）
    buoys = [(5, 9), (10, 2), (15, 6)]
    for r, c in buoys:
        grid.place_object(r, c, BUOY)

    return grid


def demo_scene_buoy_inspection(size: int = 20) -> WaterGrid:
    """
    Demo 场景 2：浮标巡检
    场景：水域中有多个浮标需要巡检，机器人需从码头出发依次检查并返回。
    """
    grid = WaterGrid(size)

    # 机器人从右下角码头出发
    grid.place_object(18, 18, ROBOT)

    # 少量障碍物
    obstacles = [
        (4, 4), (4, 5), (5, 4), (5, 5),
        (15, 8), (15, 9), (16, 8),
        (8, 15), (9, 15), (9, 16),
    ]
    for r, c in obstacles:
        grid.place_object(r, c, OBSTACLE)

    # 需要巡检的浮标（编号便于指令引用）
    buoys = [
        (2, 8),  # 1号浮标
        (8, 2),  # 2号浮标
        (12, 12),  # 3号浮标
        (2, 16),  # 4号浮标
    ]
    for r, c in buoys:
        grid.place_object(r, c, BUOY)

    return grid


def demo_scene_patrol(size: int = 20) -> WaterGrid:
    """
    Demo 场景 3：水域巡逻
    场景：大范围水域巡逻，目标是在巡逻过程中发现异常物体（模拟垃圾/漂浮障碍）。
    """
    grid = WaterGrid(size)

    # 机器人从左边中间出发
    grid.place_object(10, 1, ROBOT)

    # 分散的障碍物
    obstacles = [
        (3, 8), (4, 8), (5, 8),
        (15, 12), (16, 12), (17, 12),
        (6, 16), (7, 16), (8, 16),
        (14, 4), (15, 4),
    ]
    for r, c in obstacles:
        grid.place_object(r, c, OBSTACLE)

    # 浮标
    buoys = [(4, 14), (9, 5), (16, 16)]
    for r, c in buoys:
        grid.place_object(r, c, BUOY)

    # 异常物体（模拟需要发现的垃圾/漂浮物）
    trash_items = [
        (2, 12), (3, 17), (6, 2),
        (8, 11), (11, 7), (13, 14),
        (15, 16), (17, 3), (18, 10),
    ]
    for r, c in trash_items:
        grid.place_object(r, c, TRASH)

    return grid


# 预设场景注册表（供 Gradio 下拉菜单使用）
PRESET_SCENES = {
    "垃圾清理 -- 清理河道漂浮垃圾，避开障碍物": demo_scene_trash_cleanup,
    "浮标巡检 -- 依次检查浮标并返回码头": demo_scene_buoy_inspection,
    "水域巡逻 -- 巡逻发现异常物体并标记报告": demo_scene_patrol,
}


def render_grid_as_image(
    grid: WaterGrid,
    img_size: int = 640,
    show_labels: bool = True,
) -> Image.Image:
    """
    将 WaterGrid 渲染为 PIL Image，供 YOLO 检测和可视化使用

    Args:
        grid: WaterGrid 实例
        img_size: 输出图像尺寸（正方形）
        show_labels: 是否在物体位置显示文字标签

    Returns:
        PIL Image
    """
    img = Image.new("RGB", (img_size, img_size), TYPE_COLORS[WATER])
    draw = ImageDraw.Draw(img)

    cell = img_size / grid.size

    # 绘制网格线
    for i in range(grid.size + 1):
        pos = int(i * cell)
        draw.line([(pos, 0), (pos, img_size)], fill="#2a8aaa", width=1)
        draw.line([(0, pos), (img_size, pos)], fill="#2a8aaa", width=1)

    # 绘制物体
    for r in range(grid.size):
        for c in range(grid.size):
            val = grid.grid[r, c]
            if val == WATER:
                continue

            x1 = int(c * cell) + 2
            y1 = int(r * cell) + 2
            x2 = int((c + 1) * cell) - 2
            y2 = int((r + 1) * cell) - 2

            color = TYPE_COLORS.get(val, "#ffffff")

            if val == OBSTACLE:
                # 障碍物 — 填充矩形
                draw.rectangle([x1, y1, x2, y2], fill=color)
            elif val == BUOY:
                # 浮标 — 圆形
                draw.ellipse([x1, y1, x2, y2], fill=color, outline="#00aa55", width=2)
            elif val == TRASH:
                # 垃圾 — 小圆
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r_size = int(cell * 0.35)
                draw.ellipse(
                    [cx - r_size, cy - r_size, cx + r_size, cy + r_size],
                    fill=color, outline="#ccaa00", width=1,
                )
            elif val == ROBOT:
                # 机器人 — 圆形
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r_size = int(cell * 0.4)
                draw.ellipse(
                    [cx - r_size, cy - r_size, cx + r_size, cy + r_size],
                    fill=color, outline="#00aaff", width=2,
                )
                # 机器人方向指示（小三角形）
                draw.polygon([
                    (cx, cy - r_size), (cx - r_size, cy + r_size), (cx + r_size, cy + r_size),
                ], fill="#00aaff")

            # 文字标签
            if show_labels and val != WATER:
                label = TYPE_NAMES.get(val, "")[0]  # 只取第一个字
                draw.text((x1 + 2, y1 + 2), label, fill="#ffffff")

    return img
