"""
YOLO 目标检测模块

封装 ultralytics YOLOv8，用于水面场景目标检测。
使用预训练权重，不做微调 — Demo 的重点是系统集成，不是检测精度。
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw

from config import YOLO_MODEL_NAME

# 物体标签映射（YOLO COCO 类别中的相关标签 → 我们的网格类型）
COCO_LABELS_OF_INTEREST = {
    "boat": "boat",
    "person": "person",
    "bird": "bird",
    "surfboard": "surfboard",
    "kite": "kite",
}


class Detector:
    """水面目标检测器（YOLOv8 封装）"""

    def __init__(self, model_name: str = YOLO_MODEL_NAME):
        """
        初始化检测器，加载 YOLO 模型

        Args:
            model_name: YOLO 模型名称，如 "yolov8n.pt"
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        """懒加载 YOLO 模型（首次使用时才加载，避免导入时卡顿）"""
        if self._loaded:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            self._loaded = True
        except ImportError:
            raise ImportError(
                "请先安装 ultralytics: pip install ultralytics"
            )

    def detect_on_image(
        self, image: Image.Image, conf_threshold: float = 0.15
    ) -> List[Dict]:
        """
        对 PIL Image 执行目标检测

        Args:
            image: 输入图像（PIL Image）
            conf_threshold: 置信度阈值（预训练模型对合成图效果一般，设低一点）

        Returns:
            检测结果列表: [{"label": str, "confidence": float, "bbox": (x1,y1,x2,y2)}, ...]
        """
        self._ensure_loaded()

        results = self.model(image, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = self.model.names.get(cls_id, f"class_{cls_id}")
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                })

        return detections

    def detect_and_annotate(
        self, image: Image.Image, conf_threshold: float = 0.15
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        执行检测并返回带标注框的图像

        Args:
            image: 输入图像
            conf_threshold: 置信度阈值

        Returns:
            (标注后的图像, 检测结果列表)
        """
        detections = self.detect_on_image(image, conf_threshold)

        # 复制图像用于标注
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # 简单的颜色映射
        colors = {
            "boat": "#00ff00",
            "person": "#ff8800",
            "bird": "#00ccff",
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            color = colors.get(label, "#ff0000")

            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            # 画标签
            draw.text((x1, max(0, y1 - 15)), f"{label} {conf:.2f}", fill=color)

        return annotated, detections

    def grid_coords_to_image(
        self, grid_row: int, grid_col: int, grid_size: int, img_size: int = 640
    ) -> Tuple[int, int, int, int]:
        """
        将网格坐标转换为图像边界框

        Args:
            grid_row, grid_col: 网格坐标
            grid_size: 网格大小（如 20×20）
            img_size: 输出图像尺寸

        Returns:
            (x1, y1, x2, y2) 像素坐标
        """
        cell = img_size / grid_size
        x1 = int(grid_col * cell)
        y1 = int(grid_row * cell)
        x2 = int((grid_col + 1) * cell)
        y2 = int((grid_row + 1) * cell)
        return x1, y1, x2, y2


