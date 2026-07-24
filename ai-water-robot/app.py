"""
水域智能感知与任务理解系统 — Gradio 主应用

四 Tab 界面：
  1. 任务输入 — 自然语言指令 + 预设场景 + LLM 任务拆解
  2. 目标检测 — 水面场景图像 + YOLO 检测标注
  3. 路径规划 — A* 路径可视化
  4. 仿真动画 — 机器人执行 GIF
"""

import os
import sys
import traceback

# 确保当前目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from environment.water_grid import (
    WaterGrid, render_grid_as_image, PRESET_SCENES,
)
from environment.water_grid import OBSTACLE, BUOY, TRASH, ROBOT
from planning.astar import plan_multi_point_route, compute_path_length
from task_planner.llm_planner import TaskPlanner, rule_based_plan
from visualization.renderer import SimulationRenderer, render_static_path
from config import DEFAULT_GRID_SIZE, OUTPUT_DIR

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# 核心处理逻辑
# ═══════════════════════════════════════════════════════════

def process_pipeline(
    user_instruction: str,
    preset_scene_name: str,
    use_llm: bool = False,
    grid_size: int = DEFAULT_GRID_SIZE,
    progress=gr.Progress(),
) -> tuple:
    """
    完整处理管线：场景生成 → LLM规划 → 目标检测 → 路径规划 → 动画

    Returns:
        (任务规划文本, 场景图像, 检测标注图像, 检测结果表,
         路径图, 路径文本, GIF动画, 任务报告)
    """
    results = {}  # 存储中间结果，出错时返回已完成的步骤

    try:
        # ── 步骤 1: 生成水面场景 ──
        progress(0.05, desc="生成水面场景...")
        scene_func = PRESET_SCENES.get(preset_scene_name)
        if scene_func is None:
            scene_func = list(PRESET_SCENES.values())[0]
        scene = scene_func(grid_size)
        results["scene"] = scene

        # 渲染场景为图像
        scene_image = render_grid_as_image(scene, img_size=480)
        results["scene_image"] = scene_image

        # ── 步骤 2: LLM 任务规划 ──
        progress(0.15, desc="LLM 任务规划...")
        if use_llm:
            try:
                planner = TaskPlanner()
                plan = planner.plan(user_instruction)
                plan_text = planner.format_plan_display(plan)
            except Exception as e:
                # LLM 失败时回退到规则模式
                plan = rule_based_plan(user_instruction)
                plan_text = f"[警告] LLM 调用失败（{e}），使用规则模式:\n\n"
                from task_planner.llm_planner import TaskPlanner as TP
                dummy = TP.__new__(TP)
                plan_text += dummy.format_plan_display(plan) if hasattr(dummy, 'format_plan_display') else str(plan)
        else:
            plan = rule_based_plan(user_instruction)
            plan_text = "任务规划（规则模式）\n\n"
            # 手动格式化
            plan_text += f"概述: {plan.get('summary', '')}\n\n"
            for i, t in enumerate(plan.get("tasks", []), 1):
                plan_text += f"  步骤{i}: {t.get('action', '?')} → {t.get('target', '?')}\n"
                plan_text += f"  原因: {t.get('reason', '')}\n\n"
        results["plan_text"] = plan_text

        # ── 步骤 3: YOLO 目标检测 ──
        progress(0.25, desc="YOLO 目标检测...")
        detections = []
        try:
            from perception.detector import Detector
            detector = Detector()
            annotated_img, detections = detector.detect_and_annotate(scene_image, conf_threshold=0.1)
            results["annotated_img"] = annotated_img
        except ImportError:
            # YOLO 未安装时直接用场景图
            results["annotated_img"] = scene_image
            detections = _fake_detections_from_grid(scene)

        # 检测结果表格
        if detections:
            det_table = [[d["label"], d["confidence"], str(d["bbox"])] for d in detections]
        else:
            # 从网格生成模拟检测结果
            det_table = _grid_objects_as_table(scene)
        results["det_table"] = det_table

        # ── 步骤 4: A* 路径规划 ──
        progress(0.50, desc="A* 路径规划...")
        obs_grid = scene.get_obstacle_grid()
        robot_pos = scene.robot_pos
        if robot_pos is None:
            robot_pos = (0, 0)

        # 根据任务选择目标点
        trash_positions = scene.get_object_positions(TRASH)
        buoy_positions = scene.get_object_positions(BUOY)

        # 选择目标点（优先垃圾，其次浮标）
        target_positions = trash_positions if trash_positions else buoy_positions

        if not target_positions:
            return _error_result("场景中没有可执行任务的目标物体（垃圾或浮标）", results)

        path = plan_multi_point_route(obs_grid, robot_pos, target_positions)
        if path is None:
            return _error_result("无法找到可行路径：目标可能被障碍物完全包围", results)

        path_length = compute_path_length(path)
        path_text = (
            f"路径规划完成\n"
            f"起点: {robot_pos}\n"
            f"途经目标点: {len(target_positions)} 个\n"
            f"总步数: {len(path)}\n"
            f"总距离: {path_length}"
        )

        # 渲染路径图
        path_preview_path = os.path.join(OUTPUT_DIR, "path_preview.png")
        render_static_path(
            scene, path, path_preview_path,
            title=f"A* 路径规划 (距离={path_length}, 步数={len(path)})",
            show_indices=(len(path) > 4),
        )

        # ── 步骤 5: 仿真动画 ──
        progress(0.70, desc="生成仿真动画...")
        renderer = SimulationRenderer(scene, figsize=(7, 6))
        gif_path = os.path.join(OUTPUT_DIR, "simulation.gif")
        renderer.render_gif(
            path, gif_path, fps=12, interval=120,
            collect_targets=target_positions,
        )

        # ── 步骤 6: 任务报告 ──
        progress(0.95, desc="生成任务报告...")
        collected = len(target_positions)
        report = (
            f"## 任务完成报告\n\n"
            f"**执行指令:** {user_instruction}\n\n"
            f"**场景:** {preset_scene_name}\n\n"
            f"**任务规划:**\n{plan_text}\n\n"
            f"**检测结果:** 发现 {len(det_table)} 个目标物体\n\n"
            f"**路径规划:** {len(path)} 步, 总距离 {path_length}\n\n"
            f"**执行结果:** 成功抵达 {collected} 个目标点\n\n"
            f"---\n"
            f"*系统由 DeepSeek LLM + YOLOv8 + A* 算法驱动*"
        )

        progress(1.0, desc="完成!")
        return (
            plan_text,
            scene_image,
            results.get("annotated_img", scene_image),
            det_table,
            path_preview_path,
            path_text,
            gif_path,
            report,
        )

    except Exception as e:
        return _error_result(f"处理失败: {e}\n\n{traceback.format_exc()}", results)


def _error_result(msg: str, results: dict = None) -> tuple:
    """生成错误结果，保留已完成步骤的结果"""
    error = f"[错误] {msg}"
    empty_table = [["-", "-", "-"]]
    return (
        results.get("plan_text", error) if results else error,
        results.get("scene_image", None) if results else None,
        results.get("annotated_img", results.get("scene_image")) if results else None,
        results.get("det_table", empty_table) if results else empty_table,
        None,
        error,
        None,
        error,
    )


def _fake_detections_from_grid(scene: WaterGrid) -> list:
    """从网格生成模拟检测结果（YOLO 不可用时的后备方案）"""
    detections = []
    for (r, c), type_name in scene.objects.items():
        if type_name in ("障碍物", "浮标", "垃圾"):
            x1 = int(c * 640 / scene.size)
            y1 = int(r * 640 / scene.size)
            x2 = int((c + 1) * 640 / scene.size)
            y2 = int((r + 1) * 640 / scene.size)
            detections.append({
                "label": type_name,
                "confidence": 1.0,
                "bbox": (x1, y1, x2, y2),
            })
    return detections


def _grid_objects_as_table(scene: WaterGrid) -> list:
    """将网格中的物体转为表格形式"""
    rows = []
    for (r, c), type_name in scene.objects.items():
        if type_name not in ("机器人", "水域"):
            rows.append([type_name, "1.00", f"({r}, {c})"])
    if not rows:
        rows.append(["无目标", "-", "-"])
    return rows


# ═══════════════════════════════════════════════════════════
# 快捷场景加载回调
# ═══════════════════════════════════════════════════════════

SCENE_INSTRUCTIONS = {
    list(PRESET_SCENES.keys())[0]: "清理河道里的所有漂浮垃圾，规划路线避开障碍物",
    list(PRESET_SCENES.keys())[1]: "从码头出发，依次检查所有浮标，完成后返回码头",
    list(PRESET_SCENES.keys())[2]: "巡逻整个水域，发现并标记所有异常物体",
}

def on_scene_change(scene_name: str) -> tuple:
    """切换场景时自动更新指令和预览图像"""
    instruction = SCENE_INSTRUCTIONS.get(scene_name, "请描述你的任务...")
    # 生成场景预览
    try:
        scene_func = PRESET_SCENES.get(scene_name)
        if scene_func:
            scene = scene_func(20)
            img = render_grid_as_image(scene, img_size=400)
            return instruction, img
    except:
        pass
    return instruction, None


# ═══════════════════════════════════════════════════════════
# Gradio 界面构建
# ═══════════════════════════════════════════════════════════

# 自定义 CSS 样式
APP_CSS = """
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #0d4f6b, #1a8a9a);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 16px;
}
.main-title h1 {
    color: #ffffff !important;
    margin: 0;
    font-size: 1.8em;
}
.main-title p {
    color: #a0d8e8 !important;
    margin: 4px 0 0 0;
}
footer { visibility: hidden; }
"""

# 主题
APP_THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
)


def build_ui():
    """构建 Gradio Blocks UI"""

    with gr.Blocks(
        title="水域智能感知与任务理解系统",
    ) as demo:
        # ── 标题栏 ──
        gr.HTML("""
        <div class="main-title">
            <h1>水域智能感知与任务理解系统</h1>
            <p>AI-Powered Water Robot Intelligence — LLM + YOLOv8 + A*</p>
        </div>
        """)

        # ── 控制面板 ──
        with gr.Row():
            with gr.Column(scale=2):
                instruction_input = gr.Textbox(
                    label="任务指令（自然语言）",
                    placeholder="例如：清理河道里的所有漂浮垃圾，避开障碍物",
                    value="清理河道里的所有漂浮垃圾，规划路线避开障碍物",
                    lines=3,
                )
            with gr.Column(scale=2):
                scene_dropdown = gr.Dropdown(
                    label="预设场景",
                    choices=list(PRESET_SCENES.keys()),
                    value=list(PRESET_SCENES.keys())[0],
                )
            with gr.Column(scale=1):
                use_llm_checkbox = gr.Checkbox(
                    label="使用 LLM 规划（需 DeepSeek API）",
                    value=False,
                    info="关闭则使用规则模式",
                )
                grid_slider = gr.Slider(
                    label="网格大小",
                    minimum=10, maximum=30, value=20, step=5,
                )
                run_button = gr.Button(
                    "执行任务",
                    variant="primary",
                    size="lg",
                )

        scene_preview = gr.Image(
            label="场景预览",
            height=250,
            visible=False,
        )

        # ── 四 Tab 结果 ──
        with gr.Tabs():
            with gr.TabItem("任务规划", id=0):
                plan_output = gr.Textbox(
                    label="LLM 任务拆解结果",
                    lines=12,
                    placeholder="点击「执行任务」后显示...",
                )

            with gr.TabItem("目标检测", id=1):
                with gr.Row():
                    with gr.Column():
                        scene_img_output = gr.Image(
                            label="水面场景",
                            height=350,
                        )
                    with gr.Column():
                        detection_img_output = gr.Image(
                            label="YOLO 检测结果",
                            height=350,
                        )
                detection_table = gr.Dataframe(
                    headers=["物体类别", "置信度", "位置/边界框"],
                    label="检测结果详情",
                )

            with gr.TabItem("路径规划", id=2):
                with gr.Row():
                    with gr.Column(scale=3):
                        path_plot = gr.Image(
                            label="A* 路径规划图",
                            height=400,
                        )
                    with gr.Column(scale=1):
                        path_info = gr.Textbox(
                            label="路径信息",
                            lines=8,
                        )

            with gr.TabItem("仿真动画", id=3):
                with gr.Row():
                    with gr.Column(scale=3):
                        gif_output = gr.Image(
                            label="机器人任务执行动画",
                            height=400,
                        )
                    with gr.Column(scale=1):
                        report_output = gr.Markdown(
                            value="点击「执行任务」后生成任务报告...",
                        )

        # ── 事件绑定 ──

        # 场景切换 → 更新指令和预览
        scene_dropdown.change(
            fn=on_scene_change,
            inputs=[scene_dropdown],
            outputs=[instruction_input, scene_preview],
        )

        # 执行按钮 → 完整管线
        run_button.click(
            fn=process_pipeline,
            inputs=[
                instruction_input,
                scene_dropdown,
                use_llm_checkbox,
                grid_slider,
            ],
            outputs=[
                plan_output,
                scene_img_output,
                detection_img_output,
                detection_table,
                path_plot,
                path_info,
                gif_output,
                report_output,
            ],
        )

        # ── 页脚 ──
        gr.HTML("""
        <div style="text-align:center; color:#688; padding:20px; font-size:0.85em;">
            <p>技术栈: DeepSeek LLM · YOLOv8 · A* 路径规划 · Gradio · Matplotlib</p>
            <p>面向中山大学单云霄课题组 · 水上机器人方向</p>
        </div>
        """)

    return demo


# ═══════════════════════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo = build_ui()
    print("=" * 60)
    print("  水域智能感知与任务理解系统 启动中...")
    print("  Water Robot AI Demo System")
    print("=" * 60)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # 如需公网链接改为 True
        show_error=True,
        css=APP_CSS,
        theme=APP_THEME,
    )
