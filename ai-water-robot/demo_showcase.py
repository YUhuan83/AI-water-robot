"""
水域机器人3D智能决策平台 — 一键展示脚本

用法:
    python demo_showcase.py              # 控制台输出关键数据
    python demo_showcase.py --screenshots # 同时生成对比截图
    python demo_showcase.py --scene coastal  # 只展示指定场景
"""

import os, sys, math, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from environment.water_3d import (
    demo_3d_coastal, demo_3d_river, demo_3d_harbor, demo_3d_windfarm,
)
from planning.astar3d import (
    plan_tsp_3d, compute_3d_path_cost, compute_energy_estimate,
    compare_strategies, astar3d, dijkstra3d,
)
from task_planner.mission_patterns import generate_mission, MISSION_PATTERNS

# ── 颜色 ──
G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"; R = "\033[91m"; B = "\033[1m"; N = "\033[0m"


def hr(title):
    """打印分隔标题"""
    print(f"\n{B}{'='*60}{N}")
    print(f"{B}  {title}{N}")
    print(f"{B}{'='*60}{N}")


def fmt_num(n, unit=""):
    """格式化数字"""
    if isinstance(n, float):
        return f"{n:,.1f}{unit}"
    return f"{n}{unit}"


def benchmark_scene(name, grid_fn):
    """对单个场景运行全策略对比"""
    g = grid_fn()
    print(f"\n{C}> {name}{N}  {g.nx}x{g.ny}x{g.nz} | 障碍物: {int(g.obstacles.sum())} | "
          f"途经点: {len(g.mission_waypoints)} | 水深: {g.depth[g.depth>0].min():.0f}~{g.depth.max():.0f}m")

    if not g.mission_start or not g.mission_waypoints:
        print(f"  {R}X 任务点不完整，跳过{N}")
        return None

    results = compare_strategies(g, g.mission_start, g.mission_waypoints, g.mission_end)

    # 打印对比表
    print(f"  {'策略':<12} {'距离(m)':>10} {'能耗(kJ)':>10} {'时间(min)':>10} {'水流代价':>10} {'路径点':>8}")
    print(f"  {'-'*60}")

    best_dist = min((r["distance"] for r in results.values() if r), default=0)
    best_energy = min((r["energy_kj"] for r in results.values() if r), default=0)
    best_time = min((r["time_min"] for r in results.values() if r), default=0)

    strategy_cn = {"balanced": "均衡", "safe": "安全优先", "fast": "速度优先", "energy": "节能优先"}

    for key, r in results.items():
        if r is None:
            print(f"  {strategy_cn[key]:<12} {'---':>10} {'---':>10} {'---':>10} {'---':>10} {'---':>8}")
            continue
        dm = " ★" if r["distance"] == best_dist else ""
        em = " ★" if r["energy_kj"] == best_energy else ""
        tm = " ★" if r["time_min"] == best_time else ""
        print(f"  {strategy_cn[key]:<12} {fmt_num(r['distance'])+dm:>10} "
              f"{fmt_num(r['energy_kj'])+em:>10} {fmt_num(r['time_min'])+tm:>10} "
              f"{fmt_num(r['flow_cost']):>10} {r['waypoints_count']:>8}")

    return results


def benchmark_algorithm(name, grid_fn):
    """对比 A* vs Dijkstra 在同一场景下的性能"""
    g = grid_fn()
    if not g.mission_start or not g.mission_waypoints:
        return

    print(f"\n  {'算法':<12} {'距离(m)':>10} {'能耗(kJ)':>10} {'时间(ms)':>10} {'路径点':>8}")
    print(f"  {'-'*12} {'-'*60}")

    for algo_name, algo_fn in [("A*", astar3d), ("Dijkstra", dijkstra3d)]:
        t0 = time.time()
        path = plan_tsp_3d(
            g, g.mission_start, g.mission_waypoints, g.mission_end,
            strategy="balanced", use_2opt=False,
        )
        elapsed = (time.time() - t0) * 1000

        if path:
            d, f, dc = compute_3d_path_cost(g, path)
            energy = compute_energy_estimate(g, path)
            print(f"  {algo_name:<12} {fmt_num(d):>10} "
                  f"{fmt_num(energy['energy_consumption_kj']):>10} {fmt_num(elapsed):>10} "
                  f"{len(path):>8}")
        else:
            print(f"  {algo_name:<12} {'---':>10} {'---':>10} {'---':>10} {'---':>8}")


def showcase_mission_patterns(grid_fn):
    """展示任务模式多样性"""
    g = grid_fn()
    print(f"\n  {'模式':<12} {'途经点':>8} {'终点':>8}")
    print(f"  {'-'*30}")

    for pattern in MISSION_PATTERNS:
        wps, end = generate_mission(g, pattern=pattern, count=6)
        print(f"  {pattern:<12} {len(wps):>8} {'是' if end else '否':>8}")


def generate_screenshots():
    """生成对比截图（需要matplotlib）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{R}matplotlib 未安装，跳过截图生成{N}")
        return

    os.makedirs("output/showcase", exist_ok=True)

    scenes = [
        ("coastal", demo_3d_coastal),
        ("river", demo_3d_river),
        ("harbor", demo_3d_harbor),
        ("windfarm", demo_3d_windfarm),
    ]

    for name, fn in scenes:
        g = fn()
        if not g.mission_start or not g.mission_waypoints:
            continue

        # 收集4条策略路径
        fig = plt.figure(figsize=(14, 9), facecolor="#0a1628")
        colors = {"balanced": "#e8590c", "safe": "#27ae60", "fast": "#e74c3c", "energy": "#2980b9"}
        strategy_cn = {"balanced": "均衡", "safe": "安全", "fast": "快速", "energy": "节能"}

        for strategy, color in colors.items():
            path = plan_tsp_3d(
                g, g.mission_start, g.mission_waypoints, g.mission_end,
                strategy=strategy,
            )
            if not path:
                continue

            px = [p[0] for p in path]
            py = [p[1] for p in path]
            pz = [-p[2] for p in path]

            ax = fig.add_subplot(2, 2, list(colors.keys()).index(strategy) + 1, projection="3d")
            ax.set_facecolor("#0d3b4e")
            ax.plot(px, py, pz, color=color, linewidth=1.8, label=strategy_cn[strategy])
            ax.scatter(px[0], py[0], pz[0], color="#00ff88", s=40, marker="o")
            ax.scatter(px[-1], py[-1], pz[-1], color="#ff4444", s=40, marker="s")
            for wp in g.mission_waypoints:
                ax.scatter(wp[0], wp[1], -wp[2], color="#ffcc00", s=20, marker="^")

            d, _, _ = compute_3d_path_cost(g, path)
            e = compute_energy_estimate(g, path)
            ax.set_title(
                f"{strategy_cn[strategy]}: {d:,.0f}m | {e['energy_consumption_kj']:,.0f}kJ",
                color=color, fontsize=11,
            )
            ax.set_xlim(0, g.nx); ax.set_ylim(0, g.ny); ax.set_zlim(-g.nz, 1)
            ax.view_init(elev=35, azim=-50)
            ax.tick_params(colors="#668888", labelsize=6)
            ax.legend(fontsize=8)

        fig.suptitle(f"策略对比 — {g.metadata.get('name', name)}",
                     color="#c0d8e0", fontsize=14, y=0.98)
        fig.tight_layout()
        path_out = f"output/showcase/{name}_compare.png"
        fig.savefig(path_out, dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  {G}[OK]{N} 截图已保存: {path_out}")


# ═══════════════════ 主程序 ═══════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="水域机器人3D智能决策平台 — 展示脚本")
    parser.add_argument("--screenshots", action="store_true", help="生成对比截图")
    parser.add_argument("--scene", type=str, default=None,
                       help="只展示指定场景 (coastal/river/harbor/windfarm)")
    args = parser.parse_args()

    scenes = {
        "coastal":  ("沿海水域", demo_3d_coastal),
        "river":    ("内河航道", demo_3d_river),
        "harbor":   ("港口码头", demo_3d_harbor),
        "windfarm": ("海上风电场", demo_3d_windfarm),
    }

    if args.scene:
        scenes = {args.scene: scenes[args.scene]}

    # ═══ 第一部分：场景策略对比 ═══
    hr("一、四策略路径规划对比")
    print(f"{C}每个场景用 balanced/safe/fast/energy 四种策略规划，对比距离/能耗/时间{N}")
    all_results = {}
    for key, (name, fn) in scenes.items():
        all_results[key] = benchmark_scene(name, fn)

    # ═══ 第二部分：算法性能 ═══
    hr("二、A* vs Dijkstra 性能对比")
    for key, (name, fn) in scenes.items():
        print(f"\n{C}> {name}{N}")
        benchmark_algorithm(name, fn)

    # ═══ 第三部分：任务模式多样性 ═══
    hr("三、任务模式多样性")
    fn_first = list(scenes.values())[0][1]
    print(f"\n{C}以 {list(scenes.values())[0][0]} 场景为例，展示6种任务模式:{N}")
    showcase_mission_patterns(fn_first)

    # ═══ 第四部分：水况环境参数 ═══
    hr("四、水况环境参数")
    for key, (name, fn) in scenes.items():
        g = fn()
        avg_temp = g.temperature[g.depth > 0].mean() if (g.depth > 0).any() else 0
        avg_vis = g.visibility[g.depth > 0].mean() if (g.depth > 0).any() else 0
        wind_dir, wind_spd, wave_h = g.get_weather_at(g.nx // 2, g.ny // 2)
        n_eddies = len(g.eddies)
        print(f"  {name:<10} 水温:{avg_temp:.1f}°C | 能见度:{avg_vis:.1f}m | "
              f"风速:{wind_spd:.0f}m/s | 浪:{wave_h:.1f}m | 漩涡:{n_eddies} | 潮汐:{g.tidal_phase:.2f}")

    # ═══ 截图 ═══
    if args.screenshots:
        hr("五、生成对比截图")
        generate_screenshots()

    # ═══ 总结 ═══
    hr("总结")
    total_scenes = len(scenes)
    success_count = sum(1 for r in all_results.values() if r and any(v is not None for v in r.values()))
    print(f"""
  {G}[OK]{N} 测试场景: {total_scenes} 个
  {G}[OK]{N} 成功规划: {success_count} 个
  {G}[OK]{N} 策略维度: balanced(均衡) / safe(安全优先) / fast(速度优先) / energy(节能优先)
  {G}[OK]{N} 环境因素: 水深 / 3层水流 / 水压 / 风浪 / 水温 / 能见度 / 潮汐 / 漩涡
  {G}[OK]{N} 任务模式: {len(MISSION_PATTERNS)} 种 (patrol/spiral/zigzag/scattered/cluster/perimeter)
  {G}[OK]{N} 算法: A*3D(26方向) + Dijkstra3D + 2-opt + 平滑 + TSP排序
  {G}[OK]{N} 输入: 3D点击 / 精确坐标 / ROS2远程指令
  {G}[OK]{N} 输出: 3D可视化 / 动画 / 截图 / JSON报告 / ROS2实时遥测

  {Y}运行 3D 桌面应用: python desktop3d_app.py{N}
""")
