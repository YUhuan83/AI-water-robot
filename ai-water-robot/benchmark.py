"""
算法基准对比 — A*3D vs Dijkstra3D 在4个场景×4种策略下的性能

用法:
    python benchmark.py           # 控制台输出表格
    python benchmark.py --csv     # 输出CSV格式
"""

import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from environment.water_3d import (
    demo_3d_coastal, demo_3d_river, demo_3d_harbor, demo_3d_windfarm,
    Water3DGrid,
)
from planning.astar3d import (
    astar3d, dijkstra3d, plan_tsp_3d,
    compute_3d_path_cost, compute_energy_estimate,
    STRATEGY_WEIGHTS,
)


def run_benchmark():
    """运行完整基准测试"""
    scenes = {
        "沿海水域": demo_3d_coastal,
        "内河航道": demo_3d_river,
        "港口码头": demo_3d_harbor,
        "海上风电场": demo_3d_windfarm,
    }
    strategies = ["balanced", "safe", "fast", "energy"]

    results = []

    for scene_name, scene_fn in scenes.items():
        g = scene_fn()
        start = g.mission_start
        waypoints = g.mission_waypoints
        end = g.mission_end

        if not start or not waypoints:
            print(f"  ⊘ {scene_name}: 任务点不完整，跳过")
            continue

        n_obs = int(g.obstacles.sum())
        n_wp = len(waypoints)

        for strategy in strategies:
            for algo_name, algo_use_dijkstra in [("A*3D", False), ("Dijkstra3D", True)]:
                # 计时
                t0 = time.perf_counter()

                if algo_use_dijkstra:
                    # 纯Dijkstra模式：使用dijkstra3d逐段规划
                    path = plan_tsp_3d(
                        g, start, waypoints, end,
                        strategy=strategy, use_2opt=False,
                        use_dijkstra_for_short=True,
                    )
                else:
                    # 纯A*模式
                    path = plan_tsp_3d(
                        g, start, waypoints, end,
                        strategy=strategy, use_2opt=False,
                        use_dijkstra_for_short=False,
                    )

                elapsed_ms = (time.perf_counter() - t0) * 1000

                if path is None:
                    results.append({
                        "scene": scene_name, "algo": algo_name, "strategy": strategy,
                        "distance_m": None, "energy_kj": None, "time_min": None,
                        "flow_cost": None, "path_points": 0, "elapsed_ms": elapsed_ms,
                        "n_obstacles": n_obs, "n_waypoints": n_wp,
                        "success": False,
                    })
                    continue

                d, f, dc = compute_3d_path_cost(g, path)
                energy = compute_energy_estimate(g, path)

                results.append({
                    "scene": scene_name, "algo": algo_name, "strategy": strategy,
                    "distance_m": d, "energy_kj": energy["energy_consumption_kj"],
                    "time_min": energy["estimated_time_min"], "flow_cost": f,
                    "path_points": len(path), "elapsed_ms": elapsed_ms,
                    "n_obstacles": n_obs, "n_waypoints": n_wp,
                    "success": True,
                })

    return results


def print_table(results):
    """打印格式化表格"""
    strategy_cn = {"balanced": "均衡", "safe": "安全", "fast": "快速", "energy": "节能"}

    # 按场景分组
    scenes_order = ["沿海水域", "内河航道", "港口码头", "海上风电场"]
    for scene in scenes_order:
        scene_results = [r for r in results if r["scene"] == scene]
        if not scene_results:
            continue

        n_obs = scene_results[0]["n_obstacles"]
        n_wp = scene_results[0]["n_waypoints"]
        print(f"\n{'='*100}")
        print(f"  {scene}  ({n_obs}个障碍物, {n_wp}个途经点)")
        print(f"{'='*100}")
        print(f"  {'算法':<14} {'策略':<10} {'距离(m)':>12} {'能耗(kJ)':>12} {'时间(min)':>12} {'水流代价':>12} {'路径点':>8} {'耗时(ms)':>10} {'成功':>6}")
        print(f"  {'-'*98}")

        for r in scene_results:
            if r["success"]:
                print(f"  {r['algo']:<14} {strategy_cn.get(r['strategy'], r['strategy']):<10} "
                      f"{r['distance_m']:>12,.0f} {r['energy_kj']:>12,.0f} "
                      f"{r['time_min']:>12,.1f} {r['flow_cost']:>12,.0f} "
                      f"{r['path_points']:>8} {r['elapsed_ms']:>10,.1f} {'✓':>6}")
            else:
                print(f"  {r['algo']:<14} {strategy_cn.get(r['strategy'], r['strategy']):<10} "
                      f"{'---':>12} {'---':>12} {'---':>12} {'---':>12} {'---':>8} "
                      f"{r['elapsed_ms']:>10,.1f} {'✗':>6}")


def print_summary(results):
    """打印汇总统计"""
    success = [r for r in results if r["success"]]
    a_star = [r for r in success if r["algo"] == "A*3D"]
    dijkstra = [r for r in success if r["algo"] == "Dijkstra3D"]

    print(f"\n{'='*100}")
    print(f"  汇总统计")
    print(f"{'='*100}")

    if a_star:
        avg_dist_a = np.mean([r["distance_m"] for r in a_star])
        avg_energy_a = np.mean([r["energy_kj"] for r in a_star])
        avg_time_a = np.mean([r["elapsed_ms"] for r in a_star])
        print(f"  A*3D:        平均距离={avg_dist_a:,.0f}m | 平均能耗={avg_energy_a:,.0f}kJ | 平均规划耗时={avg_time_a:,.1f}ms")

    if dijkstra:
        avg_dist_d = np.mean([r["distance_m"] for r in dijkstra])
        avg_energy_d = np.mean([r["energy_kj"] for r in dijkstra])
        avg_time_d = np.mean([r["elapsed_ms"] for r in dijkstra])
        print(f"  Dijkstra3D:  平均距离={avg_dist_d:,.0f}m | 平均能耗={avg_energy_d:,.0f}kJ | 平均规划耗时={avg_time_d:,.1f}ms")

    if a_star and dijkstra:
        print(f"  Dijkstra/A* 距离比: {avg_dist_d/avg_dist_a:.3f} (Dijkstra更优={avg_dist_d <= avg_dist_a})")
        print(f"  Dijkstra/A* 速度比: {avg_time_d/avg_time_a:.2f}x (A*更快)" if avg_time_a > 0 else "")

    # 策略维度对比
    print(f"\n  按策略分组:")
    for strat in ["balanced", "safe", "fast", "energy"]:
        strat_r = [r for r in success if r["strategy"] == strat]
        if strat_r:
            avg_d = np.mean([r["distance_m"] for r in strat_r])
            avg_e = np.mean([r["energy_kj"] for r in strat_r])
            strategy_cn = {"balanced": "均衡", "safe": "安全", "fast": "快速", "energy": "节能"}
            print(f"    {strategy_cn[strat]:<8} 平均距离={avg_d:,.0f}m | 平均能耗={avg_e:,.0f}kJ")

    success_rate = len(success) / len(results) * 100 if results else 0
    print(f"\n  总测试: {len(results)} | 成功: {len(success)} | 成功率: {success_rate:.1f}%")


def export_csv(results, path="benchmark_results.csv"):
    """导出为CSV"""
    import csv
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  CSV已导出: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A* vs Dijkstra 算法基准对比")
    parser.add_argument("--csv", action="store_true", help="同时导出CSV文件")
    args = parser.parse_args()

    print("=" * 100)
    print("  水域机器人路径规划 — A*3D vs Dijkstra3D 基准测试")
    print("=" * 100)
    print("  环境: 水流/水压/风浪/水温/能见度/潮汐/漩涡")
    print("  A*3D:  26方向 + 欧几里得启发式")
    print("  Dijkstra3D: 26方向 + 无启发式 (保证最优但更慢)")
    print()

    t_start = time.perf_counter()
    results = run_benchmark()
    total_elapsed = time.perf_counter() - t_start

    print_table(results)
    print_summary(results)
    print(f"\n  基准测试总耗时: {total_elapsed:.1f}s")

    if args.csv:
        export_csv(results)
