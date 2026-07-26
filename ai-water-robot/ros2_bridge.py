"""
ROS2 桥接节点 — 水域机器人与ROS2生态互联

发布Topic:
  /water_robot/pose           geometry_msgs/PoseStamped    机器人位置和朝向
  /water_robot/odom           nav_msgs/Odometry            航速/航向
  /water_robot/battery        sensor_msgs/BatteryState     电量百分比
  /water_robot/sensor         sensor_msgs/FluidPressure    深度/水压/水温
  /water_robot/path           nav_msgs/Path                规划路径
  /water_robot/status         std_msgs/String              状态文字

订阅Topic:
  /water_robot/cmd_goal       geometry_msgs/PoseStamped    设置终点
  /water_robot/cmd_start      geometry_msgs/PoseStamped    设置起点
  /water_robot/cmd_waypoint   geometry_msgs/PoseStamped    添加途经点
  /water_robot/cmd_strategy   std_msgs/String              切换策略

Service:
  /water_robot/start_mission  std_srvs/SetBool             启动/停止动画
  /water_robot/clear_all      std_srvs/Trigger             重置场景

TF:
  map -> water_robot/base_link
"""

import threading
import math
from typing import Optional, Dict, List, Tuple, Any

# ROS2 可选依赖
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
    from nav_msgs.msg import Odometry, Path as RosPath
    from sensor_msgs.msg import BatteryState, FluidPressure
    from std_msgs.msg import String, Header
    from std_srvs.srv import SetBool, Trigger
    from tf2_ros import TransformBroadcaster
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class WaterRobotBridge:
    """水域机器人 ROS2 桥接器 — 线程安全

    当 rclpy 未安装时，创建实例会抛出 RuntimeError。
    使用前检查 ROS2_AVAILABLE 或捕获异常。
    """

    def __init__(self, node_name="water_robot_bridge"):
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "rclpy 未安装。请运行: pip install rclpy && pip install geometry_msgs nav_msgs sensor_msgs std_msgs std_srvs tf2_ros"
            )

        self._lock = threading.Lock()
        self._running = False
        self._node = None
        self._spin_thread = None
        self._node_name = node_name

        # ── GUI → ROS2 数据（GUI线程写入，ROS2线程读取） ──
        self._robot_pose = (0.0, 0.0, 0.0)
        self._robot_heading = 0.0
        self._robot_speed_kn = 0.0
        self._battery_pct = 100.0
        self._depth_m = 0.0
        self._temperature_c = 15.0
        self._visibility_m = 10.0
        self._water_pressure_kpa = 101.3
        self._status_text = "就绪"
        self._path_points = []
        self._grid_resolution = 50.0
        self._frame_id = "map"

        # ── ROS2 → GUI 数据（ROS2线程写入，GUI线程轮询） ──
        self._pending_goal = None
        self._pending_start = None
        self._pending_waypoints = []
        self._pending_strategy = None
        self._cmd_start_mission = None
        self._cmd_clear_all = False

    # ═══════════════════ 生命周期 ═══════════════════

    def start(self):
        """启动ROS2节点（独立线程）"""
        if not ROS2_AVAILABLE:
            return False
        if self._running:
            return True
        self._running = True
        self._spin_thread = threading.Thread(target=self._ros2_spin, daemon=True)
        self._spin_thread.start()
        return True

    def stop(self):
        """停止ROS2节点"""
        self._running = False
        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)

    @property
    def is_running(self):
        return self._running

    # ═══════════════════ GUI → ROS2 (数据写入) ═══════════════════

    def update_robot_state(self, x, y, z, heading_rad, speed_kn):
        with self._lock:
            self._robot_pose = (x, y, z)
            self._robot_heading = heading_rad
            self._robot_speed_kn = speed_kn

    def update_battery(self, pct):
        with self._lock:
            self._battery_pct = pct

    def update_water_sensor(self, depth_m, temp_c, vis_m, pressure_kpa):
        with self._lock:
            self._depth_m = depth_m
            self._temperature_c = temp_c
            self._visibility_m = vis_m
            self._water_pressure_kpa = pressure_kpa

    def update_status(self, text):
        with self._lock:
            self._status_text = text

    def update_path(self, path, resolution):
        with self._lock:
            self._path_points = list(path)
            self._grid_resolution = resolution

    # ═══════════════════ ROS2 → GUI (数据读取) ═══════════════════

    def poll_commands(self):
        """GUI线程轮询ROS2指令（非阻塞）"""
        with self._lock:
            cmds = {
                "goal": self._pending_goal,
                "start": self._pending_start,
                "waypoints": list(self._pending_waypoints),
                "strategy": self._pending_strategy,
                "start_mission": self._cmd_start_mission,
                "clear_all": self._cmd_clear_all,
            }
            # 消费后清零
            self._pending_goal = None
            self._pending_start = None
            self._pending_waypoints = []
            self._pending_strategy = None
            self._cmd_start_mission = None
            self._cmd_clear_all = False
        return cmds

    # ═══════════════════ ROS2 内部 ═══════════════════

    def _ros2_spin(self):
        """ROS2事件循环（独立线程）"""
        rclpy.init(args=None)
        self._node = Node(self._node_name)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Publishers
        self._pub_pose = self._node.create_publisher(PoseStamped, "/water_robot/pose", qos)
        self._pub_odom = self._node.create_publisher(Odometry, "/water_robot/odom", qos)
        self._pub_battery = self._node.create_publisher(BatteryState, "/water_robot/battery", qos)
        self._pub_sensor = self._node.create_publisher(FluidPressure, "/water_robot/sensor", qos)
        self._pub_path = self._node.create_publisher(RosPath, "/water_robot/path", qos)
        self._pub_status = self._node.create_publisher(String, "/water_robot/status", qos)

        # Subscribers
        self._node.create_subscription(PoseStamped, "/water_robot/cmd_goal", self._on_goal, qos)
        self._node.create_subscription(PoseStamped, "/water_robot/cmd_start", self._on_start, qos)
        self._node.create_subscription(PoseStamped, "/water_robot/cmd_waypoint", self._on_waypoint, qos)
        self._node.create_subscription(String, "/water_robot/cmd_strategy", self._on_strategy, qos)

        # Services
        self._node.create_service(SetBool, "/water_robot/start_mission", self._on_start_mission_srv)
        self._node.create_service(Trigger, "/water_robot/clear_all", self._on_clear_all_srv)

        # TF broadcaster
        self._tf_broadcaster = TransformBroadcaster(self._node)

        # Publish timer (10Hz)
        self._node.create_timer(0.1, self._publish_all)

        self._node.get_logger().info("水域机器人 ROS2 桥接已启动")
        while self._running and rclpy.ok():
            rclpy.spin_once(self._node, timeout_sec=0.05)

        self._node.destroy_node()
        rclpy.shutdown()

    # ═══════════════════ 定时发布 (10Hz) ═══════════════════

    def _publish_all(self):
        now = self._node.get_clock().now().to_msg()

        with self._lock:
            x, y, z = self._robot_pose
            heading = self._robot_heading
            speed_kn = self._robot_speed_kn
            battery = self._battery_pct
            pressure = self._water_pressure_kpa
            temp = self._temperature_c
            status = self._status_text
            path_pts = list(self._path_points)
            resolution = self._grid_resolution

        # Pose
        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=now, frame_id=self._frame_id)
        pose_msg.pose.position = Point(x=float(x), y=float(y), z=float(z))
        cy = math.cos(heading * 0.5)
        sy = math.sin(heading * 0.5)
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=float(sy), w=float(cy))
        self._pub_pose.publish(pose_msg)

        # Odometry (speed in m/s from knots)
        odom_msg = Odometry()
        odom_msg.header = Header(stamp=now, frame_id=self._frame_id)
        odom_msg.child_frame_id = "water_robot/base_link"
        odom_msg.pose.pose = pose_msg.pose
        speed_ms = speed_kn * 0.514
        odom_msg.twist.twist.linear.x = speed_ms * math.cos(heading)
        odom_msg.twist.twist.linear.y = speed_ms * math.sin(heading)
        self._pub_odom.publish(odom_msg)

        # Battery
        batt_msg = BatteryState()
        batt_msg.header = Header(stamp=now, frame_id=self._frame_id)
        batt_msg.percentage = battery / 100.0
        batt_msg.present = True
        self._pub_battery.publish(batt_msg)

        # Sensor (FluidPressure carries pressure, variance carries temperature)
        sensor_msg = FluidPressure()
        sensor_msg.header = Header(stamp=now, frame_id="water_robot/sensor")
        sensor_msg.fluid_pressure = float(pressure)
        sensor_msg.variance = float(temp)
        self._pub_sensor.publish(sensor_msg)

        # Path
        if path_pts:
            path_msg = RosPath()
            path_msg.header = Header(stamp=now, frame_id=self._frame_id)
            for pt in path_pts:
                pose_st = PoseStamped()
                pose_st.header = Header(stamp=now, frame_id=self._frame_id)
                pose_st.pose.position = Point(
                    x=float(pt[0]) * resolution,
                    y=float(pt[1]) * resolution,
                    z=-float(pt[2]) * resolution,
                )
                path_msg.poses.append(pose_st)
            self._pub_path.publish(path_msg)

        # Status
        self._pub_status.publish(String(data=status))

        # TF
        tf_msg = TransformStamped()
        tf_msg.header = Header(stamp=now, frame_id=self._frame_id)
        tf_msg.child_frame_id = "water_robot/base_link"
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.translation.z = float(z)
        tf_msg.transform.rotation = pose_msg.pose.orientation
        self._tf_broadcaster.sendTransform(tf_msg)

    # ═══════════════════ 订阅回调 ═══════════════════

    def _on_goal(self, msg):
        with self._lock:
            self._pending_goal = (
                int(msg.pose.position.x),
                int(msg.pose.position.y),
                int(msg.pose.position.z),
            )

    def _on_start(self, msg):
        with self._lock:
            self._pending_start = (
                int(msg.pose.position.x),
                int(msg.pose.position.y),
                int(msg.pose.position.z),
            )

    def _on_waypoint(self, msg):
        with self._lock:
            self._pending_waypoints.append((
                int(msg.pose.position.x),
                int(msg.pose.position.y),
                int(msg.pose.position.z),
            ))

    def _on_strategy(self, msg):
        with self._lock:
            self._pending_strategy = msg.data

    def _on_start_mission_srv(self, request, response):
        with self._lock:
            self._cmd_start_mission = request.data
        response.success = True
        response.message = "指令已接收"
        return response

    def _on_clear_all_srv(self, request, response):
        with self._lock:
            self._cmd_clear_all = True
        response.success = True
        response.message = "重置指令已接收"
        return response


# ═══════════════════ 无ROS2时的Mock ═══════════════════

class MockBridge:
    """ROS2不可用时的空桥接（兼容接口）"""
    def __init__(self):
        self.is_running = False

    def start(self):
        return False

    def stop(self):
        pass

    def update_robot_state(self, *args):
        pass

    def update_battery(self, pct):
        pass

    def update_water_sensor(self, *args):
        pass

    def update_status(self, text):
        pass

    def update_path(self, path, res):
        pass

    def poll_commands(self):
        return {}
