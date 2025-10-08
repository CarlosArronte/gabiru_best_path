import numpy as np
from scipy import interpolate
import pandas as pd
from qpsolvers import solve_qp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory
import os

class BestPath(Node):
    def __init__(self, csv_file_path, name):
        super().__init__('best_path_node')
        self.publisher_ = self.create_publisher(PoseArray, '/optimal_path', 10)
        self.client = self.create_client(Trigger, 'ready_to_receive_path')
        self.csv_file_path = csv_file_path
        self.name = name
        self.sended = False
        if not os.path.exists(self.csv_file_path):
            self.get_logger().error(f"CSV file not found: {self.csv_file_path}")
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        self.get_logger().info(f"CSV file found: {self.csv_file_path}")
        self.wait_for_segmentation_node()

    def wait_for_segmentation_node(self):
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for Segmentation Node to start...")
        request = Trigger.Request()
        future = self.client.call_async(request)
        future.add_done_callback(self.ready_response_callback)

    def ready_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"SegmentationNode ready: {response.message}")
                self.publish_best_path()
            else:
                self.get_logger().error(f"SegmentationNode preparation failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Error calling service: {e}")

    def publish_best_path(self):
        if self.sended:
            self.get_logger().info("Best path already sent.")
            return
        try:
            # Generate optimal trajectory
            trajMCP, _ = self.min_curvature_path_gen(self.csv_file_path, self.name)
            # Convert to PoseArray
            msg = PoseArray()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            for x, y in trajMCP:
                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = 0.0
                msg.poses.append(pose)
            # Publish
            self.publisher_.publish(msg)
            self.get_logger().info(f"Optimal path published with {len(msg.poses)} poses")
            self.sended = True
        except Exception as e:
            self.get_logger().error(f"Error publishing trajectory: {e}")

    def _validate_csv_headers(self, data):
        try:
            if data.shape[1] != 4:
                raise ValueError("CSV file must have exactly 4 columns: x_m, y_m, w_tr_right_m, w_tr_left_m")
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                raise ValueError("CSV contains NaN or infinite values")
        except Exception as e:
            raise Exception(f"Error validating CSV file: {e}")

    def _get_interpolation_step_length(self, track):
        stepLengths = np.sqrt(np.sum(np.diff(track, axis=0)**2, axis=1))
        return np.concatenate(([0], stepLengths))

    def _stack_xy(self, track):
        return np.column_stack((track["x"], track["y"]))

    def _compute_cumulative_length(self, track):
        return np.cumsum(self._get_interpolation_step_length(track))

    def _interpolate_xy(self, cumulative_len, track, step):
        return interpolate.interp1d(cumulative_len, track, axis=0)(step)

    def _interpolate_scalar(self, cumulative_len, values, steps, kind="cubic"):
        return interpolate.interp1d(cumulative_len, values, kind=kind)(steps)

    def get_centerline_data(self, centerline_csv):
        self.get_logger().info(f"Loading CSV file: {centerline_csv}")
        try:
            data = pd.read_csv(centerline_csv, comment='#', header=None)
            self._validate_csv_headers(data)
            self.get_logger().info(f"CSV loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            return data.values
        except Exception as e:
            self.get_logger().error(f"Error loading CSV: {e}")
            raise

    def close_track(self, track):
        if not np.allclose(track[0, :2], track[-1, :2]):
            self.get_logger().warning("First and last points do not match. Appending first point to close the loop.")
            return np.vstack([track, track[0, :]])
        return track

    def extract_track_features(self, track):
        return {
            "x": track[:, 0],
            "y": track[:, 1],
            "twr": track[:, 2],
            "twl": track[:, 3]
        }

    def interpolate_track(self, track):
        pathXY = self._stack_xy(track)
        cumulativeLen = self._compute_cumulative_length(pathXY)
        finalStepLocs = np.linspace(0, cumulativeLen[-1], 1500)
        return {
            "xt": self._interpolate_xy(cumulativeLen, pathXY, finalStepLocs)[:, 0],
            "yt": self._interpolate_xy(cumulativeLen, pathXY, finalStepLocs)[:, 1],
            "twrt": self._interpolate_scalar(cumulativeLen, track["twr"], finalStepLocs),
            "twlt": self._interpolate_scalar(cumulativeLen, track["twl"], finalStepLocs)
        }

    def min_curvature_path_gen(self, csv_file_path, name):
        track = self.get_centerline_data(csv_file_path)
        track = self.close_track(track)
        track_data = self.extract_track_features(track)
        track_interpolated = self.interpolate_track(track_data)
        dx = np.gradient(track_interpolated["xt"])
        dy = np.gradient(track_interpolated["yt"])
        dL = np.hypot(dx, dy)

        if np.any(dL == 0):
            self.get_logger().error("Zero segment length detected in dL")
            raise ValueError("Zero segment length detected")

        def xoff(a, i): return -a * dy[i] / dL[i] + track_interpolated["xt"][i]
        def yoff(a, i): return a * dx[i] / dL[i] + track_interpolated["yt"][i]

        offset = np.column_stack((-track_interpolated["twrt"], track_interpolated["twlt"]))
        xin = np.zeros_like(track_interpolated["xt"])
        yin = np.zeros_like(track_interpolated["yt"])
        xout = np.zeros_like(track_interpolated["xt"])
        yout = np.zeros_like(track_interpolated["yt"])

        for i in range(len(track_interpolated["xt"])):
            xin[i] = xoff(offset[i, 0], i)
            yin[i] = yoff(offset[i, 0], i)
            xout[i] = xoff(offset[i, 1], i)
            yout[i] = yoff(offset[i, 1], i)

        delx = xout - xin
        dely = yout - yin
        n = len(delx)
        H = np.zeros((n, n))
        B = np.zeros(n)

        for i in range(1, n-1):
            H[i-1, i-1] += delx[i-1]**2 + dely[i-1]**2
            H[i-1, i] += -2 * delx[i-1] * delx[i] - 2 * dely[i-1] * dely[i]
            H[i-1, i+1] += delx[i-1] * delx[i+1] + dely[i-1] * dely[i+1]
            H[i, i-1] += -2 * delx[i-1] * delx[i] - 2 * dely[i-1] * dely[i]
            H[i, i] += 4 * delx[i]**2 + 4 * dely[i]**2
            H[i, i+1] += -2 * delx[i] * delx[i+1] - 2 * dely[i] * dely[i+1]
            H[i+1, i-1] += delx[i-1] * delx[i+1] + dely[i-1] * dely[i+1]
            H[i+1, i] += -2 * delx[i] * delx[i+1] - 2 * dely[i] * dely[i+1]
            H[i+1, i+1] += delx[i+1]**2 + dely[i+1]**2

        for i in range(1, n-1):
            B[i-1] += 2 * (xin[i+1] + xin[i-1] - 2*xin[i]) * delx[i-1] + 2 * (yin[i+1] + yin[i-1] - 2*yin[i]) * dely[i-1]
            B[i] += -4 * (xin[i+1] + xin[i-1] - 2*xin[i]) * delx[i] - 4 * (yin[i+1] + yin[i-1] - 2*yin[i]) * dely[i]
            B[i+1] += 2 * (xin[i+1] + xin[i-1] - 2*xin[i]) * delx[i+1] + 2 * (yin[i+1] + yin[i-1] - 2*yin[i]) * dely[i+1]

        lb = np.zeros(n)
        ub = np.ones(n)
        Aeq = np.zeros(n)
        Aeq[0] = 1
        Aeq[-1] = -1
        beq = np.array([0.0])
        P = 2 * H
        q = B
        G = np.vstack([np.eye(n), -np.eye(n)])
        h = np.hstack([ub, -lb])
        A = Aeq.reshape(1, -1)

        # Validate matrices
        if np.any(np.isnan(P)) or np.any(np.isinf(P)):
            self.get_logger().error(f"Invalid P matrix: NaN={np.any(np.isnan(P))}, Inf={np.any(np.isinf(P))}")
            raise ValueError("Invalid P matrix")
        if np.any(np.isnan(q)) or np.any(np.isinf(q)):
            self.get_logger().error(f"Invalid q vector: NaN={np.any(np.isnan(q))}, Inf={np.any(np.isinf(q))}")
            raise ValueError("Invalid q vector")

        resMCP = solve_qp(P, q, G, h, A, beq, solver="osqp")
        if resMCP is None:
            self.get_logger().error("solve_qp returned None: Optimization failed")
            raise ValueError("Quadratic programming solver failed")

        xresMCP = xin + resMCP * delx
        yresMCP = yin + resMCP * dely
        trajMCP = np.column_stack((xresMCP, yresMCP))
        return trajMCP, None

def main(args=None):
    rclpy.init(args=args)
    track_name = "SP"
    # Use package share directory to locate CSV
    csv_path = os.path.join(
        get_package_share_directory('gabiru_best_path'),
        'gabiru_best_path',
        'SaoPaulo_centerline.csv'
    )
    try:
        node = BestPath(csv_path, track_name)
        rclpy.spin(node)
        node.destroy_node()
    except Exception as e:
        print(f"Error in BestPath node: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()