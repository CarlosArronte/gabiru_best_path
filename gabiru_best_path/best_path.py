import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
from qpsolvers import solve_qp  
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header

import importlib.resources as pkg_resources
import gabiru_best_path


class BestPath(Node):
    def __init__(self, csv_file_path, name):
        super().__init__('best_path_node')
        self.publisher_ = self.create_publisher(PoseArray, '/optimal_path', 10)

        # Genera trayectoria óptima
        trajMCP, _ = self.min_curvature_path_gen(csv_file_path, name)

        # Convierte a PoseArray
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

        # Publica una sola vez
        self.publisher_.publish(msg)
        self.get_logger().info(f"Optimal path published with {len(msg.poses)} poses")

    def validateCenterlineDataHeaders(self,data):
        try:
            if data.shape[1] != 4:
                raise ValueError("CSV file must have exactly 4 columns: x_m, y_m, w_tr_right_m, w_tr_left_m")            
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")

    def get_centerline_data(self,centerline_csv):
        # Read track data from CSV, skipping comment lines starting with '#'       
        data = pd.read_csv(centerline_csv, comment='#', header=None)
        self.validateCenterlineDataHeaders(data)
        return data.values
    
    def close_track(self,track):
        # Check if the first and last points are the same (closed loop)
        if not np.allclose(track[0, :2], track[-1, :2]):
            print("Warning: First and last points do not match. Appending first point to close the loop.")
            return np.vstack([track, track[0, :]])
        return track
    
    def extract_track_features(self,track):
        features = {}
        features["x"] = track[:, 0]
        features["y"] = track[:, 1]
        features["twr"] = track[:, 2]
        features["twl"] = track[:, 3]
        return features

    def get_interpolation_step_length(self,track):
        stepLengths = np.sqrt(np.sum(np.diff(track, axis=0)**2, axis=1))
        return np.concatenate(([0], stepLengths))  # Add starting point
        
    
    def interpolate_track(self,track):
        
        nseg = 1500
        pathXY = np.column_stack((track["x"], track["y"]))        
        cumulativeLen = np.cumsum(self.get_interpolation_step_length(pathXY))
        finalStepLocs = np.linspace(0, cumulativeLen[-1], nseg)

        # Interpolate x, y, twr, twl
        finalPathXY = interpolate.interp1d(cumulativeLen, pathXY, axis=0)(finalStepLocs)
        interpolated_track = {}
        interpolated_track["xt"] = finalPathXY[:, 0]
        interpolated_track["yt"] = finalPathXY[:, 1]
        interpolated_track["twrt"] = interpolate.interp1d(cumulativeLen, track["twr"], kind='cubic')(finalStepLocs)
        interpolated_track["twlt"] = interpolate.interp1d(cumulativeLen,  track["twl"], kind='cubic')(finalStepLocs)

        return interpolated_track
    

    def min_curvature_path_gen(self,csv_file_path, name):
      
        track = self.get_centerline_data(csv_file_path)
        
        track = self.close_track(track)

        track_data = self.extract_track_features(track) 

        track_interpolated = self.interpolate_track(track_data)       

        # Interpolate data to get finer curve with equal distances between segments
        nseg = 1500
        pathXY = np.column_stack((track_data["x"], track_data["y"]))
        stepLengths = np.sqrt(np.sum(np.diff(pathXY, axis=0)**2, axis=1))
        stepLengths = np.concatenate(([0], stepLengths))  # Add starting point
        cumulativeLen = np.cumsum(stepLengths)
        finalStepLocs = np.linspace(0, cumulativeLen[-1], nseg)

        # Interpolate x, y, twr, twl
        finalPathXY = interpolate.interp1d(cumulativeLen, pathXY, axis=0)(finalStepLocs)
        xt = finalPathXY[:, 0]
        yt = finalPathXY[:, 1]
        twrt = interpolate.interp1d(cumulativeLen, track_data["twr"], kind='cubic')(finalStepLocs)
        twlt = interpolate.interp1d(cumulativeLen,  track_data["twl"], kind='cubic')(finalStepLocs)

        # Normal direction for each vertex
        dx = np.gradient(xt)
        dy = np.gradient(yt)
        dL = np.hypot(dx, dy)

        # Offset curve functions for a specific index
        def xoff(a, i): return -a * dy[i] / dL[i] + xt[i]
        def yoff(a, i): return a * dx[i] / dL[i] + yt[i]

        # Plot reference line
        plt.figure()
        plt.plot(xt, yt, 'g', label='Center Line')
        # Offset data
        offset = np.column_stack((-twrt, twlt))
        xin = np.zeros_like(xt)
        yin = np.zeros_like(yt)
        xout = np.zeros_like(xt)
        yout = np.zeros_like(yt)

        for i in range(len(xt)):
            xin[i] = xoff(offset[i, 0], i)  # Inner offset curve
            yin[i] = yoff(offset[i, 0], i)
            xout[i] = xoff(offset[i, 1], i)  # Outer offset curve
            yout[i] = yoff(offset[i, 1], i)

        # Plot inner and outer tracks
        plt.plot(xin, yin, color='b', linewidth=2, label='Inner Border')
        plt.plot(xout, yout, color='r', linewidth=2, label='Outer Border')
        plt.legend()
        plt.xlabel('x (m)', fontweight='bold', fontsize=14)
        plt.ylabel('y (m)', fontweight='bold', fontsize=14)
        plt.title(name, fontsize=16)
        plt.axis('equal')
        plt.show()

        # Form delta matrices
        delx = xout - xin
        dely = yout - yin
        trackData = np.column_stack((xt, yt, xin, yin, xout, yout))

        # Matrix Definition
        n = len(delx)
        H = np.zeros((n, n))
        B = np.zeros(n)

        # Formation of H matrix (nxn)
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

        # Formation of B matrix (1xn)
        for i in range(1, n-1):
            B[i-1] += 2 * (xin[i+1] + xin[i-1] - 2*xin[i]) * delx[i-1] + 2 * (yin[i+1] + yin[i-1] - 2*yin[i]) * dely[i-1]
            B[i] += -4 * (xin[i+1] + xin[i-1] - 2*xin[i]) * delx[i] - 4 * (yin[i+1] + yin[i-1] - 2*yin[i]) * dely[i]
            B[i+1] += 2 * (xin[i+1] + xin[i-1] - 2*xin[i]) * delx[i+1] + 2 * (yin[i+1] + yin[i-1] - 2*yin[i]) * dely[i+1]

        # Define constraints
        lb = np.zeros(n)
        ub = np.ones(n)
        Aeq = np.zeros(n)
        Aeq[0] = 1
        Aeq[-1] = -1
        beq = np.array([0.0])

        # ---- Quadratic Programming Solver (qpsolvers) ----
        P = 2 * H   # Hessian
        q = B       # Linear term
        G = np.vstack([np.eye(n), -np.eye(n)])   # Inequality constraints
        h = np.hstack([ub, -lb])
        A = Aeq.reshape(1, -1)

        resMCP = solve_qp(P, q, G, h, A, beq, solver="osqp")


        # Plotting results (original plot)
        xresMCP = xin + resMCP * delx
        yresMCP = yin + resMCP * dely

        plt.figure()
        plt.plot(xresMCP, yresMCP, color='r', linewidth=2, label='Optimal Trajectory')
        plt.plot([xin[0], xout[0]], [yin[0], yout[0]], color='b', linewidth=2, label='Starting Line')
        plt.plot(xt, yt, '--', color='g', label='Center Line')
        plt.plot(xin, yin, color='k', label='Inner Border')
        plt.plot(xout, yout, color='k', label='Outer Border')
        plt.legend()
        plt.xlabel('x (m)', fontweight='bold', fontsize=14)
        plt.ylabel('y (m)', fontweight='bold', fontsize=14)
        plt.title(f"{name} - Minimum Curvature Trajectory", fontsize=16)
        plt.axis('equal')

        # Additional plot for centerline, left border, right border
        plt.figure()
        plt.plot(trackData[:, 0], trackData[:, 1], color='g', linewidth=1.5, label='Centerline')
        plt.plot(trackData[:, 2], trackData[:, 3], color='b', linewidth=1.5, label='Inner Border')
        plt.plot(trackData[:, 4], trackData[:, 5], color='r', linewidth=1.5, label='Outer Border')
        plt.legend()
        plt.xlabel('x (m)', fontweight='bold', fontsize=14)
        plt.ylabel('y (m)', fontweight='bold', fontsize=14)
        plt.title(f"{name} - Track Borders and Centerline", fontsize=16)
        plt.axis('equal')
        plt.show()

        trajMCP = np.column_stack((xresMCP, yresMCP))
        return trajMCP, trackData
    

def main(args=None):
    rclpy.init(args=args)

    # Ajusta según tu pista
    track_name = "SP"

    with pkg_resources.path(gabiru_best_path, "SaoPaulo_centerline.csv") as csv_path:
        node = BestPath(str(csv_path), track_name)   # <-- pasamos ruta, no DataFrame

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()