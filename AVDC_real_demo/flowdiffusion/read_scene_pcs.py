import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import open3d as o3d

class RealSense():
    def __init__(self, ):
        # Setup:
        pipe = rs.pipeline()
        cfg = rs.config()
        # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.

        serial = rs.context().devices[0].get_info(rs.camera_info.serial_number)

        rs.config.enable_device(cfg, str(serial))
        # Configure the pipeline to stream the depth stream
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        self.cfg = cfg
        self.pipe = pipe
        self.serial = serial

    def get_data(self,):
        pipe = self.pipe
        cfg = self.cfg
        pipe.start(cfg)
        flag_stop = False
        while True:
            color_image, depth_image = self.capture()
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            cv2.imshow('RealSense', color_image)
            ch = cv2.waitKey(25)
            if ch == 115:
                break
            elif ch & 0xFF == ord('q') or ch == 27:
                cv2.destroyAllWindows()
                flag_stop = True
                break
        pipe.stop()
        return color_image, depth_image, flag_stop

    def capture(self,):
        # Skip 5 first frames to give the Auto-Exposure time to adjust
        pipe = self.pipe
        serial = self.serial
        cfg = self.cfg
        pipe.wait_for_frames()
        
        # Store next frameset for later processing:
        frameset = pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        # Cleanup:

        ## TODO: for debug
        # self.get_pcd(depth_frame, color_frame)
        # import ipdb;ipdb.set_trace()
        
        color = np.asanyarray(color_frame.get_data())

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        color_image = np.asanyarray(color)[:, :, ::-1]
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def get_pcd(self, depth_frame, color_frame):
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        pointcloud = pc.calculate(depth_frame)
        pointcloud.export_to_ply("1.ply", color_frame)
        pcd = o3d.io.read_point_cloud('1.ply')
        o3d.io.write_point_cloud("1.ply", pcd)
        return pcd
