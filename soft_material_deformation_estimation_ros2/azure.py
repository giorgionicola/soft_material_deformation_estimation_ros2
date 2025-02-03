import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import pykinect_azure as pykinect
from sensor_msgs.msg import Image, CameraInfo
from sentry_sdk.profiler import frame_id


class Azure(Node):
    def __init__(self):
        super().__init__('azure')
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        device_config.synchronized_images_only = False
        pykinect.initialize_libraries(track_body=False)

        self.azure = pykinect.start_device(config=device_config)
        self.calibration = self.azure.get_calibration(device_config.depth_mode, device_config.color_resolution)
        self.color_calib = self.calibration.color_camera

        self.br = CvBridge()
        self.img_publisher = self.create_publisher(topic='rgb_img', msg_type=Image, qos_profile=10)
        self.info_publisher = self.create_publisher(topic='camera_info', msg_type=CameraInfo, qos_profile=10)

        self.camera_info_msg = CameraInfo(frame_id='camera_color_optical_frame',
                                          width=self.color_calib.resolution_width,
                                          height=self.color_calib.resolution_height,
                                          K=[self.color_calib.fx, 0.0, self.color_calib.cx,
                                             0.0, self.color_calib.fy, self.color_calib.cy,
                                             0.0, 0.0, 1.0],
                                          D=list(self.color_calib.distortion_coefficients),
                                          distortion_model='plumb_bob',
                                          R=[1.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0,
                                             0.0, 0.0, 1.0],
                                          P=[self.color_calib.fx, 0.0, self.color_calib.cx, 0.0,
                                             0.0, self.color_calib.fy, self.color_calib.cy, 0.0,
                                             0.0, 0.0, 1.0, 0.0]
                                          )

        self.create_timer(timer_period_sec=1 / 30, callback=self.pub_rgb_image)
        self.create_timer(timer_period_sec=1, callback=self.publish_camera_info)

    def pub_rgb_image(self):
        capture = self.azure.update()
        _, rgb = capture.get_color_image()

        if rgb is not None:
            img_msg: Image = self.br.cv2_to_imgmsg(rgb)
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.img_publisher.publish(img_msg)

    def publish_camera_info(self):
        self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        self.info_publisher.publish(self.camera_info_msg)

    def pub_depth_image(self):
        capture = self.azure.update()
        _, depth = capture.get_depth_image()
        if depth is not None:
            img_msg: Image = self.br.cv2_to_imgmsg(depth)
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.img_publisher.publish(img_msg)


def main():
    rclpy.init()
    node = Azure()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
