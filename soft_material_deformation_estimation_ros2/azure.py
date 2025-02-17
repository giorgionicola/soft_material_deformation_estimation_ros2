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
        self.color_calib = self.calibration.color_params

        self.br = CvBridge()
        self.img_publisher = self.create_publisher(topic='rgb_img', msg_type=Image, qos_profile=10)
        self.info_publisher = self.create_publisher(topic='camera_info', msg_type=CameraInfo, qos_profile=10)

        self.camera_info_msg = CameraInfo(frame_id='camera_color_optical_frame',
                                          width=1920,
                                          height=1080,
                                          K=[919.24267578125, 0.0, 957.8638305664062,
                                             0.0, 919.4411010742188, 552.330810546875,
                                             0.0, 0.0, 1.0],
                                          D=[0.509122371673584, -2.7297146320343018, 0.0004077378544025123,
                                             -0.0001945431431522593, 1.5738297700881958, 0.38606059551239014,
                                             -2.545588731765747, 1.4968762397766113],
                                          distortion_model='rational_polynomial',
                                          R=[1.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0,
                                             0.0, 0.0, 1.0],
                                          P=[919.24267578125, 0.0, 957.8638305664062, 0.0,
                                             0.0, 919.4411010742188, 552.330810546875, 0.0,
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
