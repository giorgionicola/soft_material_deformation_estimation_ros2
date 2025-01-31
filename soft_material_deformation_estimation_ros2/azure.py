import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import pykinect_azure as pykinect
from sensor_msgs.msg import Image, CameraInfo


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

        info = CameraInfo()

        self.br = CvBridge()
        self.img_publisher = self.create_publisher(topic='/rgb_img', msg_type=Image, qos_profile=10)

    def pub_image(self):
        capture = self.azure.update()
        _, rgb = capture.get_color_image()
        if rgb is not None:
            img_msg: Image = self.br.cv2_to_imgmsg(rgb)
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.img_publisher.publish(img_msg)


def main():
    rclpy.init()
    node = Azure()
    # node.create_timer(timer_period_sec=1 / 30, callback=node.estimate_deformation)
    node.create_timer(timer_period_sec=1 / 30, callback=node.pub_image)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
