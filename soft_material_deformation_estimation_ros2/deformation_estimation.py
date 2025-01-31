import numpy as np
import torch
from geometry_msgs.msg import WrenchStamped
import rclpy
from rclpy.node import Node
import pykinect_azure as pykinect
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension
import siamese_models

input_shape = (1, 1, 224, 224)
model_folder = ''


def make_mask(crop_size) -> np.ndarray:
    left_diagonal = lambda x: np.tan(np.deg2rad(90 - 30)) * x + (512 - np.tan(np.deg2rad(90 - 30)) * 150)

    left_diagonal_mask = np.ones((512, 512))
    for x in range(150):
        for y in range(512):
            if y > left_diagonal(x):
                left_diagonal_mask[y, x] = 0

    right_diagonal = lambda x: -np.tan(np.deg2rad(60)) * x + (512 + np.tan(np.deg2rad(60)) * (512 - 150))

    right_diagonal_mask = np.ones((512, 512))
    for x in range(512 - 150, 512):
        for y in range(512):
            if y > right_diagonal(x):
                right_diagonal_mask[y, x] = 0

    top_side_mask = np.ones((512, 512))
    top_side_mask[:150, :] = 0

    bottom_side_mask = np.ones((512, 512))
    bottom_side_mask[450:, :] = 0

    mask = left_diagonal_mask * right_diagonal_mask * top_side_mask * bottom_side_mask

    return mask[crop_size[0][0]: crop_size[0][1], crop_size[1][0]: crop_size[1][1]]


class DeformationNode(Node):

    def __init__(self):

        super().__init__('deformation_estimation')

        self.segmentation_threshold = 0.5
        self.publish_classes = True
        self.publish_probabilities = True
        self.publish_raw_depth = True
        self.publish_rest_ply_shape = True
        self.publish_depth = True

        self.logger = self.get_logger()

        segmentation_model_path = '/home/kildall/ros2_projects/deformable_ws/src/soft_material_deformation_estimation_ros2/models/segmentation/best_model.pth'
        deformation_model_path = '/home/kildall/ros2_projects/deformable_ws/src/soft_material_deformation_estimation_ros2/models/deformation_estimation/epoch52.pth'

        self.logger.info(f'{segmentation_model_path}')
        self.logger.info(f'{deformation_model_path}')

        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        device_config.synchronized_images_only = False
        pykinect.initialize_libraries(track_body=False)

        self.azure = pykinect.start_device(config=device_config)

        self.nn_segmentor = torch.load(segmentation_model_path)
        self.nn_segmentor.eval()
        self.device = 'cuda'

        # Start device
        # Start body tracker

        self.depth_threshold = 1300
        self.crop_size: tuple = ((90, 512), (50, 472))
        self.static_mask = make_mask(self.crop_size)
        a = (900 - 1600) / (450 - 200)
        b = 1600 - 200 * a
        depth_line = lambda y: a * y + b
        depth_matrix = np.ones((512, 512))
        for col in range(512):
            depth_matrix[:, col] *= depth_line(np.array([v for v in range(512)]))
        self.depth_matrix = depth_matrix[self.crop_size[0][0]: self.crop_size[0][1],
                            self.crop_size[1][0]: self.crop_size[1][1]]

        # self.calibration = self.azure.get_calibration(device_config.depth_mode, device_config.color_resolution)
        if self.publish_depth:
            self.br = CvBridge()
            self.segmented_depth_pub = self.create_publisher(topic='/segmented_depth',
                                                             msg_type=Image,
                                                             qos_profile=10)
            self.raw_depth_pub = self.create_publisher(topic='/raw_depth',
                                                       msg_type=Image,
                                                       qos_profile=10)

        if self.publish_probabilities:
            self.prob_def_pub = self.create_publisher(topic='/deformation_probabilities',
                                                     msg_type=Float32MultiArray,
                                                     qos_profile=10)

        if self.publish_classes:
            self.classes_def_pub = self.create_publisher(topic='/deformation_classes',
                                                      msg_type=Int32MultiArray,
                                                      qos_profile=10)


        self.deformation_estimator: siamese_models.SiameseMultiHeadNetwork = torch.load(deformation_model_path)

        self.deformation_publisher = self.create_publisher(topic='/deformation_estimation',
                                                           msg_type=WrenchStamped,
                                                           qos_profile=10)
        self.twist_publisher = self.create_publisher(topic='/imm/commands',
                                                     msg_type=Twist,
                                                     qos_profile=10)

        self.ply_rest_shape_client = self.create_service(srv_type=Trigger, srv_name='get_rest_ply_shape', callback=self.get_rest_ply_shape)

        self.rest_ply_shape = torch.zeros(input_shape, dtype=torch.float, device=self.device)
        self.rest_ply_shape_features = None


        self.filter_length = 1
        self.resolution = (224, 224)
        self.depth_resolution = (512, 512)


    @torch.no_grad()
    def get_segmented_depth(self):
        capture = self.azure.update()
        _, depth = capture.get_depth_image()
        depth = depth[self.crop_size[0][0]: self.crop_size[0][1], self.crop_size[1][0]: self.crop_size[1][1]]

        if self.publish_raw_depth:
            img_msg: Image = self.br.cv2_to_imgmsg((depth).astype(np.uint16))
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.raw_depth_pub.publish(img_msg)

        depth *= depth < self.depth_threshold
        depth = depth * self.static_mask
        depth *= depth < self.depth_matrix
        depth = cv2.resize(depth, dsize=self.resolution)

        tensor_depth = torch.from_numpy(depth.reshape(1, 1, self.resolution[0], self.resolution[1])).float().to(
            self.device) / 1000

        mask = self.nn_segmentor(tensor_depth).squeeze().detach().to('cpu').numpy() > self.segmentation_threshold
        depth *= mask

        if self.publish_depth:
            img_msg: Image = self.br.cv2_to_imgmsg((depth * 1000).astype(np.uint16))
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.segmented_depth_pub.publish(img_msg)

        return depth

    @torch.no_grad()
    def estimate_deformation(self):

        if self.rest_ply_shape_features is not None:
            segmented_depth = self.get_segmented_depth()

            if np.sum(segmented_depth) > 0:
                depth = torch.from_numpy(segmented_depth.reshape(input_shape)).float().to(self.device) / 1000
                prob_deformation: np.array = self.deformation_estimator.fast_predict_classification(self.rest_ply_shape_features,
                                                                                                    depth).squeeze().cpu().detach().numpy()
                if self.publish_probabilities:
                    dim1 = MultiArrayDimension()
                    dim1.label = "dofs"
                    dim1.size = 4  # Number of rows
                    dim1.stride = 5  # Total number of elements in each row (this will be the number of columns)

                    dim2 = MultiArrayDimension()
                    dim2.label = "classes"
                    dim2.size = 5  # Number of columns
                    dim2.stride = 1  # There are 5 columns in total

                    # Set the layout of the message
                    msg = Float32MultiArray()
                    msg.layout.dim = [dim1, dim2]
                    msg.data = prob_deformation.ravel().tolist()
                    self.prob_def_pub.publish(msg)

                if self.publish_classes:
                    deformation_classes = np.argmax(prob_deformation, axis=-1)
                    msg = Int32MultiArray()
                    msg.data = deformation_classes.tolist()
                    self.classes_def_pub.publish(msg)

    def get_rest_ply_shape(self, request, response: Trigger.Response):
        self.rest_ply_shape[0,0] = torch.from_numpy(self.get_segmented_depth()).float().to(self.device)
        self.rest_ply_shape_features = self.deformation_estimator.extract_features(self.rest_ply_shape)
        response.success = True
        return response


def main():
    rclpy.init()
    node = DeformationNode()
    node.create_timer(timer_period_sec=1 / 30, callback=node.estimate_deformation)

    rclpy.spin(node)
    # executor = rclpy.executors.MultiThreadedExecutor()
    # executor.add_node(node)
    # executor.spin()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
