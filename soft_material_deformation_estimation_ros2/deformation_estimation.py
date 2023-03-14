import numpy as np
import torch
from geometry_msgs.msg import WrenchStamped
import rclpy
from rclpy.node import Node

import pyKinectAzure.pykinect_azure as pykinect
from pyKinectAzure.pykinect_azure.k4abt.body import Body
from pyKinectAzure.pykinect_azure.k4abt._k4abtTypes import k4abt_body_t
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
import cv2
import tf2_ros
from geometry_msgs.msg import TransformStamped, Twist

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
    segmentation_methods = ['NN', 'skeletal tracker']

    def __init__(self):

        super().__init__('deformation_estimation')

        self.logger = self.get_logger()

        self.declare_parameter('dofs')
        self.declare_parameter('half_range_def')
        self.declare_parameter('mid_def')
        self.declare_parameter('segmentation_method')
        self.declare_parameter('segmentation_model_path')
        self.declare_parameter('deformation_model_path')

        self.dofs = self.get_parameter('dofs').value
        self.half_range_def = self.get_parameter('half_range_def').value
        self.mid_def = self.get_parameter('mid_def').value

        self.segmentation_method = self.get_parameter('segmentation_method').value
        if self.segmentation_method == 'NN':
            segmentation_model_path = self.get_parameter('segmentation_model_path').value
        else:
            segmentation_model_path = None

        deformation_model_path = self.get_parameter('deformation_model_path').value

        self.logger.info(f'{self.dofs}')
        self.logger.info(f'{self.half_range_def}')
        self.logger.info(f'{self.mid_def}')
        self.logger.info(f'{self.segmentation_method}')
        self.logger.info(f'{segmentation_model_path}')
        self.logger.info(f'{deformation_model_path}')

        # deformation_model_path = "/home/mullis/drapebot_dataset/m28_demonstration/F20/models/regression/Densenet121_0.0001_1/epoch39.pth"
        # self.dofs = 4
        # self.half_range_def = [0.12, 0.365, 0.12, 0.4188790204786391]
        # self.mid_def = [0.0, 0.905, 0.0, 0.0]
        # self.segmentation_method = "NN"
        # segmentation_model_path = "/home/mullis/ros2_projects/deformable_ws/src/soft_material_deformation_estimation_ros2/models/segmentation/UNet_resnet18_1e-05_0.pth"

        self.gains = np.array([1, 1, 1, 1])
        self.dead_band = np.array([0.03, 0.03, 0.03, np.deg2rad(3)])
        self.deformation_setpoint = np.array([0, 1.1, 0, 0])

        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        device_config.synchronized_images_only = True
        pykinect.initialize_libraries(track_body=self.segmentation_method == 'skeletal tracker')

        self.publish_rviz = False
        self.publish_hands_pos = True

        if self.segmentation_method == 'skeletal tracker':
            self.apriltag_segmentor = None
            self.bodyTracker = pykinect.start_body_tracker()
            self.nn_segmentor = None
        elif self.segmentation_method == 'NN':
            self.apriltag_segmentor = None
            self.bodyTracker = None
            self.nn_segmentor = torch.load(segmentation_model_path)
            self.nn_segmentor.eval()
            self.device = 'cuda'
        else:
            print(f'Unknown segmentation method: {self.segmentation_method}')
            print(f'Avalaible segmentation methods: {self.segmentation_methods}')

        # Start device
        self.azure = pykinect.start_device(config=device_config)
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
        if self.publish_rviz:
            self.br = CvBridge()
            self.img_pub_rviz = self.create_publisher(topic='/preprocessed_image_rviz', msg_type=Image, qos_profile=10)

        if self.publish_hands_pos and self.segmentation_method == 'skeletal tracker':
            self.tf_pub = tf2_ros.TransformBroadcaster()
            self.tf_buffer = tf2_ros.Buffer()
            self.listener = tf2_ros.TransformListener(buffer=self.tf_buffer, node=self)

            self.tf_buffer.wait_for_transform_async('reference_base', 'end_effector', self.get_clock().now())
            _, rot_ee = self.tf_buffer.lookup_transform_async('reference_base', 'end_effector',
                                                              self.get_clock().now())

            self.Rot_EndEffector_start = R.from_quat(rot_ee).as_matrix()

        self.deformation_estimator = torch.load(deformation_model_path)

        self.deformation_publisher = self.create_publisher(topic='/deformation_estimation',
                                                           msg_type=WrenchStamped,
                                                           qos_profile=10)
        self.twist_publisher = self.create_publisher(topic='/twist_cmd',
                                                           msg_type=Twist,
                                                           qos_profile=10)

        self.deformation_buffer = []
        self.filter_length = 1
        self.resolution = (224, 224)
        self.depth_resolution = (512, 512)

        self.deformation_msg = WrenchStamped()
        self.twist_msg = Twist()

    def get_segmented_depth(self):
        capture = self.azure.update()

        _, depth = capture.get_depth_image()
        if self.segmentation_method == 'apriltags':
            _, color_image = capture.get_transformed_color_image()
        else:
            color_image = None

        return self.segment_depth(depth, color_image)

    def segment_depth(self, depth, color_img=None):

        depth = depth[self.crop_size[0][0]: self.crop_size[0][1], self.crop_size[1][0]: self.crop_size[1][1]]
        depth *= depth < self.depth_threshold
        depth = depth * self.static_mask
        depth *= depth < self.depth_matrix
        depth = cv2.resize(depth, dsize=self.resolution)

        if self.segmentation_method == 'apriltags':
            gray_img = self.apriltag_segmentor.GrayConversion(color_img)
            apriltag_mask, status = self.apriltag_segmentor.segment(gray_image=gray_img, pixel_offset=17)
            if status:
                depth *= apriltag_mask
        elif self.segmentation_method == 'skeletal tracker':
            depth *= self.segment_with_skeleton_tracker()
        elif self.segmentation_method == 'NN':
            tensor_depth = torch.from_numpy(depth.reshape(1, 1, self.resolution[0], self.resolution[1])).float().to(
                self.device) / 1000
            with torch.no_grad():
                mask_0 = self.nn_segmentor(tensor_depth).squeeze().detach().to('cpu').numpy()
                mask = torch.round(self.nn_segmentor(tensor_depth).squeeze()).detach().to('cpu').numpy()
                depth *= mask

        if self.publish_rviz:
            img_msg: Image = self.br.cv2_to_imgmsg((mask_0 * 100).astype(np.uint16))
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.img_pub_rviz.publish(img_msg)

        return depth

    def segment_with_skeleton_tracker(self):
        depth_mask = np.ones(self.depth_resolution)

        body_frame = self.bodyTracker.update()
        _, human_depth_image = body_frame.get_body_index_map_image()

        if body_frame.get_num_bodies() == 0:
            self.logger.warning('No body found')
        elif body_frame.get_num_bodies() > 1:
            self.logger.warning('More than 1 body found')
        else:

            body2d = body_frame.get_body2d()
            left_hand2d = [body2d.joints[8].position.x, body2d.joints[8].position.y]
            right_hand2d = [body2d.joints[15].position.x, body2d.joints[15].position.y]

            a = (right_hand2d[1] - left_hand2d[1]) / (right_hand2d[0] - left_hand2d[0])
            b = right_hand2d[1] - a * right_hand2d[0]
            line = lambda x: a * x + b

            for i in range(depth_mask.shape[1]):
                depth_mask[:int(line(i)), i] = 0

            if self.publish_hands_pos:
                self.compute_and_publish_hand_pos(body_frame)

        return depth_mask

    def compute_and_publish_hand_pos(self, body_frame):

        if body_frame.get_num_bodies() == 0:
            self.logger.warning('No body found')
        elif body_frame.get_num_bodies() > 1:
            self.logger.warning('More than 1 body found')
        else:
            body_handle = k4abt_body_t()
            body_handle.skeleton = body_frame.get_body_skeleton(0)
            body = Body(body_handle)

            left_hand_v = np.array([body.joints[8].position.x / 1000,
                                    body.joints[8].position.y / 1000,
                                    body.joints[8].position.z / 1000,
                                    1]).reshape((4, 1))
            right_hand_v = np.array([body.joints[15].position.x / 1000,
                                     body.joints[15].position.y / 1000,
                                     body.joints[15].position.z / 1000,
                                     1]).reshape((4, 1))

            trans_ee, rot_ee = self.tf_buffer.lookupTransform('reference_base', 'end_effector', self.get_clock().now())

            T_EndEff = np.eye(4)
            T_EndEff[:3, 3] = trans_ee
            T_EndEff[:3, :3] = R.from_quat(rot_ee).as_matrix()

            T_left_hand = T_EndEff @ self.T_EndEffector_DepthLink @ left_hand_v
            T_right_hand = T_EndEff @ self.T_EndEffector_DepthLink @ right_hand_v

            x_axis = T_right_hand[:3, 0] - T_left_hand[:3, 0]
            x_axis /= np.linalg.norm(x_axis)

            z_axis = np.array([0, 0, -1])
            y_axis = np.cross(z_axis, x_axis)

            T_CenterGrasp = np.eye(4)
            T_CenterGrasp[:3, 3] = (T_left_hand[:3, 0] + T_right_hand[:3, 0]) / 2
            # T_CenterGrasp[:3, :3] = Rot_EndEffector_start
            T_CenterGrasp[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)

            left_hand_tf = TransformStamped()
            left_hand_tf.header.stamp = self.get_clock().now().to_msg()
            left_hand_tf.header.frame_id = 'reference_base'
            left_hand_tf.child_frame_id = 'left_hand'
            left_hand_tf.transform.translation.x = T_left_hand[0, 0]
            left_hand_tf.transform.translation.x = T_left_hand[1, 0]
            left_hand_tf.transform.translation.x = T_left_hand[2, 0]
            left_hand_tf.transform.rotation.x = 0
            left_hand_tf.transform.rotation.y = 0
            left_hand_tf.transform.rotation.z = 0
            left_hand_tf.transform.rotation.w = 1

            self.tf_pub.sendTransform(left_hand_tf)

            right_hand_tf = TransformStamped()
            right_hand_tf.header.stamp = self.get_clock().now().to_msg()
            right_hand_tf.header.frame_id = 'reference_base'
            right_hand_tf.child_frame_id = 'right_hand'
            right_hand_tf.transform.translation.x = T_right_hand[0, 0]
            right_hand_tf.transform.translation.x = T_right_hand[1, 0]
            right_hand_tf.transform.translation.x = T_right_hand[2, 0]
            right_hand_tf.transform.rotation.x = 0
            right_hand_tf.transform.rotation.y = 0
            right_hand_tf.transform.rotation.z = 0
            right_hand_tf.transform.rotation.w = 1
            self.tf_pub.sendTransform(right_hand_tf)

            center_grasp_tf = TransformStamped()
            center_grasp_tf.header.stamp = self.get_clock().now().to_msg()
            center_grasp_tf.header.frame_id = 'reference_base'
            center_grasp_tf.child_frame_id = 'right_hand'
            center_grasp_tf.transform.translation.x = T_CenterGrasp[0, 0]
            center_grasp_tf.transform.translation.x = T_CenterGrasp[1, 0]
            center_grasp_tf.transform.translation.x = T_CenterGrasp[2, 0]
            center_grasp_tf.transform.rotation.x = 0
            center_grasp_tf.transform.rotation.y = 0
            center_grasp_tf.transform.rotation.z = 0
            center_grasp_tf.transform.rotation.w = 1
            self.tf_pub.sendTransform(center_grasp_tf)

    def estimate_deformation(self):

        segmented_depth = self.get_segmented_depth()

        if np.sum(segmented_depth) > 0:

            depth = torch.from_numpy(segmented_depth.reshape(input_shape)).float().to('cuda') / 1000
            with torch.no_grad():
                deformationm_estimation = self.deformation_estimator(depth).squeeze().cpu().detach().numpy()
            #     self.deformation_buffer.append(self.deformation_estimator(depth).squeeze().cpu().detach().numpy())
            #
            # if len(self.deformation_buffer) > self.filter_length:
            #     self.deformation_buffer.pop(0)

            # deformation_estimation = np.mean(deformation_buffer, axis=1)
            # deformation_estimation = self.deformation_buffer[-1] * self.half_range_def + self.mid_def
            deformation_estimation = deformationm_estimation * self.half_range_def + self.mid_def

            self.deformation_msg.header.stamp = self.get_clock().now().to_msg()

            if self.dofs == 3:
                self.deformation_msg.wrench.force.x = deformation_estimation[0]
                self.deformation_msg.wrench.force.y = deformation_estimation[1]
                self.deformation_msg.wrench.force.z = deformation_estimation[2]
            elif self.dofs == 4:
                self.deformation_msg.wrench.force.x = deformation_estimation[0]
                self.deformation_msg.wrench.force.y = deformation_estimation[1]
                self.deformation_msg.wrench.force.z = deformation_estimation[2]
                self.deformation_msg.wrench.torque.z = deformation_estimation[3]
            elif self.dofs == 5:
                self.deformation_msg.wrench.force.x = deformation_estimation[0]
                self.deformation_msg.wrench.force.y = deformation_estimation[1]
                self.deformation_msg.wrench.force.z = deformation_estimation[2]
                self.deformation_msg.wrench.torque.y = deformation_estimation[3]
                self.deformation_msg.wrench.torque.z = deformation_estimation[4]
            else:
                print('Unsupported number of deformation degreeS of freedom')
                exit()

            self.deformation_publisher.publish(self.deformation_msg)

            delta_def = deformation_estimation - self.deformation_setpoint
            delta_def *= np.abs(delta_def) > self.dead_band

            twist = delta_def* self.gains
            self.twist_msg.linear.x = twist[0]
            self.twist_msg.linear.y = twist[1]
            self.twist_msg.linear.z = twist[2]
            self.twist_msg.angular.z = twist[3]

        else:
            self.twist_msg.linear.x = 0.0
            self.twist_msg.linear.y = 0.0
            self.twist_msg.linear.z = 0.0
            self.twist_msg.angular.z = 0.0

        self.twist_publisher.publish(self.twist_msg)


def main():
    rclpy.init()
    node = DeformationNode()
    node.create_timer(timer_period_sec=1 / 30, callback=node.estimate_deformation)

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
