import os.path
import time
import cv2
import tf2_ros
from tf2_ros.buffer import Buffer
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
import pykinect_azure as pykinect
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import torch


def make_mask(crop_size) -> np.ndarray:
    pixel_start_diagonal = 100
    left_diagonal = lambda x: np.tan(np.deg2rad(90 - 30)) * x + (
                512 - np.tan(np.deg2rad(90 - 30)) * pixel_start_diagonal)
    left_diagonal_mask = np.ones((512, 512))
    for x in range(pixel_start_diagonal):
        for y in range(512):
            if y > left_diagonal(x):
                left_diagonal_mask[y, x] = 0

    right_diagonal = lambda x: -np.tan(np.deg2rad(60)) * x + (
                512 + np.tan(np.deg2rad(60)) * (512 - pixel_start_diagonal))

    right_diagonal_mask = np.ones((512, 512))
    for x in range(512 - pixel_start_diagonal, 512):
        for y in range(512):
            if y > right_diagonal(x):
                right_diagonal_mask[y, x] = 0

    top_side_mask = np.ones((512, 512))
    top_side_mask[:150, :] = 0

    bottom_side_mask = np.ones((512, 512))
    bottom_side_mask[450:, :] = 0

    mask = left_diagonal_mask * right_diagonal_mask * top_side_mask * bottom_side_mask

    return mask[crop_size[0][0]: crop_size[0][1], crop_size[1][0]: crop_size[1][1]]


class ImageCollector(Node):
    def __init__(self):
        super().__init__('image_collector')

        self.logger = self.get_logger()

        self.resolution = (224, 224)

        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
        device_config.synchronized_images_only = False
        pykinect.initialize_libraries(track_body=False)

        self.azure = pykinect.start_device(config=device_config)

        # mettere la rototraslazione iniziale uomo-robot usata per move_to_start_pose.py
        self.start_label = np.array([0, 0.75, 0, 0, -180])

        # mettere la rotostralazione ottenuta tramite move_to_start_pose.py
        self.start_pose = np.eye(4)
        self.start_pose[:3, 3] = [-0.06075623,  0.58222196,  0.55929441]
        self.start_pose[:3, :3] = R.from_euler('xyz', [180, 0,-178.55681831], degrees=True).as_matrix()
        self.base_frame = 'azrael/base_link'

        delta_label = np.random.uniform(low=[-0.15, -0.15, -0.15, -20, -30],
                                        high=[0.15, 0.15, 0.15, 20, 30],
                                        size=[50, 5])

        self.tf_buffer = Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        while not self.tf_buffer.can_transform('azrael/base_link',
                                               'end_effector',
                                               time=self.get_clock().now(),
                                               timeout=Duration(seconds=5)):
            self.logger.warning('Can\'t find end_effector')

        self.labels = self.start_label + delta_label

        segmentation_model_path = '/home/kildall/ros2_projects/deformable_ws/src/soft_material_deformation_estimation_ros2/models/segmentation/best_model.pth'

        self.segmentation_threshold = 0.5
        self.nn_segmentor = torch.load(segmentation_model_path)
        self.nn_segmentor.eval()
        self.device = 'cuda'

        self.depth_threshold = 1300
        self.crop_size: tuple = ((90, 512), (50, 472))
        self.static_mask = make_mask(self.crop_size)
        a = (900 - 1600) / (450 - 200)
        b = 1600 - 200 * a
        depth_line = lambda y: a * y + b
        # depth_matrix = np.ones((512, 512))
        # for col in range(512):
        #     depth_matrix[:, col] *= depth_line(np.array([v for v in range(512)]))
        # self.depth_matrix = depth_matrix[self.crop_size[0][0]: self.crop_size[0][1],
        #                     self.crop_size[1][0]: self.crop_size[1][1]]

        self.target_poses = []
        for dl in delta_label:
            delta_pose = np.eye(4)
            delta_pose[:3, 3] = dl[:3]
            delta_pose[:3, :3] = R.from_euler('xyz', [0, dl[3], dl[4]], degrees=True).as_matrix()

            # Verificare in che frame sono definite le label qui stiamo assumendo che siano
            self.target_poses.append(self.start_pose @ delta_pose)

        self.target_frame_pub = self.create_publisher(topic='/cartesian_motion_controller/target_frame',
                                                      msg_type=PoseStamped,
                                                      qos_profile=10)

        self.folder = '/home/kildall/Desktop/ply_carbonio'
        self.logger.info(f'Saving at: {self.folder}')
        os.makedirs(self.folder, exist_ok=False)

    def move_to_pose(self, target):

        target_msg = PoseStamped()
        target_msg.header.stamp = rclpy.time.Time().to_msg()
        target_msg.header.frame_id = self.base_frame
        target_msg.pose.position.x = target[0, 3]
        target_msg.pose.position.y = target[1, 3]
        target_msg.pose.position.z = target[2, 3]
        quat = R.from_matrix(target[:3, :3]).as_quat()
        target_msg.pose.orientation.x = quat[0]
        target_msg.pose.orientation.y = quat[1]
        target_msg.pose.orientation.z = quat[2]
        target_msg.pose.orientation.w = quat[3]
        self.target_frame_pub.publish(target_msg)

        counter = 0
        while not self.tf_buffer.can_transform(self.base_frame, 'end_effector', self.get_clock().now(),
                                               Duration(seconds=0.2)):
            counter += 1
            if counter > 100:
                raise RuntimeError()

        while True:
            t = self.tf_buffer.lookup_transform(self.base_frame, 'end_effector', rclpy.time.Time())
            tcp = np.eye(4)
            tcp[0, 3] = t.transform.translation.x
            tcp[1, 3] = t.transform.translation.y
            tcp[2, 3] = t.transform.translation.z
            tcp[:3, :3] = R.from_quat([t.transform.rotation.x,
                                       t.transform.rotation.y,
                                       t.transform.rotation.z,
                                       t.transform.rotation.w]).as_matrix()
            lin_dist = np.linalg.norm(tcp[:3, 3] - target[:3, 3])
            delta_ang = np.rad2deg(np.abs(R.from_matrix(target[:3, :3]).as_euler('xyz') -
                                          R.from_matrix(tcp[:3, :3]).as_euler('xyz')))
            delta_ang = [d if d < 360 else d - 360 for d in delta_ang[1:]]
            ang_dist = np.sum(delta_ang)


            self.logger.info(f'{lin_dist}\t {ang_dist}', throttle_duration_sec=0.5)

            if lin_dist < 0.005 and ang_dist < 1:
                self.logger.info('arrivato')
                break

    def get_depth_image(self, label):
        capture = self.azure.update()
        _, depth = capture.get_depth_image()

        depth = depth[self.crop_size[0][0]: self.crop_size[0][1], self.crop_size[1][0]: self.crop_size[1][1]]
        depth *= depth < self.depth_threshold
        depth = depth * self.static_mask
        # depth *= depth < self.depth_matrix
        depth = cv2.resize(depth, dsize=self.resolution)

        tensor_depth = torch.from_numpy(depth.reshape(1, 1, self.resolution[0], self.resolution[1])).float().to(
            self.device) / 1000

        mask = self.nn_segmentor(tensor_depth).squeeze().detach().to('cpu').numpy() > self.segmentation_threshold
        depth *= mask

        np.save(os.path.join(self.folder, f'{label[0]}_{label[1]}_{label[2]}_{label[3]}_{label[4]}'),
                depth)


def main():
    rclpy.init()
    node = ImageCollector()

    for p, (pose, label) in enumerate(zip(node.target_poses, node.labels)):
        node.logger.info(f'Pose {p + 1}/{len(node.target_poses)}')

        node.move_to_pose(target=pose)

        time.sleep(1)

        node.get_depth_image(label=label)


if __name__ == '__main__':
    main()
