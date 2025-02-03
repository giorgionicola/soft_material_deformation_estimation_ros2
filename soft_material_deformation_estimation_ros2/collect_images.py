import os.path
import time

import tf2_ros
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
import pykinect_azure as pykinect
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R


class ImageCollector(Node):
    def __init__(self):
        super().__init__('image_collector')

        self.folder =''

        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        device_config.synchronized_images_only = False
        pykinect.initialize_libraries(track_body=False)

        self.azure = pykinect.start_device(config=device_config)

        # mettere la rototraslazione iniziale uomo-robot usata per move_to_start_pose.py
        self.start_label = np.array([0, 1.0, 0, 180])

        # mettere la rotostralazione ottenuta tramite move_to_start_pose.py
        self.start_pose = np.eye(4)
        self.start_pose[:3, 3] = [-0.09237911, -0.54523431, 0.35505615]
        self.start_pose[:3, :3] = R.from_euler('xyz', [180, 0, 5.55070477], degrees=True).as_matrix()
        self.base_frame = 'azrael/base'

        delta_label = np.random.uniform(low=[-0.2, -0.4, -0.2, -30],
                                        high=[0.2, 0, -0.2, 30],
                                        size=50)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_pub = tf2_ros.TransformBroadcaster(self)
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.labels = self.start_label + delta_label

        self.target_poses = []
        for dl in delta_label:
            delta_pose = np.eye(4)
            delta_pose[:3,3] = dl[:3]
            delta_pose[:3,:3] = R.from_euler('xzy', dl[3], degrees=True)

            # Verificare in che frame sono definite le label qui stiamo assumendo che siano
            self.target_poses.append(self.start_pose @ delta_pose)

        self.target_frame_pub = self.create_publisher(topic='/my_cartesian_motion_controller/target_frame',
                                                      msg_type=PoseStamped,
                                                      qos_profile=10)

        self.logger = self.get_logger()

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
        while not self.tf_buffer.can_transform(self.base_frame, 'tcp', rclpy.time.Time(), Duration(seconds=0.1)):
            counter +=1
            if counter > 100:
                raise RuntimeError()

        while True:
            t = self.tf_buffer.lookup_transform(self.base_frame, 'tcp', rclpy.time.Time())
            tcp = np.eye(4)
            tcp[0, 3] = t.transform.translation.x
            tcp[1, 3] = t.transform.translation.y
            tcp[2, 3] = t.transform.translation.z
            tcp[:3, :3] = R.from_quat([t.transform.rotation.x,
                                       t.transform.rotation.y,
                                       t.transform.rotation.z,
                                       t.transform.rotation.w]).as_matrix()
            lin_dist = np.linalg.norm(tcp[:3, 3] - self.target_poses[0][:3])
            ang_dist = np.rad2deg(np.abs(R.from_matrix(self.target_poses[:3, :3]).as_euler('xyz')[2] -
                                         R.from_matrix(tcp[:3, :3]).as_euler('xyz')[2]))

            self.logger.info(f'{lin_dist}\t {ang_dist}')

            if lin_dist < 0.005 and ang_dist < 1:
                self.logger.info('arrivato')
                break

    def get_depth_image(self):
       capture = self.azure.update()
       _, depth = capture.get_depth_image()

       np.save(os.path.join(self.folder, f'{self.labels[0]}_{self.labels[1]}_{self.labels[2]}_{self.labels[3]}'))

def main():
    node = ImageCollector()

    for p, pose in enumerate(node.target_poses):
        node.logger.info(f'Pose {p+1}/{len(node.target_poses)}')

        node.move_to_pose(target=pose)

        time.sleep(1)

        node.get_depth_image()


if __name__ == '__main__':
    main()
