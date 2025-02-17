import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
import tf2_ros
from tf2_ros.buffer import Buffer
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation as R


class MoveToStartNode(Node):
    def __init__(self):

        super().__init__('move_to_start')

        self.first_time = True
        self.logger = self.get_logger()

        self.start_distance = 0.85
        self.half_fake_hand_size = 0.17

        # self.tf_pub = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        self.base_frame = 'azrael/base_link'
        # self.base_frame = 'camera_base'

        self.target_frame_pub = self.create_publisher(topic='/cartesian_motion_controller/target_frame',
                                                      msg_type=PoseStamped,
                                                      qos_profile=10)

        self.got_tags = False
        self.tag10_pose = np.eye(4)
        self.tag20_pose = np.eye(4)

    def get_tag_pose(self):
        while not self.got_tags:
            if not self.tf_buffer.can_transform(self.base_frame,
                                                'tag36h11:10',
                                                # 'camera_base',
                                                rclpy.time.Time(), Duration(seconds=0.1)):
                self.logger.warning('Can\'t find tag_10')
                return
            if not self.tf_buffer.can_transform(self.base_frame, 'tag36h11:20', rclpy.time.Time(), Duration(seconds=5)):
                self.logger.warning('Can\'t find tag_20')
                return
            # try:
            t = self.tf_buffer.lookup_transform(self.base_frame, 'tag36h11:10', rclpy.time.Time())
            self.tag10_pose[:3, 3] = [t.transform.translation.x,
                                      t.transform.translation.y,
                                      t.transform.translation.z]

            self.tag10_pose[:3, :3] = R.from_quat([t.transform.rotation.x,
                                                   t.transform.rotation.y,
                                                   t.transform.rotation.z,
                                                   t.transform.rotation.w]).as_matrix()

            t = self.tf_buffer.lookup_transform(self.base_frame, 'tag36h11:20', rclpy.time.Time())
            self.tag20_pose[:3, 3] = [t.transform.translation.x,
                                      t.transform.translation.y,
                                      t.transform.translation.z]

            self.tag20_pose[:3, :3] = R.from_quat([t.transform.rotation.x,
                                                   t.transform.rotation.y,
                                                   t.transform.rotation.z,
                                                   t.transform.rotation.w], ).as_matrix()

            self.got_tags = True
        return

    def estimate_start_pose(self):

        if self.got_tags:
            middle_frame = np.eye(4)
            middle_frame_pos = (self.tag10_pose[:3, 3] + self.tag20_pose[:3, 3]) / 2
            # middle_frame_vx = (tag20_pose[:3, 3] - tag10_pose[:3, 3])
            middle_frame_vx = (self.tag10_pose[:3, 3] - self.tag20_pose[:3, 3])
            middle_frame_vx[2] = 0
            middle_frame_vx /= np.linalg.norm(middle_frame_vx)

            middle_frame_vz = np.array([0, 0, -1])
            middle_frame_vy = np.cross(middle_frame_vz, middle_frame_vx)

            middle_frame[:3, 0] = middle_frame_vx
            middle_frame[:3, 1] = middle_frame_vy
            middle_frame[:3, 2] = middle_frame_vz
            middle_frame[:3, 3] = middle_frame_pos

            static_transformStamped = TransformStamped()

            static_transformStamped.header.stamp = rclpy.time.Time().to_msg()
            static_transformStamped.header.frame_id = self.base_frame
            static_transformStamped.child_frame_id = 'middle_frame'
            static_transformStamped.transform.translation.x = middle_frame_pos[0]
            static_transformStamped.transform.translation.y = middle_frame_pos[1]
            static_transformStamped.transform.translation.z = middle_frame_pos[2]

            quat = R.from_matrix(middle_frame[:3, :3]).as_quat()
            static_transformStamped.transform.rotation.x = quat[0]
            static_transformStamped.transform.rotation.y = quat[1]
            static_transformStamped.transform.rotation.z = quat[2]
            static_transformStamped.transform.rotation.w = quat[3]
            # self.tf_pub.sendTransform(static_transformStamped)

            ee_target_pos = [0, self.start_distance + 0.15, 0]
            ee_target_rot = [0, 0, np.deg2rad(180)]

            ee_target = np.eye(4)
            ee_target[:3, :3] = R.from_euler(angles=ee_target_rot, seq='xyz').as_matrix()
            ee_target[:3, 3] = ee_target_pos
            ee_target = middle_frame @ ee_target
            rot_ee_target = R.from_matrix(ee_target[:3, :3]).as_euler(seq='xyz', degrees=True)

            middle_frame_to_right_hand = np.eye(4)
            middle_frame_to_right_hand[:2, 3] = [-0.165, 0.15]
            right_hand = middle_frame @ middle_frame_to_right_hand

            static_transformStamped2 = TransformStamped()
            static_transformStamped2.header.stamp = rclpy.time.Time().to_msg()
            static_transformStamped2.header.frame_id = self.base_frame
            static_transformStamped2.child_frame_id = 'right_hand'
            static_transformStamped2.transform.translation.x = right_hand[0, 3]
            static_transformStamped2.transform.translation.y = right_hand[1, 3]
            static_transformStamped2.transform.translation.z = right_hand[2, 3]
            quat = R.from_matrix(right_hand[:3, :3]).as_quat()
            static_transformStamped2.transform.rotation.x = quat[0]
            static_transformStamped2.transform.rotation.y = quat[1]
            static_transformStamped2.transform.rotation.z = quat[2]
            static_transformStamped2.transform.rotation.w = quat[3]
            # self.tf_pub.sendTransform(static_transformStamped2)

            middle_frame_to_left_hand = np.eye(4)
            middle_frame_to_left_hand[:2, 3] = [0.165, 0.15]
            left_hand = middle_frame @ middle_frame_to_left_hand

            static_transformStamped2 = TransformStamped()
            static_transformStamped2.header.stamp = rclpy.time.Time().to_msg()
            static_transformStamped2.header.frame_id = self.base_frame
            static_transformStamped2.child_frame_id = 'left_hand'
            static_transformStamped2.transform.translation.x = left_hand[0, 3]
            static_transformStamped2.transform.translation.y = left_hand[1, 3]
            static_transformStamped2.transform.translation.z = left_hand[2, 3]
            quat = R.from_matrix(left_hand[:3, :3]).as_quat()
            static_transformStamped2.transform.rotation.x = quat[0]
            static_transformStamped2.transform.rotation.y = quat[1]
            static_transformStamped2.transform.rotation.z = quat[2]
            static_transformStamped2.transform.rotation.w = quat[3]
            # self.tf_pub.sendTransform(static_transformStamped2)

            static_transformStamped2 = TransformStamped()
            static_transformStamped2.header.stamp = rclpy.time.Time().to_msg()
            static_transformStamped2.header.frame_id = self.base_frame
            static_transformStamped2.child_frame_id = 'target_frame'
            static_transformStamped2.transform.translation.x = ee_target[0, 3]
            static_transformStamped2.transform.translation.y = ee_target[1, 3]
            static_transformStamped2.transform.translation.z = ee_target[2, 3]
            quat = R.from_matrix(ee_target[:3, :3]).as_quat()
            static_transformStamped2.transform.rotation.x = quat[0]
            static_transformStamped2.transform.rotation.y = quat[1]
            static_transformStamped2.transform.rotation.z = quat[2]
            static_transformStamped2.transform.rotation.w = quat[3]
            # self.tf_pub.sendTransform(static_transformStamped2)

            ee_to_rhand = np.linalg.inv(ee_target) @ right_hand
            distance_ee_to_rhand = np.linalg.norm(ee_to_rhand[:3, 3])
            rot_ee_to_rhand = R.from_matrix(ee_to_rhand[:3, :3]).as_euler(seq='xyz', degrees=True)

            ee_to_lhand = np.linalg.inv(ee_target) @ left_hand
            distance_ee_to_lhand = np.linalg.norm(ee_to_lhand[:3, 3])
            rot_ee_to_lhand = R.from_matrix(ee_to_lhand[:3, :3]).as_euler(seq='xyz', degrees=True)

            middle_frame_to_center_hand = np.eye(4)
            middle_frame_to_center_hand[1, 3] = 0.15
            center_hand = middle_frame @ middle_frame_to_center_hand

            ee_to_chand = np.linalg.inv(ee_target) @ center_hand
            rot_ee_to_chand = R.from_matrix(ee_to_chand[:3, :3]).as_euler(seq='xyz', degrees=True)

            if self.first_time:
                print(f'Reference base to End effector: \n'
                      f'\tTranslation {ee_target[:3, 3]}\n'
                      f'\tRotation Euler xyz: {rot_ee_target}')

                print(f'End effector to Left Hand: \n'
                      f'\tTranslation {ee_to_lhand[:3, 3]}\n'
                      f'\tRotation Euler xyz: {rot_ee_to_lhand}')

                print(f'End effectrot  to Right Hand: \n'
                      f'\tTraslation {ee_to_rhand[:3, 3]}\n'
                      f'\tRotation Euler xyz: {rot_ee_to_rhand}')

                print(f'Grasp Position: \n{center_hand}')
                print(f'End effector to Grasp Position: \n'
                      f'\tTraslation {ee_to_chand[:3, 3]}\n'
                      f'\tRotation Euler xyz: {rot_ee_to_chand}')

                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = self.base_frame
                pose_msg.pose.position.x = ee_target[0, 3]
                pose_msg.pose.position.y = ee_target[1, 3]
                pose_msg.pose.position.z = ee_target[2, 3]
                pose_msg.pose.orientation.x = quat[0]
                pose_msg.pose.orientation.y = quat[1]
                pose_msg.pose.orientation.z = quat[2]
                pose_msg.pose.orientation.w = quat[3]
                self.target_frame_pub.publish(pose_msg)
                self.destroy_publisher(self.target_frame_pub)
                self.first_time = False
        else:
            return



if __name__ == '__main__':
    rclpy.init()
    node = MoveToStartNode()

    time.sleep(2)
    # node.create_timer(timer_period_sec=1 / 30, callback=node.estimate_deformation)
    # node.create_timer(timer_period_sec=1 / 30, callback=node.estimate_start_pose)

    while rclpy.ok():
        node.get_tag_pose()
        node.estimate_start_pose()

