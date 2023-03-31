import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger


class MoveToNextNode(Node):
    def __init__(self):
        super().__init__('compare_methods')

        self.logger = self.get_logger()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_pub = tf2_ros.TransformBroadcaster(self)
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.service = self.create_service(Trigger, 'next_pose', self.move_to_next_pose)

        self.target_frame_pub = self.create_publisher(topic='/my_cartesian_motion_controller/target_frame',
                                                      msg_type=PoseStamped,
                                                      qos_profile=10)

        self.target_poses = [[0,0,0,0],
                             [0, -0.1, 0, 10],
                             [-0.15, -0.05, 0, -10],
                             [0.1, 0, -0.05, 0],
                             [-0.2, 0, 0.2, 15],
                             [0.1, -0.05, 0, -10],
                             [-0.1, 0, 0.05, 0],
                             [0., -0.2, -0.2, 5],
                             [0., -0.1, 0.2, 5],
                             [0.05, -0.2, -0.05, 15],
                             [0., -0.05, 0.1, 5],
                             ]
        self.base_frame = 'azrael/base'

        self.T_start = np.eye(4)
        self.T_start[:3, 3] = [-0.09237911, -0.54523431, 0.35505615]
        self.T_start[:3, :3] = R.from_euler('xyz', [180, 0, 5.55070477], degrees=True).as_matrix()

    def move_to_next_pose(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:

        if len(self.target_poses) > 0:

            target = np.eye(4)
            target[:3, 3] = self.target_poses[0][:3]
            target[:3, :3] = R.from_euler(seq='xyz', angles=[0, 0, self.target_poses[0][3]], degrees=True).as_matrix()
            target = self.T_start @ target

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

            while not self.tf_buffer.can_transform(self.base_frame, 'tcp', rclpy.time.Time(), Duration(seconds=0.1)):
                pass

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
                tcp = np.linalg.inv(self.T_start) @ tcp
                lin_dist = np.linalg.norm(tcp[:3, 3] - self.target_poses[0][:3])
                ang = np.rad2deg(R.from_matrix(tcp[:3, :3]).as_euler('xyz')[2])
                ang_dist = np.abs(self.target_poses[0][3] - ang)

                self.logger.info(f'{lin_dist}\t {ang_dist}')

                if lin_dist < 0.005 and ang_dist < 1:
                    self.logger.info('arrivato')
                    break

            self.target_poses.pop(0)
            response.success= True
        else:
            response.success= False
            response.message = 'Pose finite'

        return response


def main():
    rclpy.init()
    node = MoveToNextNode()

    excutor = rclpy.executors.MultiThreadedExecutor()
    excutor.add_node(node)
    excutor.spin()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
