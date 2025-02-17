import math
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from rclpy.executors import MultiThreadedExecutor



class FramePublisher(Node):

    def __init__(self):
        super().__init__('tf2_frame_publisher')

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer1 = self.create_timer(timer_period_sec=1 / 20, callback=self.pub_tag10)
        self.timer2 = self.create_timer(timer_period_sec=1 / 50, callback=self.pub_tag20)

    def pub_tag10(self):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'azrael/base_link'
        t.child_frame_id = 'tag36h11:10'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = 0.8
        t.transform.translation.y = 0.8
        t.transform.translation.z = 1.0

        # For the same reason, turtle can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        q = R.from_matrix(matrix=np.eye(3)).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def pub_tag20(self):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'azrael/base_link'
        t.child_frame_id = 'tag36h11:20'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = 0.8
        t.transform.translation.y = 1.2
        t.transform.translation.z = 1.0

        # For the same reason, turtle can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        q = R.from_matrix(matrix=np.eye(3)).as_quat()
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)


def main():
    print('hello')

    rclpy.init()
    node = FramePublisher()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    print('hello')
    while rclpy.ok():
        try:
            executor.spin()
        except KeyboardInterrupt:
            break
    print('done')

    rclpy.shutdown()

if __name__ == '__main__':
    main()