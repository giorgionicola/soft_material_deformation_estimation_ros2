import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from collections import deque
import time
import numpy as np


class DeformationToTwistNode(Node):
    def __init__(self):
        super().__init__('twist_publisher')

        self.window_size = 5
        self.dofs = 4
        self.n_classes = 5
        self.twist_conv = np.array([[-0.2, -0.05, 0, 0.05, 0.2],
                                    [-0.2, -0.05, 0, 0.05, 0.2],
                                    [-0.2, -0.05, 0, 0.05, 0.2],
                                    [-0.2, -0.05, 0, 0.05, 0.2]])
        self.dead_band = np.array([0.02, 0.02, 0.02, 0.02])

        self.prob_deformation = np.zeros((self.dofs, self.n_classes ))

        # Subscriber to the Float32MultiArray topic
        self.def_listener = self.create_subscription(msg_type=Float32MultiArray,
                                                     topic='/deformation_probabilities',
                                                     callback=self.listener_callback,
                                                     qos_profile=10)

        # Publisher for the Twist topic
        self.twist_publisher = self.create_publisher(topic='/imm/commands',
                                                     msg_type=Twist,
                                                     qos_profile=10)

        # Moving average buffers for linear and angular velocity
        self.twist_buffer = [deque([0] * self.window_size, maxlen=self.window_size) for _ in range(self.dofs)]

        # Time of the last received message
        self.last_message_time = -np.inf

        # Timeout threshold (in seconds) for resetting the Twist command
        self.message_timeout = 1.0

        # Timer for publishing Twist messages at a fixed frequency
        timer_period = 1.0 / 125  # Frequency of 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_twist)

        self.get_logger().info("Node started")
        self.last_log_time = 0.0  # Variable to track the last time we logged
        self.log_interval = 0.5  # Log every 1 second (adjust as needed)

    def initialize_buffer(self):
        self.twist_buffer = [deque([0] * self.window_size, maxlen=self.window_size) for _ in range(self.dofs)]

    def listener_callback(self, msg: Float32MultiArray):
        # Process the received data from the Float64MultiArray
        data = msg.data
        self.last_message_time = time.time()

        if msg.layout.dim[0].size != self.dofs:
            self.get_logger().error(
                f'Received data mismatch in dofs. Expected {self.dofs} and received in metadata {msg.layout.dim[0].size}')
            return
        if msg.layout.dim[1].size != self.twist_conv.shape[1]:
            self.get_logger().error(f'Received data mismatch in n classes. Expected {self.twist_conv.shape[1]} and '
                                    f'received in metadata {msg.layout.dim[1].size}')
            return

        for i in range(self.dofs):
            self.prob_deformation[i] = data[i * self.n_classes: (i + 1) * self.n_classes]
            self.twist_buffer[i].append(np.dot(self.prob_deformation[i], self.twist_conv[i]))

    def publish_twist(self):
        twist_msg = Twist()

        # Check if the last message is too old
        current_time = time.time()
        if current_time - self.last_message_time > self.message_timeout:
            self.initialize_buffer()
            twist_msg.linear.x = 0.0
            twist_msg.linear.y = 0.0
            twist_msg.linear.z = 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = 0.0
            if current_time - self.last_log_time >= self.log_interval:
                self.get_logger().warn(f'Deformation estimation too old')
                self.last_log_time = current_time
        else:

            twist = np.array([sum(t_buffer)/self.window_size for t_buffer in self.twist_buffer])
            twist *= (np.abs(twist) > self.dead_band)

            twist_msg.linear.x = twist[0]
            twist_msg.linear.y = twist[1]
            twist_msg.linear.z = twist[2]
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = twist[3]

        # Publish the Twist command
        self.twist_publisher.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)

    node = DeformationToTwistNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)


    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
