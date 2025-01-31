import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger
import numpy as np
import time


class FakeDeformationEstimator(Node):
    def __init__(self):
        super().__init__('fake_deformation_publisher')

        self.prob_def_pub = self.create_publisher(topic='deformation_probabilities',
                                                  msg_type=Float32MultiArray,
                                                  qos_profile=10)

        self.classes_def_pub = self.create_publisher(topic='deformation_classes',
                                                     msg_type=Int32MultiArray,
                                                     qos_profile=10)

        self.ply_rest_shape_client = self.create_service(srv_type=Trigger,
                                                         srv_name='get_rest_ply_shape',
                                                         callback=self.get_rest_ply_shape)

        self.start_publish_deformation = False
        self.publish_probabilities = True
        self.time_start = None
        self.previous_pose = np.eye(4)
        self.step = 0.02
        self.center = 0

        self.get_logger().info("Node started")

    def estimate_deformation(self):
        if self.start_publish_deformation:
            width = 0.3
            indices = np.arange(-2, 3, 1)
            # Compute Gaussian values for each index
            if self.center + self.step > +2 or self.center + self.step < -2:
                self.step = -self.step
            self.center += self.step

            gaussian = np.exp(-0.5 * ((indices - self.center) / width) ** 2)
            # Normalize to ensure the sum equals 1
            normalized = gaussian / np.sum(gaussian)

            prob_deformation = np.array([normalized.tolist(),
                                         normalized.tolist(),
                                         normalized.tolist(),
                                         normalized.tolist()])

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

    def get_rest_ply_shape(self, request, response: Trigger.Response):
        self.start_publish_deformation = True
        response.success = True
        self.get_logger().info("Acquired rest shape")
        return response


def main():
    rclpy.init()
    node = FakeDeformationEstimator()
    time.sleep(5)

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
