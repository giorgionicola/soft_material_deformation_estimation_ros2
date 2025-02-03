import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger  # Replace with your actual service type
from std_msgs.msg import Bool
import beepy


class ServiceCaller(Node):
    def __init__(self):
        super().__init__('service_caller')
        self.service = self.create_service(Trigger, 'ciak', self.ciak_callback)
        self.client = self.create_client(Trigger, 'get_rest_ply_shape')
        self.ciak_publisher = self.create_publisher(Bool, 'action', qos_profile=10)
        self.ciak_msg = Bool()
        self.ciak_msg.data = False

        self.create_timer(timer_period_sec=1/100, callback=self.pub_ciak)

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for target_service...')

        self.get_logger().info("Node started")

    def pub_ciak(self):
        self.ciak_publisher.publish(self.ciak_msg)

    async def ciak_callback(self, request, response : Trigger):
        self.get_logger().info('Received request, calling target_service...')

        req = Trigger.Request()
        future = self.client.call_async(req)

        # Wait for the response asynchronously
        future.add_done_callback(lambda future: self.process_response(future, response))

        response.success = True
        response.message = 'Ciak!!! Si gira'

        return response

    def process_response(self, future, response):
        if future.result().success :
            self.get_logger().info('Acquired rest ply shape, playing sound!')
            self.ciak_msg.data = True
            beepy.beep(4)
        else:
            self.get_logger().error('Error in acquiring rest ply shape')


def main(args=None):
    rclpy.init(args=args)
    node = ServiceCaller()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    while rclpy.ok():
        executor.spin_once(timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
