import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger  # Replace with your actual service type
import beepy


class ServiceCaller(Node):
    def __init__(self):
        super().__init__('service_caller')
        self.service = self.create_service(Trigger, 'ciak', self.ciak_callback)
        self.client = self.create_client(Trigger, 'get_rest_ply_shape')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for target_service...')

        self.get_logger().info("Node started")

    async def ciak_callback(self, request, response):
        self.get_logger().info('Received request, calling target_service...')

        req = Trigger.Request()
        future = self.client.call_async(req)

        # Wait for the response asynchronously
        out = future.add_done_callback(lambda future: self.process_response(future, response))

        response.success = True
        response.message = 'Ciak!!! Si gira'

        return response

    def process_response(self, future, response):
        if future.result().success :
            self.get_logger().info('Acquired rest ply shape, playing sound!')
            beepy.beep(4)
        else:
            self.get_logger().error('Error in acquiring rest ply shape')


def main(args=None):
    rclpy.init(args=args)
    node = ServiceCaller()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # ✅ Spin periodically to allow the callbacks to run
    while rclpy.ok():
        executor.spin_once(timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
