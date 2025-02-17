from launch.launch_description import LaunchDescription
from launch.actions import OpaqueFunction, DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration

from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue

import xacro

def generate_launch_description():
  launch_args = [
    DeclareLaunchArgument(name='robot_ip', description='ur net IP'),
    DeclareLaunchArgument(name='use_fake_hardware', default_value='false', description='use fake hardware'),
    DeclareLaunchArgument(name='prefix', default_value='azrael', description='URDF prefix (without /)'),
  ]

  return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])


def launch_setup(context):

  robot_description_path = PathJoinSubstitution([FindPackageShare('soft_material_deformation_estimation_ros2'), 'urdf', 'ur10_drapebot_fake_gripper.xacro']).perform(context)
  robot_description_args = {
    'robot_ip' : LaunchConfiguration('robot_ip').perform(context),
    'use_fake_hardware' : LaunchConfiguration('use_fake_hardware').perform(context),
    'prefix' : f'{LaunchConfiguration("prefix").perform(context)}/',
  }

  robot_description = xacro.process_file(robot_description_path, mappings=robot_description_args).toprettyxml(indent=' ')

  robot_state_publisher = Node(
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'robot_description' : ParameterValue(value=robot_description, value_type=str)}]
  )

  return [
    robot_state_publisher,
  ]
