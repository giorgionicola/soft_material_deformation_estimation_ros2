# Copyright 2023 ros2_control Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory
from launch.launch_context import LaunchContext
from launch.conditions import IfCondition
from launch_ros.descriptions import ParameterValue
import xacro


def generate_launch_description():

        
    urdf_file_name = 'ur10_drapebot_fake_gripper.xacro'
    path_to_xacro = os.path.join(get_package_share_directory('soft_material_deformation_estimation_ros2'), 'urdf', urdf_file_name)
    
      
    doc = xacro.process_file(path_to_xacro)
    robot_description_config = doc.toxml()

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{'robot_description': robot_description_config}],
    )
    
    jsg = Node(
    package="joint_state_publisher_gui",
    executable="joint_state_publisher_gui",
    output="both",
    parameters=[{'robot_description': robot_description_config}],
    )

    rviz = Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
        )

    nodes = [
        robot_state_pub_node,
        rviz,
        jsg
    ]


    return LaunchDescription(nodes)
