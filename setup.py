from setuptools import setup
from glob import glob

package_name = 'soft_material_deformation_estimation_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mullis',
    maintainer_email='giorgio.nicola@stiima.cnr.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'deformation_estimation=soft_material_deformation_estimation_ros2.deformation_estimation:main',
            'twist_generator=soft_material_deformation_estimation_ros2.twist_generator:main',
            'move_to_start_pose=soft_material_deformation_estimation_ros2.move_to_start_pose:main',
            'move_to_test_poses=soft_material_deformation_estimation_ros2.move_to_test_poses:main',
            'compare_methods=soft_material_deformation_estimation_ros2.compare_methods:main'
        ],
    },
)
