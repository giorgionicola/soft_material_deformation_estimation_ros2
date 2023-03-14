from setuptools import setup

package_name = 'soft_material_deformation_estimation_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'azure_kinect=soft_material_deformation_estimation_ros2.azure_kinect'
        ],
    },
)
