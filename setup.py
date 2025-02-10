"""Install Compacter."""
import os
import setuptools


def setup_package():
    long_description = "mto"
    setuptools.setup(
        name='mto',
        description='MTO',
        version='0.0.1',
        long_description=long_description,
        license='MIT License',
        packages=setuptools.find_packages(
            exclude=['docs', 'tests', 'scripts', 'examples']),
        install_requires=[
        ],
    )

if __name__ == '__main__':
    setup_package()
