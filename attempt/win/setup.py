"""Install Compacter."""
import os
import setuptools


def setup_package():
    long_description = "mwin"
    setuptools.setup(
        name='mwin',
        description='mwin',
        version='0.0.1',
        long_description=long_description,
        license='MIT License',
        packages=setuptools.find_packages(),
        install_requires=[
            'textwrap3',
            'pyperclip'
        ],
    )

if __name__ == '__main__':
    setup_package()
