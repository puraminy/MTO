"""Install Compacter."""
import os
import setuptools


def setup_package():
    long_description = "attempt"
    setuptools.setup(
        name='attempt',
        description='ATTEMPT',
        version='0.0.1',
        long_description=long_description,
        license='MIT License',
        packages=setuptools.find_packages(
            exclude=['docs', 'tests', 'scripts', 'examples']),
        install_requires=[
            'datasets',
            'scikit-learn',
            'tensorboard',
            'matplotlib',
            'torch',
            'transformers',
            'tqdm',
            'deepdiff',
            'rouge_score'
        ],
    )

if __name__ == '__main__':
    setup_package()
