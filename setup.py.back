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
            'sentence-transformers==2.2.2',
            'transformers==4.24.0',
            'datasets==2.14.6',
            'scikit-learn', 
            'matplotlib==3.4.2',
            'torch',
            'tqdm==4.65.0', 
            'deepdiff',
            'rouge_score'
        ],
    )

if __name__ == '__main__':
    setup_package()
