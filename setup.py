from setuptools import setup, find_packages

setup(
    name='pdqp_wrap',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'juliacall',
        'scipy',
        # Add other Python dependencies here
    ],
    include_package_data=True,
    package_data={
        'pdqp_wrap': ['src/*.jl']
    },
    description='A module for PDQP wrap functionality.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://your.project.url',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
