from setuptools import setup, find_packages

setup(
    name='LogicalOperatorMachines',
    version='0.2',
    author='Tammo Rukat',
    author_email='tammorukat@gmail.com',
    packages=find_packages(exclude=('tests')),
    install_requires=['numpy', 'numba']
)
