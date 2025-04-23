

from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f if line and not line.startswith("#")]

setup(
    name='daphdetector',
    version='0.1.0',
    description='Daphnia body part detector and cropping pipeline for scientifc studies',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=install_requires,
)

