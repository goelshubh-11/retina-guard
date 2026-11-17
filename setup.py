from setuptools import setup, find_packages

setup(
    name        = "retina_guard",
    version     = "1.0.0",
    description = "AI-powered Diabetic Retinopathy detection and progression monitoring",
    packages    = find_packages(),
    python_requires = ">=3.10",
    install_requires = open("requirements.txt").read().splitlines(),
)
