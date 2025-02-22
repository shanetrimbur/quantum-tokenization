from setuptools import setup, find_packages

setup(
    name="quantum-compression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "qiskit",
        "qiskit-aer",
        "plotly",
        "scipy",
        "tqdm",
        "requests",
        "transformers",
        "tokenizers",
        "torch"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Quantum-enhanced text compression using harmonic principles",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-compression",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 