# setup.py

from setuptools import setup, find_packages

setup(
    name="sfplot",
    version="0.1.0",
    description="For plotting the Search and Find Plot (SFplot).",
    author="Taobo Hu",
    author_email="taobo.hu@scilifelab.se",
    url="https://github.com/hutaobo/sfplot",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scanpy",
        "scikit-learn",
        "spatialdata_io",  # 确保这些依赖在 PyPI 上可用，或在 README 中说明
        # 其他依赖
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
