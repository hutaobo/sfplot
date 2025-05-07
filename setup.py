# setup.py

from setuptools import setup, find_packages

setup(
    # … 其它配置 …
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "sfplot-gui = sfplot.gui.gui_app:main",
        ],
    },
    # … 其它配置 …
)
