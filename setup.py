from setuptools import setup, find_packages

setup(
    name="sfplot",
    version="0.0.2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "sfplot-gui = sfplot.gui.gui_app:main",
        ],
    },
    # … 其它配置 …
)
