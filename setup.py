from setuptools import setup, find_packages

readme = open("README.md").read()

setup(
    name="mimic_head",
    version="0.1.5",
    description="Unofficial One-click Version of LivePortrait, with Webcam Support",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "gradio",
         'numpy',
        'requests',
        "opencv-python",
        "onnxruntime",
        "onnx",
        "scikit-image",

    ],
    package_data={
        "mimic_head": [
            'config/models.yaml',
        ]
    },
    entry_points={
        "console_scripts": [
            "mimic_head=mimic_head.cli:cli",
        ],
    },
)
