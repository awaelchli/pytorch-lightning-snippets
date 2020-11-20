from setuptools import setup, find_packages

setup(
    name="pytorch-lightning-snippets",
    version="0.0.5",
    author="Adrian Wälchli",
    author_email="aedu.waelchli@gmail.com",
    scripts=[],
    entry_points={
        "console_scripts": [
            "peek = checkpoint.peek:main",
        ],
    },
    url="https://github.com/awaelchli/pytorch-lightning-snippets.git",
    license="LICENSE.md",
    description="A collection of useful tools around PyTorch Lightning.",
    long_description=open("README.md").read(),
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=[
        "pytorch-lightning>=0.9.0",
    ],
)
