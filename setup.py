# setup.py
from setuptools import setup, find_packages
# import pimm
# from os import path
# this_directory = path.abspath(path.dirname(__file__))
# long_description = None
# with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

# 项目元数据
setup(
    name="PalmScore",  # 项目名称
    version="0.1.0",   # 版本号
    author="Laip11",  # 作者名
    author_email="laip1025@gmail.com",  # 作者邮箱
    description="A package for PalmScore project",  # 简短描述
    long_description=open("README.md", "r", encoding="utf-8").read(),  # 长描述（通常是 README 文件）
    long_description_content_type="text/markdown",  # 长描述格式
    url="https://github.com/Laip11/PalmScore", 
    packages=find_packages(),  # 自动发现所有包
    install_requires=[  # 项目依赖的第三方库
        "numpy==1.26.4",
        "torch==2.5.1",
        "transformers==4.49.0",
        "argparse",
        "tqdm",
        "pandas",
        "datasets==3.1.0",
        "scipy",
        "prettytable",
    ],
    python_requires=">=3.10",  # 支持的 Python 版本
    classifiers=[
        "Development Status :: 3 - Alpha",  # 开发状态
        "Intended Audience :: Developers",  # 目标用户
        "Intended Audience :: Science/Research",  # 目标用户
        "License :: OSI Approved :: Apache Software License",  # 许可证
        "Operating System :: OS Independent",  # 操作系统兼容性
        "Programming Language :: Python :: 3",  # 编程语言
        "Programming Language :: Python :: 3.10",  # 编程语言版本
        "Topic :: Scientific/Engineering :: Artificial Intelligence",  # 主题分类
    ],
)