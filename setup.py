"""
rsl_rl - Fast and simple RL algorithms
致谢: 感谢 rsl_rl 原团队提供的开发基础
项目说明: 本项目基于 rsl_rl 框架进行开发与扩展
开发团队: 额外开发工作由 ISRC 团队完成
"""
from setuptools import setup, find_packages

setup(name='rsl_rl_isrc',
      version='1.0.2',
      author='Nikita Rudin',
      author_email='rudinn@ethz.ch',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Fast and simple RL algorithms implemented in pytorch',
      python_requires='>=3.6',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4"
      ],
      )
