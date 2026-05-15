"""
rsl_rl_isrc — 基于 rsl_rl 思路的 PyTorch 强化学习组件（PPO、TRPO、REINFORCE、SAC、
RolloutStorage/ReplayBuffer、sockets HTTP 上报等）。

致谢：rsl_rl 原团队；本仓库由 ISRC 在独立包名 ``rsl_rl_isrc`` 下维护与扩展。

开发安装：在项目根目录执行 ``pip install -e .``；运行单元测试需 ``pip install -e ".[dev]"``。
"""
from setuptools import find_packages, setup

setup(
    name="rsl_rl_isrc",
    version="0.0.1",
    author="Nikita Rudin / ISRC",
    license="BSD-3-Clause",
    description="PyTorch RL algorithms and training runners (PPO, TRPO, REINFORCE, SAC)",
    long_description=__doc__,
    long_description_content_type="text/plain",
    python_requires=">=3.8",
    packages=find_packages(include=("rsl_rl_isrc", "rsl_rl_isrc.*")),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.19.0",
        "requests>=2.25.0",
        "pyzmq>=22.0",
        "tensorboard>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "tqdm>=4.60",
            "gymnasium>=0.28.0",
        ],
    },
    keywords="reinforcement learning, pytorch, ppo, trpo, sac, rsl_rl",
)
