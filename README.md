# RSL RL
Fast and simple implementation of RL algorithms, designed to run fully on GPU.
This code is an evolution of `rl-pytorch` provided with NVIDIA's Isaac GYM.

Only PPO is implemented for now. More algorithms will be added later.
Contributions are welcome.

## 致谢与说明
- **致谢**: 感谢 rsl_rl 原团队（NVIDIA Isaac Lab / ETH Zurich Robotic Systems Lab）提供的优秀开发基础
- **项目说明**: 本项目基于 rsl_rl 框架进行开发与扩展
- **开发团队**: 额外开发工作由 ISRC 团队完成

## Setup

```
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
pip install -e .
```

或安装本项目 (rsl_rl_isrc)：
```
cd rsl_rl_withALL
pip install -e .
# 使用时: from rsl_rl_isrc.xxx import yyy
```

### Useful Links ###
Example use case: https://github.com/leggedrobotics/legged_gym  
Project website: https://leggedrobotics.github.io/legged_gym/  
Paper: https://arxiv.org/abs/2109.11978

**Maintainer**: Nikita Rudin  
**Affiliation**: Robotic Systems Lab, ETH Zurich & NVIDIA  
**Contact**: rudinn@ethz.ch  



