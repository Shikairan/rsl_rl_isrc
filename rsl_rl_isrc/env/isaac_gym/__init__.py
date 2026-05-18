# Isaac Gym + G1（rsl_rl_isrc 内置 legged 实现，延迟导入避免未安装 isaacgym 时失败）。

__all__ = [
    "IsaacG1VecEnv",
    "make_g1_isaac_env",
    "build_g1_ppo_train_cfg",
    "G1OnPolicyTestRunner",
    "has_isaac_gym",
]


def __getattr__(name: str):
    if name == "IsaacG1VecEnv":
        from rsl_rl_isrc.env.isaac_gym.isaac_g1_vec_env import IsaacG1VecEnv
        return IsaacG1VecEnv
    if name in (
        "make_g1_isaac_env",
        "build_g1_ppo_train_cfg",
        "normalize_train_cfg",
        "default_g1_log_dir",
        "has_isaac_gym",
    ):
        from rsl_rl_isrc.env.isaac_gym import make_g1_isaac as _m
        return getattr(_m, name)
    if name == "G1OnPolicyTestRunner":
        from rsl_rl_isrc.env.isaac_gym.test_runner import G1OnPolicyTestRunner
        return G1OnPolicyTestRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
