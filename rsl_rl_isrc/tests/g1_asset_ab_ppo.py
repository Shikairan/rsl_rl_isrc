#!/usr/bin/env python3
"""A/B：同一 PPO 配置下 URDF vs XML 短训，对比 TensorBoard mean_reward。

不修改 ``make_g1_isaac`` / ``test_ppo_g1_isaac``；使用标准 ``OnPolicyRunner``（无 ObsInstr、无 learn(1) 套娃）。

运行::

    python rsl_rl_isrc/tests/g1_asset_ab_ppo.py --num-envs 64 --max-iterations 200
    python rsl_rl_isrc/tests/g1_asset_ab_ppo.py --assets urdf xml --seed 1 --log-root /tmp/g1_ab

可选：仅验证 learn(1) 是否影响结果（固定 URDF）::

    python rsl_rl_isrc/tests/g1_asset_ab_ppo.py --runner-modes plain,test --assets urdf --max-iterations 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from g1_asset_ab_common import (  # noqa: E402
    ensure_project_on_path,
    check_isaac_cuda,
    compare_ab_results,
    make_g1_env_with_asset,
    read_tensorboard_scalars,
    resolve_asset_file,
)


def _run_one_ppo(
    *,
    asset_kind: str,
    runner_mode: str,
    num_envs: int,
    max_iterations: int,
    seed: int,
    device: str,
    headless: bool,
    log_dir: str,
) -> dict:
    from rsl_rl_isrc.runners.on_policy_runner import OnPolicyRunner

    env, cfg, train_cfg, robot = make_g1_env_with_asset(
        asset_kind,
        num_envs=num_envs,
        headless=headless,
        sim_device=device,
        seed=seed,
    )
    asset_path = resolve_asset_file(asset_kind)
    print(f"\n>>> 开始训练 asset={asset_kind} runner={runner_mode} log_dir={log_dir}")
    print(f"    {asset_path}")
    print(f"    num_dof={robot.num_dof} num_envs={num_envs} seed={seed}")

    train_cfg["runner"]["max_iterations"] = max_iterations
    os.makedirs(log_dir, exist_ok=True)

    if runner_mode == "plain":
        runner = OnPolicyRunner(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)
        runner.learn(max_iterations, init_at_random_ep_len=True)
    elif runner_mode == "test":
        from rsl_rl_isrc.env.isaac_gym.test_runner import G1OnPolicyTestRunner

        runner = G1OnPolicyTestRunner(
            env=env,
            train_cfg=train_cfg,
            log_dir=log_dir,
            device=device,
            enable_obs_server=False,
        )
        runner.learn(max_iterations, init_at_random_ep_len=True)
    else:
        raise ValueError(f"未知 runner_mode: {runner_mode}")

    scalars = read_tensorboard_scalars(log_dir)
    return {
        "asset_kind": asset_kind,
        "runner_mode": runner_mode,
        "log_dir": log_dir,
        "asset_path": asset_path,
        "num_dof": int(robot.num_dof),
        "scalars": scalars,
    }


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_child_from_env() -> bool:
    """子进程：环境变量 G1_AB_PPO_SPEC=json 指定单次训练。"""
    spec_json = os.environ.get("G1_AB_PPO_SPEC")
    if not spec_json:
        return False
    ensure_project_on_path()
    check_isaac_cuda()
    spec = json.loads(spec_json)
    results = {
        spec["key"]: _run_one_ppo(
            asset_kind=spec["asset_kind"],
            runner_mode=spec["runner_mode"],
            num_envs=spec["num_envs"],
            max_iterations=spec["max_iterations"],
            seed=spec["seed"],
            device=spec["device"],
            headless=spec["headless"],
            log_dir=spec["log_dir"],
        )
    }
    out_path = spec.get("result_json")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    k: {
                        **v,
                        "scalars": v["scalars"],
                    }
                    for k, v in results.items()
                },
                f,
                indent=2,
            )
    return True


def main() -> None:
    ensure_project_on_path()
    if _run_child_from_env():
        return

    parser = argparse.ArgumentParser(
        description="G1 URDF vs XML PPO A/B（独立脚本，不改现有训练入口）"
    )
    parser.add_argument(
        "--assets",
        type=str,
        default="urdf,xml",
        help="逗号分隔: urdf, xml, current（current=与 make_g1_isaac 相同的 xml）",
    )
    parser.add_argument(
        "--runner-modes",
        type=str,
        default="plain",
        help="plain=OnPolicyRunner.learn(N); test=G1OnPolicyTestRunner learn(1)xN",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--log-root",
        type=str,
        default=None,
        help="根目录；默认 /tmp/g1_asset_ab_<timestamp>",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="50,100,150,200",
        help="对比表中列出的 iter",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()

    check_isaac_cuda()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    runner_modes = [m.strip() for m in args.runner_modes.split(",") if m.strip()]
    checkpoint_iters = tuple(int(x) for x in args.checkpoints.split(",") if x.strip())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = args.log_root or os.path.join("/tmp", f"g1_asset_ab_{stamp}")

    import subprocess

    results: dict[str, dict] = {}
    script = os.path.abspath(__file__)
    for asset_kind in assets:
        for runner_mode in runner_modes:
            run_name = f"{asset_kind}_{runner_mode}_s{args.seed}"
            log_dir = os.path.join(log_root, run_name)
            key = run_name
            result_json = os.path.join(log_dir, "run_result.json")
            os.makedirs(log_dir, exist_ok=True)
            spec = {
                "key": key,
                "asset_kind": asset_kind,
                "runner_mode": runner_mode,
                "num_envs": args.num_envs,
                "max_iterations": args.max_iterations,
                "seed": args.seed,
                "device": args.device,
                "headless": args.headless,
                "log_dir": log_dir,
                "result_json": result_json,
            }
            env = {**os.environ, "G1_AB_PPO_SPEC": json.dumps(spec)}
            print(f"\n>>> 子进程训练 {key}")
            subprocess.run(
                [sys.executable, script],
                check=True,
                cwd=_project_root(),
                env=env,
            )
            with open(result_json, encoding="utf-8") as f:
                chunk = json.load(f)
            results.update(chunk)

    report = compare_ab_results(results, checkpoint_iters=checkpoint_iters)
    print(report)

    summary_path = os.path.join(log_root, "ab_summary.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            "asset_kind": v["asset_kind"],
            "runner_mode": v["runner_mode"],
            "log_dir": v["log_dir"],
            "asset_path": v["asset_path"],
            "num_dof": v["num_dof"],
            "Train/mean_reward": v["scalars"].get("Train/mean_reward", []),
            "Train/mean_episode_length": v["scalars"].get(
                "Train/mean_episode_length", []
            ),
        }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n已写入: {summary_path}")
    print(f"TensorBoard: tensorboard --logdir {log_root}")


if __name__ == "__main__":
    main()
