#!/usr/bin/env python3
"""仅加载仿真、对比 URDF vs XML 在 Isaac 中的 DOF/限位（不训练 PPO）。

运行::

    python rsl_rl_isrc/tests/g1_asset_ab_inspect.py
    python rsl_rl_isrc/tests/g1_asset_ab_inspect.py --num-envs 4
"""

from __future__ import annotations

import argparse
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from g1_asset_ab_common import (  # noqa: E402
    ensure_project_on_path,
    check_isaac_cuda,
    format_robot_summary,
    get_g1_asset_paths,
    make_g1_env_with_asset,
    resolve_asset_file,
)


def _parse_hip_roll_limits_from_file(path: str) -> dict[str, str]:
    """从 urdf/xml 文本粗读 hip_roll 限位（静态，不依赖 Isaac）。"""
    if not os.path.isfile(path):
        return {}
    text = open(path, encoding="utf-8", errors="replace").read()
    out: dict[str, str] = {}
    if path.endswith(".urdf"):
        for side in ("left", "right"):
            m = re.search(
                rf'<joint name="{side}_hip_roll_joint"[^>]*>.*?<limit lower="([^"]+)" upper="([^"]+)"',
                text,
                re.DOTALL,
            )
            if m:
                out[f"{side}_hip_roll"] = f"[{m.group(1)}, {m.group(2)}]"
    elif path.endswith(".xml"):
        for line in text.splitlines():
            for side in ("left", "right"):
                key = f'name="{side}_hip_roll_joint"'
                if key in line and "range=" in line:
                    m = re.search(r'range="([^"]+)"', line)
                    if m:
                        out[f"{side}_hip_roll"] = m.group(1)
    return out


def main() -> None:
    ensure_project_on_path()
    # 子进程仅做 Isaac 加载时跳过重复的静态段
    if os.environ.get("G1_AB_INSPECT_CHILD") == "1":
        import argparse as _ap

        p = _ap.ArgumentParser()
        p.add_argument("--asset", default="urdf")
        p.add_argument("--num-envs", type=int, default=4)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--headless", action="store_true", default=True)
        a, _ = p.parse_known_args()
        check_isaac_cuda()
        asset_path = resolve_asset_file(a.asset)
        print(f"\n--- asset_kind={a.asset} ---")
        env, _, _, robot = make_g1_env_with_asset(
            a.asset,
            num_envs=a.num_envs,
            headless=a.headless,
            sim_device=a.device,
            seed=1,
        )
        print(format_robot_summary(robot, asset_path))
        return

    parser = argparse.ArgumentParser(description="G1 URDF/XML Isaac 加载对比（无 PPO）")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--asset",
        type=str,
        default="all",
        choices=("all", "urdf", "xml"),
        help="Isaac 加载项；all 时对 urdf/xml 各启子进程（避免同进程二次 init 崩溃）",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="只打印路径与文件内限位，不启动 Isaac",
    )
    args = parser.parse_args()

    print("=== 资产路径 ===")
    paths = get_g1_asset_paths(setup_meshes=True)
    for k, v in paths.items():
        exists = os.path.isfile(v)
        print(f"  {k}: {v}  ({'OK' if exists else 'MISSING'})")

    print("\n=== 静态 hip_roll 限位（文件内文本）===")
    for kind in ("urdf", "xml"):
        p = resolve_asset_file(kind)
        limits = _parse_hip_roll_limits_from_file(p)
        print(f"  [{kind}] {os.path.basename(p)}")
        for joint, lim in limits.items():
            print(f"    {joint}: {lim}")

    if args.static_only:
        return

    kinds = ("urdf", "xml") if args.asset == "all" else (args.asset,)

    def _isaac_spawn_one(kind: str) -> None:
        check_isaac_cuda()
        asset_path = resolve_asset_file(kind)
        print(f"\n--- asset_kind={kind} ---")
        env, _, _, robot = make_g1_env_with_asset(
            kind,
            num_envs=args.num_envs,
            headless=args.headless,
            sim_device=args.device,
            seed=1,
        )
        print(format_robot_summary(robot, asset_path))

    if len(kinds) == 1:
        _isaac_spawn_one(kinds[0])
    else:
        import subprocess

        script = os.path.abspath(__file__)
        env_child = {**os.environ, "G1_AB_INSPECT_CHILD": "1"}
        for kind in kinds:
            print(f"\n>>> 子进程加载 {kind}")
            cmd = [
                sys.executable,
                script,
                "--asset",
                kind,
                "--num-envs",
                str(args.num_envs),
                "--device",
                args.device,
            ]
            if args.headless:
                cmd.append("--headless")
            subprocess.run(
                cmd,
                check=True,
                cwd=_PROJECT_ROOT_FROM_SCRIPT(),
                env=env_child,
            )

    print(
        "\n说明: 若 num_dof / dof_names 一致但训练曲线差，"
        "差异多半在碰撞体、质量分布或 MJCF/URDF 解析方式。"
    )


def _PROJECT_ROOT_FROM_SCRIPT() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    main()
