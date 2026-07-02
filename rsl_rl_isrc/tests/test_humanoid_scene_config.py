import os
import xml.etree.ElementTree as ET

from rsl_rl_isrc.env.isaac_gym.legged import ensure_h1_10dof_urdf
from rsl_rl_isrc.env.isaac_gym.make_humanoid_isaac import build_humanoid_cfg


def _movable_joint_names(path):
    root = ET.parse(path).getroot()
    names = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}:
            names.append(joint.attrib["name"])
    return names


def test_h1_10dof_asset_is_generated_with_fixed_ankle_roll_joints():
    path = ensure_h1_10dof_urdf()

    assert os.path.basename(path) == "h1_10dof.urdf"
    assert len(_movable_joint_names(path)) == 10

    joints = {
        joint.attrib["name"]: joint.attrib.get("type")
        for joint in ET.parse(path).getroot().findall("joint")
    }
    assert joints["left_ankle_roll_joint"] == "fixed"
    assert joints["right_ankle_roll_joint"] == "fixed"


def test_build_g1_slope_cfg_sets_scene_and_deterministic_reset():
    cfg, train_cfg = build_humanoid_cfg(
        robot="g1_12dof",
        scene="slope",
        num_envs=32,
        max_iterations=11,
    )

    assert cfg.env.num_envs == 32
    assert cfg.env.deterministic_reset is True
    assert cfg.terrain.scene_type == "slope"
    assert cfg.terrain.ramp_angle_deg > 0.0
    assert train_cfg["runner"]["experiment_name"] == "g1_slope"
    assert train_cfg["runner"]["max_iterations"] == 11


def test_build_h1_collision_cfg_uses_10dof_asset_and_collision_reward():
    cfg, train_cfg = build_humanoid_cfg(
        robot="h1_10dof",
        scene="collision",
        num_envs=16,
        max_iterations=7,
    )

    assert cfg.env.num_actions == 10
    assert cfg.asset.file == "{H1_DESCRIPTION_DIR}/h1_10dof.urdf"
    assert cfg.domain_rand.push_robots is True
    assert cfg.domain_rand.push_interval_s <= 2.0
    assert cfg.rewards.scales.collision < 0.0
    assert train_cfg["runner"]["experiment_name"] == "h1_collision"
    assert train_cfg["runner"]["max_iterations"] == 7
