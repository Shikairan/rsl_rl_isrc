# 本文件与 rsl_rl_isrc 内其它业务子包完全独立，无交叉导入。
# 用途：为用户自定义物理仿真环境提供完整的配置数据类体系。
# 参考：legged_gym / Isaac Lab 配置风格。
"""物理仿真环境配置数据类。

层次结构::

    BaseEnvCfg
    ├── SimParamsCfg        ← 物理引擎参数（时间步、重力、求解器等）
    ├── ViewerCfg           ← 可视化窗口参数
    ├── TerrainCfg          ← 地形类型与网格尺寸
    ├── AssetCfg            ← 机器人 URDF/MJCF 路径与物理属性
    ├── InitStateCfg        ← 初始根位姿与关节默认值
    ├── ControlCfg          ← 控制模式、PD 增益、动作缩放
    ├── CommandsCfg         ← 指令范围（速度、方向等）
    ├── DomainRandCfg       ← 域随机化开关与参数范围
    ├── NoiseCfg            ← 观测/动作噪声
    ├── ObsCfg              ← 观测维度与特权观测配置
    └── RewardCfg           ← 奖励权重字典

使用方式::

    from rsl_rl_isrc.env.base_env_cfg import BaseEnvCfg

    class MyCfg(BaseEnvCfg):
        class sim(BaseEnvCfg.SimParamsCfg):
            dt = 0.005
        class asset(BaseEnvCfg.AssetCfg):
            file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/anymal_c/urdf/anymal_c.urdf"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 物理引擎仿真参数
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SimParamsCfg:
    """物理引擎核心参数。"""

    dt: float = 0.005
    """仿真时间步（秒）。控制策略的决策频率 = 1 / (dt * decimation)。"""

    substeps: int = 1
    """每个仿真步内的子步数（提高接触解算精度）。"""

    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    """重力加速度向量 (x, y, z)，单位 m/s²。"""

    up_axis: str = "z"
    """世界坐标系的向上轴，``'z'`` 或 ``'y'``。"""

    use_gpu_pipeline: bool = True
    """是否使用 GPU 管线（需要支持 CUDA 的物理引擎，如 Isaac Gym）。"""

    physx_num_threads: int = 10
    """PhysX CPU 解算线程数（仅在 CPU 管线时有效）。"""

    physx_solver_type: int = 1
    """PhysX 求解器类型：0 = PGS，1 = TGS。"""

    physx_num_position_iterations: int = 4
    """位置迭代次数，越大越精确但越慢。"""

    physx_num_velocity_iterations: int = 0
    """速度迭代次数。"""

    physx_contact_offset: float = 0.01
    """接触偏移量（米）：生成接触对的距离阈值。"""

    physx_rest_offset: float = 0.0
    """静止偏移量（米）：接触恢复后物体的间隙。"""

    physx_bounce_threshold_velocity: float = 0.5
    """弹跳阈值速度（m/s）：低于此速度的碰撞不产生弹跳。"""

    physx_max_depenetration_velocity: float = 1.0
    """最大去穿透速度（m/s）。"""


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ViewerCfg:
    """仿真可视化窗口配置（仅调试时有效，训练通常关闭）。"""

    ref_env: int = 0
    """摄像机跟随的参考环境索引。"""

    pos: Tuple[float, float, float] = (10.0, 0.0, 6.0)
    """初始相机位置 (x, y, z)，单位米。"""

    lookat: Tuple[float, float, float] = (11.0, 5.0, 3.0)
    """初始相机注视点 (x, y, z)，单位米。"""

    headless: bool = False
    """无头模式（True = 不显示窗口，适合服务器训练）。"""


# ─────────────────────────────────────────────────────────────────────────────
# 地形
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TerrainCfg:
    """地形生成与加载参数。"""

    mesh_type: str = "plane"
    """
    地形网格类型：
    - ``'plane'``：平坦地面（最简单，调试用）
    - ``'heightfield'``：高度场地形（使用程序生成的随机地形）
    - ``'trimesh'``：三角网格地形（最逼真，从 .obj/.stl 加载或程序生成）
    """

    static_friction: float = 1.0
    """地面静摩擦系数。"""

    dynamic_friction: float = 1.0
    """地面动摩擦系数。"""

    restitution: float = 0.0
    """地面弹性系数（0 = 完全非弹性）。"""

    # 仅 heightfield / trimesh 模式有效
    num_rows: int = 10
    """地形行数（沿 x 轴方向的地块数量）。"""

    num_cols: int = 10
    """地形列数（沿 y 轴方向的地块数量）。"""

    terrain_width: float = 12.0
    """单块地形宽度（米）。"""

    terrain_length: float = 12.0
    """单块地形长度（米）。"""

    horizontal_scale: float = 0.1
    """高度场水平采样间隔（米/像素）。"""

    vertical_scale: float = 0.005
    """高度场垂直精度（米/步）。"""

    slope_treshold: float = 0.75
    """将斜面三角形转为垂直面的坡度阈值（用于悬崖生成）。"""

    curriculum: bool = False
    """是否启用课程学习地形（根据成功率动态调整难度）。"""

    difficulty_scale: float = 1.0
    """课程难度整体缩放系数（0=最简单，1=最难）。"""

    terrain_types: List[str] = field(default_factory=lambda: [
        "flat", "rough", "slope", "stairs_up", "stairs_down",
        "discrete", "stepping_stones", "gap",
    ])
    """启用的地形种类列表（仅 heightfield/trimesh 模式有效）。"""

    terrain_proportions: List[float] = field(default_factory=lambda: [
        0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1,
    ])
    """各地形种类的比例，长度须与 ``terrain_types`` 一致，总和为 1。"""

    max_init_terrain_level: int = 5
    """课程初始化时的最大地形难度等级（0-based）。"""

    border_size: float = 25.0
    """围绕所有地块的边界宽度（米），防止机器人走出地形范围。"""


# ─────────────────────────────────────────────────────────────────────────────
# 机器人资产
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AssetCfg:
    """机器人 URDF/MJCF 文件路径与物理属性配置。"""

    file: str = ""
    """机器人描述文件路径（支持 {ASSET_ROOT} 等环境变量占位符）。必填。"""

    name: str = "robot"
    """在仿真场景中的 actor 名称。"""

    foot_name: str = "foot"
    """脚部刚体名称的子串（用于脚部接触检测）。"""

    penalize_contacts_on: List[str] = field(default_factory=list)
    """碰撞惩罚部件名称列表（子串匹配），如 ``["thigh", "calf"]``。"""

    terminate_after_contacts_on: List[str] = field(default_factory=list)
    """碰撞即终止的部件名称列表（子串匹配），如 ``["base"]``。"""

    disable_gravity: bool = False
    """是否对机器人关闭重力（用于悬浮调试）。"""

    collapse_fixed_joints: bool = True
    """是否合并固定关节（减少自由度，提高仿真速度）。"""

    fix_base_link: bool = False
    """是否将根连杆固定在世界坐标系（用于纯手臂或桌面场景）。"""

    default_dof_drive_mode: int = 3
    """关节驱动模式：1=无力矩，2=位置，3=速度，4=力矩（枚举值依赖仿真引擎）。"""

    self_collisions: int = 0
    """自碰撞过滤：0=关闭，1=开启。"""

    replace_cylinder_with_capsule: bool = True
    """将圆柱碰撞体替换为胶囊体（提高碰撞检测效率）。"""

    flip_visual_attachments: bool = True
    """是否翻转视觉网格附件（修正 URDF y-up 到 z-up 问题）。"""

    density: float = 0.001
    """刚体密度（kg/m³），仅在 URDF 未指定惯性时生效。"""

    angular_damping: float = 0.0
    """根刚体角速度阻尼（Nms/rad）。"""

    linear_damping: float = 0.0
    """根刚体线速度阻尼（Ns/m）。"""

    max_angular_velocity: float = 1000.0
    """最大允许角速度（rad/s），超过会被 clip。"""

    max_linear_velocity: float = 1000.0
    """最大允许线速度（m/s），超过会被 clip。"""

    armature: float = 0.0
    """关节电枢惯量附加值（kg·m²），用于稳定仿真。"""

    thickness: float = 0.01
    """碰撞厚度偏移（米），防止穿透。"""


# ─────────────────────────────────────────────────────────────────────────────
# 初始状态
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class InitStateCfg:
    """机器人出生点姿态与关节初始值配置。"""

    # 根连杆
    pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    """根连杆初始位置 (x, y, z)，单位米。z 轴通常需要高于地面以防初始穿透。"""

    rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """根连杆初始四元数姿态 (qx, qy, qz, qw)。"""

    lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """根连杆初始线速度 (vx, vy, vz)，单位 m/s。"""

    ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """根连杆初始角速度 (wx, wy, wz)，单位 rad/s。"""

    # 关节
    default_joint_angles: Dict[str, float] = field(default_factory=dict)
    """
    关节名称到初始角度（弧度）的映射。
    示例（ANYmal）::

        default_joint_angles = {
            "LF_HAA": 0.0, "LH_HAA": 0.0,
            "RF_HAA": -0.0, "RH_HAA": -0.0,
            "LF_HFE": 0.4, "LH_HFE": -0.4,
            "RF_HFE": 0.4, "RH_HFE": -0.4,
            "LF_KFE": -0.8, "LH_KFE": 0.8,
            "RF_KFE": -0.8, "RH_KFE": 0.8,
        }
    """


# ─────────────────────────────────────────────────────────────────────────────
# 控制
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ControlCfg:
    """关节控制器配置（PD 增益、动作缩放等）。"""

    control_type: str = "P"
    """
    控制类型：
    - ``'P'``：位置控制（PD 控制器）
    - ``'V'``：速度控制
    - ``'T'``：力矩直接控制
    """

    stiffness: Dict[str, float] = field(default_factory=dict)
    """
    位置增益（Nm/rad），按关节名称子串匹配。
    示例::

        stiffness = {"HAA": 80.0, "HFE": 80.0, "KFE": 80.0}
    """

    damping: Dict[str, float] = field(default_factory=dict)
    """
    速度阻尼（Nms/rad），按关节名称子串匹配。
    示例::

        damping = {"HAA": 2.0, "HFE": 2.0, "KFE": 2.0}
    """

    action_scale: float = 0.25
    """策略输出动作的缩放系数（相对于默认关节角度的偏移量缩放）。"""

    hip_scale_reduction: float = 0.5
    """髋关节动作缩放系数（相对于 action_scale 的额外缩减，防止过大转矩）。"""

    decimation: int = 4
    """策略决策频率缩减比：物理仿真步数与策略步数之比。
    策略频率 = 1 / (sim.dt * decimation)。"""


# ─────────────────────────────────────────────────────────────────────────────
# 指令
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CommandsCfg:
    """指令（速度/朝向）采样范围配置。"""

    curriculum: bool = False
    """是否根据跟踪误差动态调整指令难度（课程学习）。"""

    max_curriculum: float = 1.0
    """课程最大难度（用于限制采样范围）。"""

    num_commands: int = 4
    """指令维度：通常为 4（vx, vy, yaw_rate, heading），或 3（vx, vy, yaw_rate）。"""

    resampling_time: float = 10.0
    """指令重采样间隔（秒）。"""

    heading_command: bool = True
    """
    是否使用朝向指令模式（True）还是偏航速率模式（False）。
    - True：第 4 维为目标朝向角（绝对），控制器内部转换为偏航速率
    - False：第 4 维为偏航角速率（rad/s）
    """

    # 各指令的范围 [min, max]
    lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
    """线速度 x 方向范围 (m/s)。"""

    lin_vel_y: Tuple[float, float] = (-1.0, 1.0)
    """线速度 y 方向范围 (m/s)。"""

    ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)
    """偏航角速率范围 (rad/s)。"""

    heading: Tuple[float, float] = (-3.14159, 3.14159)
    """目标朝向范围（弧度，仅 heading_command=True 时使用）。"""


# ─────────────────────────────────────────────────────────────────────────────
# 域随机化
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DomainRandCfg:
    """域随机化开关与参数范围（提升策略鲁棒性）。"""

    randomize_friction: bool = True
    """是否随机化地面/脚部摩擦系数。"""

    friction_range: Tuple[float, float] = (0.5, 1.25)
    """摩擦系数随机范围 [min, max]。"""

    randomize_base_mass: bool = False
    """是否对根连杆附加随机质量。"""

    added_mass_range: Tuple[float, float] = (-1.0, 1.0)
    """附加质量范围（kg）[min, max]。"""

    push_robots: bool = True
    """是否定期向机器人施加随机推力（干扰鲁棒性测试）。"""

    push_interval_s: float = 15.0
    """推力施加间隔（秒）。"""

    max_push_vel_xy: float = 1.0
    """推力引起的最大水平速度变化（m/s）。"""


# ─────────────────────────────────────────────────────────────────────────────
# 观测噪声
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NoiseCfg:
    """向观测与动作添加噪声的参数（模拟真实传感器误差）。"""

    add_noise: bool = True
    """是否在观测中添加高斯噪声。"""

    noise_level: float = 1.0
    """噪声整体缩放系数（训练时可逐步增大）。"""

    # 各传感器噪声标准差
    lin_vel: float = 0.1
    """线速度噪声标准差（m/s）。"""

    ang_vel: float = 0.2
    """角速度噪声标准差（rad/s）。"""

    gravity: float = 0.05
    """重力方向噪声标准差。"""

    dof_pos: float = 0.01
    """关节位置噪声标准差（rad）。"""

    dof_vel: float = 1.5
    """关节速度噪声标准差（rad/s）。"""

    height_measurements: float = 0.1
    """高度扫描噪声标准差（米）。"""


# ─────────────────────────────────────────────────────────────────────────────
# 观测空间
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ObsCfg:
    """观测空间维度与特权观测配置。"""

    num_obs: int = 48
    """策略观测维度（Actor 输入）。"""

    num_privileged_obs: Optional[int] = None
    """
    Critic（价值函数）特权观测维度。
    - ``None`` 表示 Critic 使用与 Actor 相同的观测
    - 设为整数时，Critic 可获得额外信息（如地形高度、摩擦系数等）
    """

    send_timeouts: bool = True
    """是否在 infos 中传递超时标志（用于正确处理 bootstrap 边界值）。"""


# ─────────────────────────────────────────────────────────────────────────────
# 奖励
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RewardCfg:
    """奖励函数权重配置。

    子类示例（legged_gym 风格）::

        @dataclass
        class rewards(BaseEnvCfg.RewardCfg):
            class scales:
                termination         = -0.0
                tracking_lin_vel    =  1.0
                tracking_ang_vel    =  0.5
                lin_vel_z           = -2.0
                ang_vel_xy          = -0.05
                orientation         = -0.0
                torques             = -0.00001
                dof_vel             = -0.0
                dof_acc             = -2.5e-7
                base_height         = -0.0
                feet_air_time       =  1.0
                collision           = -1.0
                feet_stumble        = -0.0
                action_rate         = -0.01
                stand_still         = -0.0
    """

    only_positive_rewards: bool = True
    """是否将总奖励 clip 到非负（防止策略学习一味求死）。"""

    tracking_sigma: float = 0.25
    """速度跟踪奖励的高斯核宽度 σ（越小要求越精确）。"""

    soft_dof_pos_limit: float = 1.0
    """关节位置软限制系数（相对于 URDF 硬限制的比例，超出部分产生惩罚）。"""

    soft_dof_vel_limit: float = 1.0
    """关节速度软限制系数。"""

    soft_torque_limit: float = 1.0
    """力矩软限制系数。"""

    base_height_target: float = 1.0
    """目标底盘高度（米），用于高度跟踪奖励。"""

    max_contact_force: float = 100.0
    """最大接触力阈值（N），超出产生惩罚。"""

    class scales:
        """奖励缩放系数字典（用内嵌类模拟 namespace）。
        子类中可直接覆盖各字段。
        """
        termination: float = -0.0
        tracking_lin_vel: float = 1.0
        tracking_ang_vel: float = 0.5
        lin_vel_z: float = -2.0
        ang_vel_xy: float = -0.05
        orientation: float = -0.0
        torques: float = -0.00001
        dof_vel: float = -0.0
        dof_acc: float = -2.5e-7
        base_height: float = -0.0
        feet_air_time: float = 1.0
        collision: float = -1.0
        feet_stumble: float = -0.0
        action_rate: float = -0.01
        stand_still: float = -0.0


# ─────────────────────────────────────────────────────────────────────────────
# 顶层配置类
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BaseEnvCfg:
    """物理仿真环境完整配置，汇总所有子配置。

    使用方式（推荐用内嵌类覆盖子配置）::

        class AnymalCCfg(BaseEnvCfg):
            # 直接覆盖子配置类
            class sim(BaseEnvCfg.SimParamsCfg):
                dt = 0.005

            class terrain(BaseEnvCfg.TerrainCfg):
                mesh_type = "trimesh"
                curriculum = True

            class asset(BaseEnvCfg.AssetCfg):
                file = "{ANYMAL_ROOT}/urdf/anymal_c.urdf"
                foot_name = "FOOT"

            class init_state(BaseEnvCfg.InitStateCfg):
                pos = (0.0, 0.0, 0.6)
                default_joint_angles = {
                    "LF_HAA": 0.0, "LH_HAA": 0.0, ...
                }

            class control(BaseEnvCfg.ControlCfg):
                stiffness = {"HAA": 80., "HFE": 80., "KFE": 80.}
                damping    = {"HAA": 2.,  "HFE": 2.,  "KFE": 2.}

            class rewards(BaseEnvCfg.RewardCfg):
                class scales:
                    tracking_lin_vel = 1.5
                    torques = -1e-5
    """

    # ── 子配置类（作为类型注解，支持内嵌类覆盖） ──────────────────────────────
    SimParamsCfg = SimParamsCfg
    ViewerCfg = ViewerCfg
    TerrainCfg = TerrainCfg
    AssetCfg = AssetCfg
    InitStateCfg = InitStateCfg
    ControlCfg = ControlCfg
    CommandsCfg = CommandsCfg
    DomainRandCfg = DomainRandCfg
    NoiseCfg = NoiseCfg
    ObsCfg = ObsCfg
    RewardCfg = RewardCfg

    # ── 顶层字段（可直接在实例上访问）─────────────────────────────────────────
    num_envs: int = 4096
    """并行环境总数。"""

    env_spacing: float = 3.0
    """环境之间的网格间距（米），用于 Isaac Gym 的场景布局。"""

    episode_length_s: float = 20.0
    """单回合最大时长（秒）。超过则 truncation。"""

    send_timeouts: bool = True
    """是否在 infos 中传递 time_outs 张量（用于 PPO bootstrap 边界修正）。"""

    seed: int = 0
    """随机种子（Python / NumPy / PyTorch）。"""
