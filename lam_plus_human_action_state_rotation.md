# lam_plus_human: action/state 旋转表示清单（仅基于数据集本身 + info.json）

约束：
- 仅使用 `lam_plus_human` 中各数据集自身的 `meta/info.json`（`features` 的 key/shape/names）做判断。
- 不引用 `data_config.py`、`modality.json`、训练侧映射逻辑。

判定规则（基于 `info.json`）：
- 轴名出现 `roll/pitch/yaw` 或 `droll/dpitch/dyaw` -> `Euler RPY`
- 轴名出现 `rx/ry/rz/rw`（4维）-> `Quaternion`
- 字段名出现 `rotation_6d` -> `rotation_6d`
- 仅有抽象维度名（如 `action_0..5`）-> `未知`

## State（按混合集逐项）

| dataset_name | info.json 中 state 相关特征 | 推断的旋转表示 | 判断依据（仅 info.json） |
|---|---|---|---|
| `libero_all` | `observation.state` / `observation.states.ee_state` | axis-angle(rotvec) | 轴名是 `axis_angle1/2/3` |
| `BridgeV2` | `observation.state`（joint+gripper） | 无明确姿态字段 | state 只有关节/夹爪轴名 |
| `fractal_lerobot` | `observation.state` | Quaternion | 轴名含 `rx,ry,rz,rw` |
| `droid_1.0.1` | `observation.state.cartesian_position` | Euler RPY | 轴名含 `roll,pitch,yaw` |
| `AgiBot_merge_delta_action` | `observation.states.end.orientation` | Quaternion | 8维，左右各 `x,y,z,w` |
| `robomind_agilex_3rgb` | `observation.states.end_effector_left/right` | Quaternion | 7维中姿态为 `rx,ry,rz,rw` |
| `robomind_franka_1rgb` | `observation.states.end_effector` | Euler RPY | 轴名 `x,y,z,r,p,y`（rpy） |
| `robomind_franka_3rgb` | `observation.states.end_effector` | Euler RPY | 轴名 `x,y,z,r,p,y`（rpy） |
| `robomind_ur_1rgb` | `observation.states.end_effector` | Euler RPY | 轴名 `x,y,z,r,p,y`（rpy） |
| `robomind_franka_fr3_dual` | `observation.states.end_effector` | Euler RPY | names 为 `left_xyzrpy/right_xyzrpy` |
| `epic_kitchens_100_lerobot` | `state.xyz_rotation_6d_gripper` | rotation_6d（打包） | 字段名直接包含 `rotation_6d` |

## Action（按混合集逐项）

| dataset_name | info.json 中 action 相关特征 | 推断的旋转表示 | 判断依据（仅 info.json） |
|---|---|---|---|
| `libero_all` | `action` | axis-angle(rotvec) | 轴名是 `axis_angle1/2/3` |
| `BridgeV2` | `action` | 未知 | 轴名仅 `action_0..5`，无法从 info.json 唯一判定 |
| `fractal_lerobot` | `action` | Euler RPY | 轴名含 `roll,pitch,yaw` |
| `droid_1.0.1` | `action.cartesian_velocity` | Euler RPY | 轴名含 `roll,pitch,yaw` |
| `AgiBot_merge_delta_action` | `actions.delta` | axis-angle(rotvec)（高置信） | 轴名为 `delta_end_ori_*_rx/ry/rz` |
| `robomind_agilex_3rgb` | `actions.delta_eef_left/right` | axis-angle(rotvec)（高置信） | 轴名为 `dx,dy,dz,rx,ry,rz,dg` |
| `robomind_franka_1rgb` | `actions.eef_rot` | Euler delta（RPY） | 轴名为 `droll,dpitch,dyaw` |
| `robomind_franka_3rgb` | `actions.eef_rot` | Euler delta（RPY） | 轴名为 `droll,dpitch,dyaw` |
| `robomind_ur_1rgb` | `actions.joint_position` | 无明确姿态 action 字段 | action 只有关节+夹爪轴名 |
| `robomind_franka_fr3_dual` | `actions.joint_position` | 无明确姿态 action 字段 | action 只有双臂关节+夹爪轴名 |
| `epic_kitchens_100_lerobot` | `action.xyz_rotation_6d_gripper` | rotation_6d（打包） | 字段名直接包含 `rotation_6d` |

