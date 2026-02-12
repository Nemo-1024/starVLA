# 多数据集元数据合并机制文档

## 概述

本文档记录了 `LeRobotMixtureDataset` 在混合多个数据集训练时的元数据合并机制，包括本次改动的内容、键名要求、统计信息整合方式以及常见陷阱。

---

## 1. 本次改动记录 (2026-02-04)

### 1.1 问题背景

在混合训练多个数据集时，`merge_metadata()` 函数要求同一 `embodiment_tag` 下的所有数据集具有完全一致的 modality 配置。这导致以下错误：

```
AssertionError: Multiple modality configs for modality video: [
  '{"primary_view": {"resolution": [1280, 720], ...}}',
  '{"primary_view": {"resolution": [256, 256], ...}, "wrist_view": {...}}'
]
```

### 1.2 改动内容

#### 1.2.1 Video 模态：并集合并

**文件**: `starVLA/dataloader/gr00t_lerobot/datasets.py`

**改动位置**: `merge_metadata()` 方法

```python
if modality == "video":
    # Union all video configs - different datasets may have different views
    merged_video = {}
    for config_json in configs:
        config_dict = json.loads(config_json)
        for video_key, video_meta in config_dict.items():
            if modality_used_keys is not None and video_key not in modality_used_keys:
                continue
            if video_key not in merged_video:
                merged_video[video_key] = video_meta
    merged_metadata["modalities"][modality] = merged_video
```

**原因**: 
- 不同数据集自然有不同的相机视角（1个 vs 3个）
- 不同数据集的分辨率不同（1280x720 vs 256x256）
- Video 的 `resolution` 会被 `img_resize` 参数覆盖，`fps` 从未被使用

#### 1.2.2 State/Action 模态：严格一致性检查

```python
else:
    # For state/action, require strict consistency
    assert (
        len(configs) == 1
    ), f"Multiple modality configs for modality {modality}: {list(configs)}"
    merged_metadata["modalities"][modality] = json.loads(configs.pop())
```

**原因**: State 和 Action 的元数据（shape、absolute、rotation_type）直接影响归一化计算，必须保持一致。

#### 1.2.3 构建时过滤：只包含 `modality_configs` 中定义的键

**文件**: `starVLA/dataloader/gr00t_lerobot/datasets.py`

**改动位置**: `_get_metadata()` 方法

```python
# 如果 modality 不在 modality_configs 中，跳过整个 modality
if used_keys and modality not in used_keys:
    continue

# 如果 subkey 不在 modality_configs 的 modality_keys 中，跳过该 subkey
if modality in used_keys and subkey not in used_keys[modality]:
    continue
```

**原因**: 
- 数据集的 `modality.json` 可能包含许多键（如 `joint_position`、`eef_gripper` 等）
- 但训练只使用 `data_config.py` 中定义的键

#### 1.2.4 修复统计信息合并时的权重匹配问题

**文件**: `starVLA/dataloader/gr00t_lerobot/datasets.py`

**改动位置**: `update_metadata()` 方法

**原问题**:
```python
# 旧代码 - BUG
for tag, metadatas in all_metadatas.items():
    self.merged_metadata[tag] = self.merge_metadata(
        metadatas=metadatas,  # 只有当前 tag 的数据集
        dataset_sampling_weights=self.dataset_sampling_weights.tolist(),  # 所有数据集的权重！
        ...
    )
```

当数据集顺序不是按 tag 连续排列时，权重会错误匹配。

**修复后**:
```python
# 新代码 - 修复
all_metadatas: dict[str, list[DatasetMetadata]] = {}
all_weights_by_tag: dict[str, list[float]] = {}
for idx, dataset in enumerate(self.datasets):
    if dataset.tag not in all_metadatas:
        all_metadatas[dataset.tag] = []
        all_weights_by_tag[dataset.tag] = []
    all_metadatas[dataset.tag].append(dataset.metadata)
    all_weights_by_tag[dataset.tag].append(self.dataset_sampling_weights[idx])

for tag, metadatas in all_metadatas.items():
    tag_weights = all_weights_by_tag[tag]  # 只使用当前 tag 的权重
    self.merged_metadata[tag] = self.merge_metadata(
        metadatas=metadatas,
        dataset_sampling_weights=tag_weights,
        ...
    )
```

**原因**: 确保统计信息合并时使用的权重与数据集采样概率一致
- 只构建实际使用的键的元数据，避免不同数据集因为"多余键"导致的不一致

---

## 2. 键名要求

### 2.1 `modality.json` 的键名要求

每个数据集的 `meta/modality.json` 定义了该数据集的所有可用键。

```json
{
  "state": {
    "eef_position": {"start": 0, "end": 3, "dtype": "float64", "absolute": true, ...},
    "eef_orientation": {"start": 3, "end": 6, ...},
    "joint_position": {"start": 6, "end": 13, ...},  // 可能存在但不被使用
    "gripper": {"start": 13, "end": 14, ...}
  },
  "action": {
    "eef_position": {...},
    "joint_position": {...},  // 可能存在但不被使用
    ...
  },
  "video": {
    "primary_view": {"original_key": "observation.images.camera_top"},
    "wrist_view": {"original_key": "observation.images.wrist"}
  }
}
```

**关键点**:
- `modality.json` 可以包含任意多的键
- 只有在 `data_config.py` 中定义的键会被使用
- 不在 `data_config.py` 中的键（如 `joint_position`）会被过滤掉

### 2.2 `data_config.py` 的键名要求

`data_config.py` 定义了每种机器人类型实际使用的键：

```python
class LiberoDataConfig:
    video_keys = ["video.primary_view"]
    state_keys = [
        "state.eef_position",
        "state.eef_orientation", 
        "state.gripper",
    ]
    action_keys = [
        "action.eef_position",
        "action.eef_orientation",
        "action.gripper",
    ]
```

**要求**:

| 模态 | 同一 embodiment_tag 下的要求 |
|------|------------------------------|
| **video** | 键名**无需一致**，采用并集合并 |
| **state** | 键名**必须一致**（过滤后） |
| **action** | 键名**必须一致**（过滤后） |

**示例**: 如果 FRANKA tag 下有两个数据集：

```python
# 数据集 A (libero): 正确
state_keys = ["state.eef_position", "state.gripper"]

# 数据集 B (robomind): 必须相同
state_keys = ["state.eef_position", "state.gripper"]  # ✓ 匹配
state_keys = ["state.eef_position", "state.eef_orientation", "state.gripper"]  # ✗ 不匹配
```

### 2.3 键名映射关系

```
data_config.py 定义          modality.json 中的键        实际数据列
─────────────────────────────────────────────────────────────────────
"state.eef_position"    →    state.eef_position     →    observation.state[0:3]
"video.primary_view"    →    video.primary_view     →    observation.images.camera_top
"action.gripper"        →    action.gripper         →    action[6:7]
```

---

## 3. 统计信息整合机制

### 3.1 统计信息的计算

每个 `LeRobotSingleDataset` 在初始化时计算自己的统计信息：

```python
# 文件: meta/stats_gr00t.json
{
  "observation.state": {
    "mean": [...], "std": [...], 
    "min": [...], "max": [...],
    "q01": [...], "q99": [...]
  },
  "action": {...}
}
```

### 3.2 统计信息的合并

`LeRobotMixtureDataset.compute_overall_statistics()` 合并多个数据集的统计信息：

```python
# 加权平均计算 overall mean
overall_mean = sum(weight * mean for weight, mean in zip(weights, means))

# 加权方差计算 overall std  
overall_variance = sum(weight * (std^2 + (mean - overall_mean)^2) for ...)
overall_std = sqrt(overall_variance)

# 分位数合并 (两种方法)
if percentile_mixing_method == "weighted_average":
    weighted_q01 = average(q01_list, weights=weights)
    weighted_q99 = average(q99_list, weights=weights)
elif percentile_mixing_method == "min_max":
    weighted_q01 = min(q01_list)
    weighted_q99 = max(q99_list)
```

### 3.3 统计信息的使用

合并后的统计信息用于 `StateActionTransform` 的归一化：

```python
# q99 归一化: 映射到 [-1, 1]
normalized = 2 * (x - q01) / (q99 - q01) - 1

# mean_std 归一化
normalized = (x - mean) / std

# binary 归一化 (用于 gripper)
normalized = (x > threshold).float()
```

---

## 4. 常见陷阱

### 4.1 陷阱 1: modality.json 包含未定义的键

**问题**: 数据集的 `modality.json` 包含 `data_config.py` 中没有定义的键（如 `joint_position`）。

**症状**: 
```
AssertionError: Multiple modality configs for modality action: [
  '{"eef_position": ..., "gripper": ...}',
  '{"eef_position": ..., "joint_position": ..., "gripper": ...}'
]
```

**原因**: 旧版代码没有正确过滤，导致 `modality.json` 中的所有键都被包含。

**解决**: 当前版本已修复，只包含 `modality_configs` 中定义的键。

### 4.2 陷阱 2: include_action=False 导致的过滤失效

**问题**: 当 `include_action=False` 时，`modality_configs` 中没有 action，导致过滤条件失效。

**旧代码逻辑**:
```python
if modality in used_keys and subkey not in used_keys[modality]:
    continue
# 当 action 不在 used_keys 时，条件为 False，所有键都被包含！
```

**解决**: 当前版本添加了额外检查：
```python
if used_keys and modality not in used_keys:
    continue  # 整个 modality 跳过
```

### 4.3 陷阱 3: 同一 embodiment_tag 下 state/action 键不一致

**问题**: 两个 FRANKA 数据集使用不同的 state keys。

**示例**:
```python
# libero (FRANKA)
state_keys = ["state.eef_position", "state.gripper"]

# robomind (FRANKA) 
state_keys = ["state.eef_position", "state.eef_orientation", "state.gripper"]
```

**症状**: Assertion error in `merge_metadata()`

**解决**: 确保同一 `embodiment_tag` 下的所有数据集在 `data_config.py` 中使用相同的 state/action 键。

### 4.4 陷阱 4: Video 分辨率不一致

**问题**: 不同数据集的视频分辨率不同。

**现状**: **不再是问题**，因为：
1. Video 采用并集合并，允许不同分辨率
2. 实际分辨率会被 `img_resize` 参数覆盖
3. `VideoResize` transform 会统一尺寸

### 4.5 陷阱 5: 统计信息的维度不匹配

**问题**: 合并统计信息时，不同数据集的同一键具有不同的维度。

**示例**:
```python
# 数据集 A: eef_position 是 3 维
{"eef_position": {"mean": [0.1, 0.2, 0.3], ...}}

# 数据集 B: eef_position 是 6 维
{"eef_position": {"mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ...}}
```

**症状**: `ValueError: operands could not be broadcast together`

**解决**: 确保 `data_config.py` 中定义的键在所有数据集中具有相同的维度（通过 `modality.json` 的 `start`/`end` 索引定义）。

### 4.6 陷阱 6: 遗漏 ROBOT_TYPE_TO_EMBODIMENT_TAG 映射

**问题**: 在 `mixtures.py` 中添加新数据集，但忘记在 `embodiment_tags.py` 中添加映射。

**症状**: 
```
ValueError: Robot type 'xxx' not found in ROBOT_TYPE_TO_EMBODIMENT_TAG mapping
```

**解决**: 在 `embodiment_tags.py` 中添加新的映射：
```python
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "new_robot_type": EmbodimentTag.SOME_TAG,
    ...
}
```

### 4.7 陷阱 7: 统计信息合并权重与采样概率不匹配 (已修复)

**问题**: 在旧版本中，`update_metadata()` 传递所有数据集的权重给 `merge_metadata()`，但只有当前 tag 的数据集参与合并。

**示例**:
```
数据集顺序: [A(FRANKA), C(UR), B(FRANKA)]
权重: [0.5, 0.2, 0.3]

合并 FRANKA tag 时:
  metadatas = [metadata_A, metadata_B]
  传递的权重 = [0.5, 0.2, 0.3]  (错误！)
  
  task_idx=0 → w_i = 0.5 (A 的权重，正确)
  task_idx=1 → w_i = 0.2 (C 的权重，错误！应该是 B 的 0.3)
```

**症状**: 统计信息的加权平均计算错误，导致归一化参数不准确

**状态**: **已修复** - 现在按 tag 分组权重，确保权重与数据集正确匹配

---

## 5. 数据流总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           数据管线流程图                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  data_config.py                 modality.json              info.json        │
│  ┌─────────────┐               ┌─────────────┐           ┌──────────┐       │
│  │ video_keys  │               │ state: {...}│           │ features │       │
│  │ state_keys  │               │ action:{...}│           │ video    │       │
│  │ action_keys │               │ video: {...}│           │ metadata │       │
│  └──────┬──────┘               └──────┬──────┘           └────┬─────┘       │
│         │                             │                       │             │
│         ▼                             ▼                       ▼             │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    _get_metadata()                               │       │
│  │  1. 从 modality_configs 提取 used_keys                           │       │
│  │  2. 遍历 modality.json，只保留 used_keys 中的键                   │       │
│  │  3. 从 info.json 获取 video 元数据 (resolution, fps)              │       │
│  │  4. 从 stats_gr00t.json 获取统计信息                              │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                    │                                        │
│                                    ▼                                        │
│                          DatasetMetadata                                    │
│                    ┌─────────────────────────┐                              │
│                    │ statistics: {state, action}                            │
│                    │ modalities: {video, state, action}                     │
│                    │ embodiment_tag: FRANKA                                 │
│                    └─────────────────────────┘                              │
│                                    │                                        │
│              ┌─────────────────────┼─────────────────────┐                  │
│              ▼                     ▼                     ▼                  │
│         Dataset A             Dataset B             Dataset C               │
│         (FRANKA)              (FRANKA)              (UR)                    │
│                                    │                                        │
│              └─────────────────────┼─────────────────────┘                  │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                    merge_metadata()                              │       │
│  │  1. 按 embodiment_tag 分组                                        │       │
│  │  2. Video: 并集合并 (允许不同相机)                                 │       │
│  │  3. State/Action: 严格一致性检查                                  │       │
│  │  4. Statistics: 加权平均合并                                      │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                    │                                        │
│                                    ▼                                        │
│                       merged_metadata[tag]                                  │
│                              │                                              │
│                              ▼                                              │
│                    set_transforms_metadata()                                │
│                    (用于 StateActionTransform 归一化)                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 检查清单

添加新数据集时，请确认：

- [ ] `modality.json` 中定义了所需的所有键
- [ ] `data_config.py` 中创建了对应的配置类
- [ ] `ROBOT_TYPE_CONFIG_MAP` 中添加了映射
- [ ] `ROBOT_TYPE_TO_EMBODIMENT_TAG` 中添加了映射
- [ ] `mixtures.py` 中添加了数据集混合配置
- [ ] 同一 `embodiment_tag` 下的 state/action 键保持一致
- [ ] 统计信息文件 `stats_gr00t.json` 已生成或将自动生成
