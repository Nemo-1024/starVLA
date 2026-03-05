## Relative Action Conversion + Stats Rebuild (Align with Isaac-GR00T)

### Summary
在不改变现有默认行为（默认关闭）的前提下，为 `state/action` 处理链路加入“按配置将 absolute action 转成 relative action”的能力，并补齐与之配套的 relative 统计重建、混合数据集统计合并、训练导出统计兼容。实现语义对齐 Isaac-GR00T 的 `state_action` 设计。

### Public APIs / Interface Changes
1. 在 [state_action.py](/mnt/project_rlinf/jlchen/code/starVLA/starVLA/dataloader/gr00t_lerobot/transform/state_action.py) 的 `StateActionTransform` 增加配置字段：
   - `use_relative_action: bool = False`
   - `relative_action_configs: dict[str, RelativeActionConfig] = {}`
2. 新增 `RelativeActionConfig`（放在同文件），字段对齐 GROOT 语义：
   - `state_key: str | None`（默认按 `action.xxx -> state.xxx`）
   - `action_type: {"non_eef","eef"}`（默认 `non_eef`）
   - `action_format`（保留与 GROOT 一致的格式枚举入口）
3. 在 [schema.py](/mnt/project_rlinf/jlchen/code/starVLA/starVLA/dataloader/gr00t_lerobot/schema.py) 的 `DatasetStatistics` 增加可选字段：
   - `relative_action: dict[str, DatasetStatisticalValues] | None = None`
4. 新增缓存文件约定：
   - `meta/relative_stats_gr00t.json`

### Implementation Plan
1. 改造 [state_action.py](/mnt/project_rlinf/jlchen/code/starVLA/starVLA/dataloader/gr00t_lerobot/transform/state_action.py)：
   - 在 `set_metadata()` 解析并校验 `relative_action_configs`，解析 reference state key、维度一致性、stats 可用性。
   - `apply()` 中先执行 “absolute -> relative” 再做 rotation/normalization。
   - `unapply()` 中先 inverse normalization/rotation，再执行 “relative -> absolute”。
   - `non_eef` 路径按 GROOT：`action_chunk - reference_state(last_state)`；inverse 为加回 reference。
   - `eef` 路径按 GROOT 语义（位姿相对/绝对变换）。

2. 保障 transform 顺序正确（避免拿到已归一化 state）：
   - 在 [data_config.py](/mnt/project_rlinf/jlchen/code/starVLA/starVLA/dataloader/gr00t_lerobot/data_config.py) 的 `_build_composed_transform()` 中，当检测到 `use_relative_action=True` 时，自动重排使“action 相关 transform”先于“state 非 ToTensor transform”执行，并保持 Concat 末尾。
   - 目标：forward 时 relative conversion 用原始 state；reverse 时 state 先 unapply 再 action unapply。

3. 增加 relative 统计加载/重算并注入 metadata：
   - 在 [datasets.py](/mnt/project_rlinf/jlchen/code/starVLA/starVLA/dataloader/gr00t_lerobot/datasets.py) 增加 relative stats 常量与流程：
     - 若开启 relative 且缺少/失效 `relative_stats_gr00t.json`，自动按 episode + action horizon 计算 `(T,D)` 统计并缓存。
     - 支持 `force_recompute_stats` 强制重算。
   - 将计算结果写入 `metadata.statistics.relative_action`，并把启用 relative 的 action key 覆盖到 `metadata.statistics.action[key]`（保证训练归一化与导出统计一致）。

4. 混合数据集统计合并与导出兼容 `(T,D)`：
   - 改造 [datasets.py](/mnt/project_rlinf/jlchen/code/starVLA/starVLA/dataloader/gr00t_lerobot/datasets.py) 中 `compute_overall_statistics()` 为形状泛化（支持 1D/2D）。
   - `merge_metadata()` 合并 `relative_action`（若存在）并保持 action 侧“effective stats”覆盖策略。
   - 改造 `combine_modality_stats()`：若任一子键为 2D，则按 horizon 维拼接成 `(T,D_total)`；否则保持旧的 1D 拼接。

5. 错误处理与可观测性：
   - 缺失 reference state key、action/state 维度不匹配、缺失 relative stats、不支持格式时给出明确错误信息（包含 key 和 expected shape）。
   - 缓存文件读写保持原子性（tmp + replace）。

### Tests / Acceptance Criteria
1. `use_relative_action=False` 时，现有样本输出与改造前一致（回归不变）。
2. `non_eef` 相对转换数值正确：人工构造数据可验证每个 horizon 位置都按同一 reference state 转换。
3. `apply -> unapply` roundtrip 在 relative 配置下可恢复原 action（容许浮点误差）。
4. relative stats 自动构建后，目标 key 统计形状为 `(T,D)`，且训练时 action normalizer 使用的是 relative stats。
5. 混合数据集导出的 `dataset_statistics.json` 在 relative key 下为可用于反归一化的有效形状（1D 或 `(T,D_total)`）。

### Assumptions and Defaults
1. 默认关闭 relative 功能；只有在 `StateActionTransform` 显式配置时才生效。
2. `state_key` 未配置时默认映射 `action.xxx -> state.xxx`。
3. relative 统计默认按 horizon 逐位统计 `(T,D)`，与 GROOT 对齐。
4. 统计自动重建策略：启用 relative 时优先读缓存，缺失或 `force_recompute_stats=True` 时重算并写回缓存。
5. EEF 的相对/绝对转换按 GROOT 语义实现；若某些格式/维度在统计重算侧不支持，将按 GROOT 风格明确报错而不是静默退化。
