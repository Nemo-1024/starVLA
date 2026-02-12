# Episode–File 映射与数据加载管线分析

## 1. 结论摘要

- **数据索引（文件存在性、引用关系）是正确的**：`check_orphan_files.py` / `check_missing_files.py` 的结论成立。
- **不可靠的是 `meta/episodes` 里的 `(data/chunk_index, data/file_index)` 映射**：即“某个 `episode_index` 的步数据实际在哪个 `chunk-X/file-Y`”的元数据有误。
- **当前 `datasets.py` 的数据加载管线设计是合理的**：通过 `_build_episode_to_file_map` 用**真实数据文件**重新建映射，绕开错误的 meta，从而避免大量“轨迹为空”的问题。管线没有错，而是在**正确**地规避错误元数据。

因此：**不是数据加载管线错了，而是必须依赖 `_build_episode_to_file_map`，而不能只信 `meta/episodes` 的 `(chunk_index, file_index)`。**

---

## 2. check 脚本在验证什么

### 2.1 逻辑

- 读 **`meta/episodes/**/*.parquet`**，对每一行用 **`data/chunk_index`**、**`data/file_index`**。
- 检查：
  1. 这些 `(chunk, file)` 对应的 **`data/chunk-XXX/file-YYY.parquet` 是否都存在于磁盘**；
  2. 是否存在**未被任何 episode 引用的孤立文件**。

### 2.2 未验证的内容

- **没有**验证：对每个 `episode_index`，meta 里给的 `(chunk, file)` 是否就是**真正含有该 episode 步数据**的那个文件。
- 即：只保证“被引用的文件都存在”，**不**保证“每个 episode 指向的文件里确实有它的 `episode_index`”。

因此会出现：

- 所有被引用的 `(chunk, file)` 都存在 → check 通过；
- 但某些 episode 的 `(chunk, file)` 指错了 → 用 meta 加载时，去错文件、按 `episode_index` 过滤 → **得到空轨迹**。

---

## 3. 数据加载管线在做什么

### 3.1 流程概要

1. **Trajectory ID** = `episode_index`（来自 `meta/episodes`）。
2. 对每个 trajectory，要找到**步数据所在文件**：  
   `data/chunk-{chunk_index}/file-{file_index}.parquet`。
3. 打开该 parquet，用 `episode_index` 过滤行，得到该 trajectory 的步数据。
4. 若过滤结果为空 → 报错“轨迹为空”。

### 3.2 映射的两种来源

| 来源 | 含义 | 使用方式 |
|------|------|----------|
| **meta/episodes** | 每行自带的 `data/chunk_index`、`data/file_index` | 直接当成“该 episode 所在文件” |
| **`_build_episode_to_file_map`** | 扫 **`data/**/*.parquet`**，按 `episode_index` 列建 `episode_index → (chunk, file)` | 以**真实数据文件**为准 |

当前逻辑（`_get_trajectories`）：

- **优先**用 `_build_episode_to_file_map` 的结果；
- 若某 `episode_index` 在 map 里找不到，才 **fallback** 到 meta 的 `(chunk, file)`。

### 3.3 为何会“大量轨迹为空”

- 若**不用** `_build_episode_to_file_map`，只信 meta：
  - 对部分 episode，meta 的 `(chunk, file)` **指向了错误文件**；
  - 该文件存在（故 check 通过），但**不包含**该 `episode_index` 的步数据；
  - 过滤 → 空 → 报错。
- 使用 **`_build_episode_to_file_map`** 后：
  - 映射来自**真实数据文件**中的 `episode_index`；
  - 每个 episode 被指向**真正含有其数据**的文件；
  - 过滤有结果 → 不再出现“轨迹为空”。

所以：“**必须用 `_build_episode_to_file_map` 才能正常加载**”恰恰说明 **meta 的映射不可信**，而**管线通过建 map 纠正了这一点**。

---

## 4. 根因归纳

- **数据索引**：  
  - 哪些 `(chunk, file)` 存在、是否被引用 → **正确**，与 check 脚本结论一致。
- **meta 的 episode → file 映射**：  
  - 对部分 `episode_index`，`(data/chunk_index, data/file_index)` **与真实数据所在文件不一致** → **不可靠**。
- **数据加载管线**：  
  - 依赖 `_build_episode_to_file_map` 以数据文件为准建映射 → **设计正确**，且是修复“空轨迹”问题的必要手段。

---

## 5. 建议

### 5.1 保持现有管线设计

- 继续 **默认使用 `_build_episode_to_file_map`**，仅对 map 中缺失的 `episode_index` 才 fallback 到 meta。
- 不要改为“只信 meta、不做 scan”，否则会重新引入大量空轨迹。

### 5.2 可选：增强校验脚本

可增加一个 **episode–file 一致性检查**（例如 `check_episode_file_consistency.py`）：

- 对每个 `meta/episodes` 中的 `(episode_index, chunk_index, file_index)`：
  - 打开 `data/chunk-{chunk}/file-{file}.parquet`；
  - 检查是否存在 `episode_index` 行；
- 若有 episode 在声称的文件中找不到 → 报错，便于发现 meta 映射错误。

### 5.3 可选：修正 meta 本身

若希望**长期**不依赖 scan、仅用 meta 也能正确加载：

- 用 `_build_episode_to_file_map`（或等价逻辑）生成**正确**的 `episode_index → (chunk, file)`；
- 重写 `meta/episodes` 的 `data/chunk_index`、`data/file_index`，使其与数据文件一致；
- 再跑 check 与上述一致性脚本，确认通过。

---

## 6. 总结

- Check 脚本验证的是**文件存在性与引用关系**，**不**验证 **episode_index ↔ 实际所在文件** 是否一致。
- 你观测到的现象（**有 map 才能正确加载、否则大量空轨迹**）说明 **meta 的 episode–file 映射有误**，而非数据加载管线有误。
- 当前 **`datasets.py` 数据加载管线是正确的**：通过 `_build_episode_to_file_map` 以数据文件为真源建映射，规避错误 meta，从而消除空轨迹问题。
