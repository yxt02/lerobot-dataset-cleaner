# lerobot-dataset-cleaner

LeRobot v2.1 数据集清洗工具：检测并清理无效片段，为模仿学习 / VLA 训练准备数据。

**设计原则：优先清理无效片段，不直接删除所有被判定为异常的帧。** 迟疑段与模糊帧默认只标记，由人工或下游流程决定是否剔除。

## 清洗项目

| 项目 | 检测方法 | 处理方式 | 优先级 |
|---|---|---|---|
| **启停冗余帧** | 轨迹开头/结尾持续静止的片段（本体运动 ∪ 视频背景运动） | **裁剪**无效等待区间 | ⭐⭐⭐⭐⭐ |
| **迟疑帧** | 长时间无有效操作、无状态变化的区间 | **默认只标记**，`--trim-hesitation` 才剔除 | ⭐⭐⭐⭐⭐ |
| **模糊帧** | Laplacian 方差清晰度评分 | **只标记**异常片段，从不自动删除 | ⭐⭐⭐⭐ |

## 特性

- **机型无关**：自动使用所有 `observation.*_positions` 通道做运动检测，z0（25 维）、z02（22 维）等布局均可直接跑
- **量纲无关**：每维按自身量程归一化后再比阈值，弧度与编码器计数（0–4096）混用也能正确判定
- **防误删行驶段**：移动底座在行驶时手臂不动、AGV 通道可能全 0。脚本用「视频背景运动」区分
  `travel_hold`（底盘行驶/保持，**保留**，是底盘轨迹拼接的目标窗口）与 `hesitation`（真静止，标记）
- **坏视频可检测**：mp4 缺 moov（录制未 finalize / 传输截断）时，自动从 mdat 的 AVCC 流抢救解码前缀帧，模糊检测与背景运动检测照常工作
- **不改动原数据**：清洗结果写入新的 `--dst`；额外 meta 文件（outcomes / returns / 自定义 stats 等）自动带过去
- **视频对齐**：脚本不改写视频文件；`video_manifest.json` 记录每集需同步裁剪的帧区间，视频修复后按它对齐

## 安装

```bash
pip install -r requirements.txt
# 基础功能只需 numpy / pandas / pyarrow
# 视频检测（模糊帧、背景运动）需要 imageio-ffmpeg（自带 ffmpeg）或系统 ffmpeg
```

## 用法

### 1. 先出报告（推荐先跑这个，不动数据）

```bash
python3 -I clean_lerobot_dataset.py --src /path/to/dataset --report
```

打印每集的首尾静止帧数、迟疑帧位置、迟疑段/行驶段数量、模糊段数量，并把完整结果写到
`<src>/meta/cleaning_report.json`。

### 2. 生成清洗后的数据集

```bash
python3 -I clean_lerobot_dataset.py \
  --src /path/to/dataset \
  --dst /path/to/dataset_cleaned \
  --apply
```

默认行为：
- 裁掉开头/结尾的静止冗余帧
- 迟疑段**只标记**（写入 `meta/phases.jsonl`，不删除）
- 模糊段**只标记**（不删除）

### 3. 选择性剔除迟疑段

看过标记、确认这些区间确实该去掉之后：

```bash
python3 -I clean_lerobot_dataset.py --src /path/to/dataset --dst /path/to/out --apply --trim-hesitation
```

## 输出内容

```
<dst>/
├── data/chunk-000/episode_*.parquet   # 清洗后的轨迹（索引/时间戳已重排）
├── videos/...                          # 脚本不改写视频；请按 video_manifest.json 对齐
├── video_manifest.json                 # 每集需裁剪的帧区间（视频修复后用）
└── meta/
    ├── info.json / episodes.jsonl / episodes_stats.jsonl / tasks.jsonl
    ├── phases.jsonl                    # 启/停/迟疑/模糊标记（最终坐标 + 原始坐标）
    ├── cleaning_report.json            # 全部检测结果
    └── …                               # 源数据集的其它 meta 文件原样带过来
```

`meta/phases.jsonl` 每行一个 episode：

```json
{
  "episode_index": 0,
  "length_final": 1272,
  "frames_final":   { "hesitation": 800, "arm_end": 1271 },
  "frames_original": { "hesitation": 827, "arm_end": 1497 },
  "marks": {
    "idle_gaps": [ {"kind": "travel_hold", "start": 402, "end": 800},
                   {"kind": "hesitation", "start": 848, "end": 900} ],
    "blur_segments": [ {"start": 0, "end": 0} ]
  },
  "trimmed": { "leading": 27, "trailing": 55, "hesitation_gaps": [] }
}
```

`frames_final.hesitation` 就是「底盘→手臂切换」的迟疑帧，可直接用作底盘轨迹拼接的右对齐锚点。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--src` | 必填 | 源数据集根目录 |
| `--dst` | — | 输出目录（`--apply` 时必填，必须不存在） |
| `--report` / `--apply` | report | 只出报告 / 写清洗集 |
| `--trim-hesitation` | 关 | 连同迟疑段一起删除（默认只标记） |
| `--no-trim-bounds` | 关 | 首尾静止也只标记不裁剪 |
| `--joint-thr` | 2.5e-3 | 运动阈值，归一化单位（每帧位移占该维量程的比例） |
| `--agv-thr` | 1e-4 | AGV/底盘速度通道阈值（绝对值） |
| `--bg-thr` | 1.2 | 视频背景运动阈值（灰度 MAD） |
| `--blur-threshold` | 300 | Laplacian 方差阈值，低于此值标记为模糊。160×90 解码标定：正常画面 500–2000 |
| `--min-idle-sec` | 1.0 | 静止段最短时长，短于此不标记 |
| `--min-blur-sec` | 0.5 | 模糊段最短时长 |
| `--max-boundary-trim-sec` | 5.0 | 视频不可读且 AGV 全 0 时，首尾裁剪的安全上限 |
| `--max-video-frames` | 2000 | 每集最多解码多少帧用于检测 |
| `--skip-video` | 关 | 纯本体检测（跳过模糊/背景运动） |

## 典型工作流

```bash
# 1) 看看数据里有什么问题
python3 -I clean_lerobot_dataset.py --src data/raw/2026.09.15_2 --report

# 2) 清洗（裁启停冗余，迟疑/模糊留标记）
python3 -I clean_lerobot_dataset.py --src data/raw/2026.09.15_2 --dst data/clean/2026.09.15_2 --apply

# 3) 人工抽查 meta/phases.jsonl 里的迟疑标记，确认后再决定是否剔除
python3 -I clean_lerobot_dataset.py --src data/raw/2026.09.15_2 \
  --dst data/clean/2026.09.15_2_nohes --apply --trim-hesitation

# 4) 视频若需修复/裁剪，按 video_manifest.json 中的 drop_prefix / drop_intervals / drop_suffix 对齐
```

## 注意

- **图像统计**（`episodes_stats.jsonl` 中 video 特征）不被重算，请在视频处理完毕后用 LeRobot 自带工具重新生成
- 裁剪会重排 `frame_index` / `timestamp` / 全局 `index`，`phases.jsonl` 同时保留原始坐标便于回溯
- 若数据集的底盘速度通道是活的（非全 0），脚本会自动用它辅助识别行驶段；否则依靠视频背景运动，两者都没有时限制裁剪幅度以防误删

## 已验证

| 数据集 | 机型 | 集数 | 视频状态 | 结果 |
|---|---|---|---|---|
| `first_2026.10.16` | z0（25 维，含灵巧手/AGV） | 100 | 300 个 mp4 缺 moov（截断） | 抢救解码成功；检出启停冗余、286 个迟疑段、103 个行驶段、0 模糊段 |
| `2026.09.15_2` | z02（22 维，不同命名） | 112 | 112 个 mp4 完好 | 全流程通过；数据本身已密实，仅裁 19 帧首尾静止；索引/内容连续性校验无误 |

## License

MIT
