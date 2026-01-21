# VGGT-SLAM 详细使用说明（中文）

本文档面向**完整流程**（安装 → 数据准备 → 启动运行 → 语义优化 → 消融实验 → 结果解析）。

---

## 目录
1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [基础运行（非语义）](#3-基础运行非语义)
4. [语义优化相关配置](#4-语义优化相关配置)
5. [走廊数据的建议用法](#5-走廊数据的建议用法)
6. [消融实验脚本](#6-消融实验脚本)
7. [常见问题](#7-常见问题)

---

## 1. 环境准备

### 1.1 系统依赖
```bash
sudo apt-get update
sudo apt-get install -y git python3-pip libboost-all-dev cmake gcc g++ unzip libgl1
```

### 1.2 Python 环境
```bash
conda create -n vggt-slam python=3.11
conda activate vggt-slam
```

### 1.3 安装项目依赖
```bash
./setup.sh
```

> 如果你使用的是本仓库的集成版本，还需要保证 `vggt` 与 `salad` 子模块可被导入。

---

## 2. 数据准备

### 2.1 走廊数据（相似纹理 / 长直走廊）
建议将走廊数据整理成如下结构建议（示例）：
```
DATA/
  corridor/
    rgb/
      000001.png
      000002.png
      ...
```

只要保证是**连续帧的图像序列**即可。

---

## 3. 基础运行（非语义）

### 4.1 最小示例
```bash
python main.py \
  --image_folder office_loop \
  --max_loops 1
```

### 4.2 基础参数说明
- `--image_folder`：图像目录
- `--max_loops`：最大回环数
- `--use_sim3`：启用 Sim(3) 优化模式（较稳定）

---

## 4. 语义优化相关配置

语义优化包含两部分：
1. **语义相似度 (semantic sim)** → 衡量可信度
2. **retrieval embedding margin** → 衡量唯一性

你可以通过以下参数控制语义逻辑（详见 `main.py --help`）：

- `semantic_min_sim`：语义相似度最小阈值
- `semantic_gate_mode`：语义 gate 模式（off / filter / retrieved / both）
- `semantic_weight_mode`：语义权重模式（off / loop_only / all_edges）
- `semantic_u_enable`：启用唯一性
- `semantic_u_m0`、`semantic_u_min`：唯一性曲线控制
- `semantic_loop_margin_thr`：走廊低 margin 降权阈值
- `semantic_loop_sim_margin_thr`：语义相似度 best-second 过小则降权（走廊高相似歧义）
- `semantic_dynamic_min_sim`：是否启用动态语义阈值（EMA）
- `semantic_dynamic_min_sim_alpha`：EMA 平滑系数（越大越平滑）
- `semantic_dynamic_min_sim_margin`：动态阈值偏移（EMA - margin）
- `loop_geom_inlier_thr`：loop 几何一致性阈值（误差阈）
- `loop_geom_min_inliers`：loop 最小 inlier 数（不足直接拒绝）
- `loop_geom_ref_inliers`：用于几何权重归一化的参考 inlier 数
- `loop_geom_max_mean_err`：loop 平均几何误差上限（过大直接拒绝）

---

## 5. 走廊数据的建议用法

走廊场景常见问题是**高相似但低唯一**，因此建议配置：

- `semantic_min_sim`: 0.2 ~ 0.3
- `semantic_u_enable`: True
- `semantic_loop_margin_thr`: 0.01 ~ 0.03
- `semantic_u_min`: 0.2 ~ 0.3

这会使得**语义相似但不唯一的回环候选自动降权**，避免错误闭环主导优化。

补充策略：
- 语义仅作为**软证据**（soft gate），不会直接删掉 loop。
- loop 必须通过几何一致性门槛（inlier/误差），几何不过关的回环会被拒绝。

---

## 6. 消融实验脚本

提供两套四组消融脚本（**无图像输出** / **带图像输出**）：
- `run_ablation_4groups.py`（无图像输出）
- `run_ablation_4groups_vis.py`（带图像输出，可用 `--save_images` 导出输入样例图）

四组消融固定为：
- baseline
- gate_only
- weight_only
- gate_weight

无图像输出示例命令：
```bash
PYTHONPATH=./vggt:./salad python run_ablation_4groups.py \
  --data DATA/corridor/rgb \
  --out_dir RUNS/ablation_corridor \
  --submap_size 16 --max_loops 1 \
  --min_disparity 50 --conf_threshold 25 \
  --semantic_min_sim 0.25
```

带图像输出示例命令：
```bash
PYTHONPATH=./vggt:./salad python run_ablation_4groups_vis.py \
  --data DATA/corridor/rgb \
  --out_dir RUNS/ablation_corridor_vis \
  --submap_size 16 --max_loops 1 \
  --min_disparity 50 --conf_threshold 25 \
  --semantic_min_sim 0.25 \
  --save_images
```

输出结构（示例）：
```
RUNS/ablation_corridor/
  all_runs.csv
  edge_stats_all.csv
  baseline/edge_stats.csv
  gate_only/edge_stats.csv
  weight_only/edge_stats.csv
  gate_weight/edge_stats.csv
```

---

## 7. 常见问题

### 8.1 `ImportError: libGL.so.1`
解决：
```bash
sudo apt-get install -y libgl1
```

### 8.2 CUDA 不可用
- 确认驱动与 CUDA 版本匹配
- 若无 GPU，可继续使用 CPU（但速度会明显变慢）

### 8.3 语义模型下载慢
- 默认会从 HuggingFace 下载 `VGGT` 权重
- 建议提前下载并缓存（放到 `~/.cache/torch/hub/checkpoints/`）

---

如需进一步的走廊消融策略、参数建议、或输出图表分析，可在实验完成后继续补充统计分析脚本。
