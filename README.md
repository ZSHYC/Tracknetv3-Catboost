# Badminton Bounce Detection System: TrackNetv3 + CatBoost + Refiner

本项目是一个多阶段的羽毛球落点检测系统。它结合了传统的机器学习（CatBoost）和深度学习（CNN+BiLSTM），实现了从粗糙候选生成到精确落点筛选的完整流程。

---

## 📖 项目背景与架构

### ⚠️ 问题
仅依靠 TrackNet 提供的球坐标轨迹（x, y），很难区分“真实落地”和“近地击球”。两者在几何轨迹上非常相似（都是V型反转），导致大量误报（False Positives）。

### ✅ 解决方案：两阶段检测流水线 (Two-Stage Pipeline)

| 阶段 | 模型 | 输入特征 | 任务 | 优势 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: 候选生成** | **CatBoost** | 轨迹几何特征 (速度, 加速度, 角度, 坐标) | **召回 (Recall)**：找出所有可能是落点的地方。 | 速度快，对几何突变敏感。 |
| **Stage 2: 精细筛选** | **STFNet (Refiner)** | **图像 (Visual)** + **几何 (Geometric)** | **准确 (Precision)**：剔除假阳性，确认真实落点。 | 融合视觉纹理（看球头朝向、是否触地）与时序特征。 |

---

## ⚡ 快速开始 (Quick Start)

如果你已经配置好环境（Python, OpenCV, PyTorch, CatBoost），请按照以下顺序执行：

### 1. 全自动运行
最简单的方法是直接运行 pipeline 脚本，它会自动串联所有步骤。
```powershell
# 运行完整流程：预测 -> 精修 -> 可视化
python run_pipeline.py

# 如果只想看结果（跳过已有的预测步骤）
python run_pipeline.py --skip-catboost --skip-refiner
```

### 2. 分步手动执行
如果你是开发者，建议分步执行以确模型状态。

```powershell
# 1. (准备) 转换数据格式
python convert_data.py

# 2. (一阶段) 训练 CatBoost 并生成初步预测
python stroke_model.py

# 3. (二阶段) 构建精修模型的训练集 (挖掘难例)
python generate_dataset.py

# 4. (二阶段) 训练精修模型 STFNet
python train_refiner.py

# 5. (推理) 运行精修推理
# 注意：threshold 建议参考 step 4 训练日志中的最佳阈值
python predict_refiner.py --input predict.csv --candidates predicted_bounces.csv --model checkpoints/best_refiner.pth --output refined_bounces.csv --threshold 0.85

# 6. (可视化) 检查最终结果
python visualize_predictions.py --refined-csv refined_bounces.csv
```

---

## 📂 目录结构说明

```text
Tracknetv3-catboost/
├── 📂 data/                        # 数据根目录
│   ├── 📂 train/                   # 训练集 matches
│   └── 📂 test/                    # 测试集 matches
│
├── 📂 dataset_v2/                  # STFNet 专用数据集 (由 generate_dataset.py 生成)
│   └── 📂 train/                   # 包含 .npz 文件 (Images + Geometry)
│
├── 📂 checkpoints/                 # 模型权重保存路径
│   ├── best_refiner.pth            # STFNet 最佳权重
│   └── best_refiner_threshold.txt  # STFNet 最佳分类阈值
│
├── 📂 catboost_info/               # CatBoost 训练日志
├── 📂 refined_visualizations/      # 最终可视化视频输出目录
│
├── 📝 stroke_model.py              # [Stage 1] CatBoost 训练与推理脚本
├── 📝 stroke_model.cbm             # [Stage 1] CatBoost 模型文件
│
├── 📝 generate_dataset.py          # [Stage 2] 数据构建 (Hard Negative Mining)
├── 📝 model_refiner.py             # [Stage 2] STFNet 模型架构定义 (PyTorch)
├── 📝 train_refiner.py             # [Stage 2] 训练脚本 (含自动阈值搜索)
├── 📝 predict_refiner.py           # [Stage 2] 推理脚本 (含帧缓存优化)
│
├── 📝 visualize_predictions.py     # [工具] 通用可视化工具
├── 📝 run_pipeline.py              # [工具] 总控脚本
├── 📝 convert_data.py              # [工具] 原始数据格式转换
├── 📝 diagnose_labels.py           # [工具] 标签诊断与EDA
│
├── 📊 predict.csv                  # 中间产物：CatBoost 对每一帧的预测
├── 📊 predicted_bounces.csv        # 中间产物：CatBoost 筛选出的候选点 (Recall High)
└── 📊 refined_bounces.csv          # 最终产物：Refiner 筛选出的最终落点 (Precision High)
```

---

## 📜 核心脚本详细说明 (Detailed Explanation)

为了让您完全掌控本项目，以下是对每个Python脚本的逐行级功能解析：

### 1. 🛠️ 数据准备类

#### `convert_data.py`
**作用**：数据清洗与格式统一。
- **输入**：原始的 TrackNet CSV 轨迹文件和 Label JSON 标注文件。
- **逻辑**：将分散在各个文件夹中的数据聚合，提取出 `(x, y)` 坐标序列，并打上 `event_cls` 标签（1为落点，0为非落点）。
- **输出**：生成 `data/train/matchX/bounce_train.json`，这是后续训练的基础。

#### `generate_dataset.py`
**作用**：构建第二阶段（Refiner）专用的多模态数据集。
- **核心逻辑**：
    1.  **加载候选**：读取 `stroke_model.py` 生成的 `predicted_bounces.csv`。
    2.  **难例挖掘**：将“模型认为是落点（分高）但标签说不是（False Positive）”的样本标记为困难负样本。
    3.  **多模态提取**：
        - **视觉**：打开视频，定位到对应帧，裁剪 **96x96** 的以球为中心的 ROI 区域，组成 11 帧序列。
        - **几何**：提取对应的坐标、速度、加速度、可见性、一阶段分数，组成向量序列。
- **输出**：`dataset_v2/train/match_X.npz` (Numpy 压缩文件，读取速度极快)。

---

### 2. 🤖 模型算法类

#### `stroke_model.py` (Stage 1: CatBoost)
**作用**：基于纯几何特征的快速初筛。
- **特征工程**：计算每一帧的 `dx` (速度), `dy` (垂直速度), `acc` (加速度), `angle` (轨迹夹角)。落点通常发生在轨迹 V 型反转处，几何特征极其明显。
- **算法**：使用 **CatBoost Regressor**，它对时序特征处理能力强且速度极快。
- **输出**：`stroke_model.cbm` (模型文件) 和 `predict.csv` (全量预测结果)。

#### `model_refiner.py` (Stage 2: STFNet)
**作用**：定义深度学习网络架构。
- **VisualEncoder**: 4层 CNN，将 (Batch, 11, 3, 96, 96) 的图像序列压缩为特征向量。
- **GeometricEncoder**: MLP，将坐标和运动特征映射到高维空间。
- **BiLSTM**: 双向长短时记忆网络，融合视觉和几何特征，理解“球触地反弹”的时序动态过程。

#### `train_refiner.py`
**作用**：训练 Refiner 模型。
- **亮点功能**：
    - **自动权重 (Auto-Weighting)**：自动计算正负样本比例，设置 `BCEWithLogitsLoss(pos_weight=...)`，解决正样本极少导致不收敛的问题。
    - **最佳阈值搜索**：训练结束后，自动要在验证集上跑一遍，从 0.05 到 0.95 搜索 F1 Score 最高的阈值并保存。

#### `predict_refiner.py`
**作用**：应用 Refiner 模型进行推理。
- **性能优化**：为了避免频繁使用 `cap.set(cv2.CAP_PROP_POS_FRAMES)` (非常慢)，该脚本实现了 **Frame Caching (帧缓存)** 机制。它会一次性加载候选点附近的一批帧到内存，复用读取结果，推理速度提升 10 倍以上。

---

### 3. 🎬 流程与可视化类

#### `run_pipeline.py`
**作用**：一键运行的总指挥。
- **逻辑**：按顺序调用 `stroke_model.py` -> `predict_refiner.py` -> `visualize_predictions.py`。支持通过命令行参数 `--skip-xxx` 跳过某些步骤，方便调试。

#### `visualize_predictions.py`
**作用**：生成直观的对比视频。
- **图例**：
    - 🟢 **绿色实心圆**：Ground Truth (真实标签)。
    - 🔴 **红色空心圆**：CatBoost (一阶段预测)。
    - 🟣 **紫色实心圆**：Refiner (二阶段精修后的预测)。
- **分析方法**：如果红圈出现而紫圈没出现，说明 Refiner 成功抑制了一个误报。

## 🧠 技术细节详解

### 1. 难例挖掘 (Hard Negative Mining)
在 `generate_dataset.py` 中，我们不仅采集了正样本（真实落点），还专门采集了 **CatBoost 认为置信度高但实际是错误的样本**。
*   **目的**：强迫 Stage 2 模型关注那些 Stage 1 书籍不好的 Corner Case。
*   **采样窗口**：11帧（前后各5帧）。
*   **图像处理**：中心裁剪 96x96，边缘填充黑色。

### 2. 双流网络 (STFNet)
在 `model_refiner.py` 中定义：
*   **Visual Stream**: 4层 CNN 提取逐帧图像特征 -> Flatten。
*   **Geometric Stream**: MLP 提取归一化坐标、速度、加速度、Stage 1 分数。
*   **Fusion**: 拼接两个流的特征，送入 **BiLSTM** 进行时序建模。
*   **Head**: 取中心帧时间步的特征进行二分类。

### 3. 动态推理优化
在 `predict_refiner.py` 中，我们没有对每一帧都读取视频（太慢）。
*   **优化**：按 Video 分组，**缓存**相关的视频帧到内存中。
*   **逻辑**：只对 `predicted_bounces.csv` 中的候选点进行精修，极大减少计算量。

---

## 🛠️ 配置与参数

大多数脚本的头部都有配置区域，主要的超参数如下：

| 文件 | 参数变量 | 默认值 |不仅 |
| :--- | :--- | :--- | :--- |
| `generate_dataset.py` | `ROI_SIZE` | `(96, 96)` | 裁剪图像大小 |
| `train_refiner.py` | `pos_weight` | *Auto* | 自动计算正负样本不平衡权重 |
| `train_refiner.py` | `LR` | `1e-4` | 学习率 |
| `predict_refiner.py` | `threshold` | `0.95` | 二分类阈值 (建议根据训练日志调整) |

---

## ❓ 常见问题 (FAQ)

**Q1: 为什么我的 Refiner 训练出来 F1 是 0？**
*   **原因**: 样本极端不平衡（负样本太多）。
*   **解决**: `train_refiner.py` 中已加入 `pos_weight` 自动加权逻辑，确保正样本 Loss 权重更大。如果依然为0，请检查 `generate_dataset.py` 是否正确生成了正样本（查看 npz 文件内容）。

**Q2: 为什么可视化视频里的框是歪的？**
*   **原因**: TrackNet 的坐标预测可能本身有抖动。
*   **说明**: 本项目只负责“分类”（是/不是落点），不负责“修正坐标”。绿圈是 Ground Truth，红圈是预测，如果不重合通常是因为预测坐标有偏差，而不是分类错误。

**Q3: 运行 `predict_refiner.py` 报错 `FileNotFound`？**
*   **check**: 确保不仅生成了 `predict.csv`，而且要有 `checkpoints/best_refiner.pth`。如果是第一次运行，必须先跑 `train_refiner.py`。

---

## 📝 开发者指南

*   **添加新特征**: 修改 `generate_dataset.py` 中的 `vectors` 提取逻辑，并在 `model_refiner.py` 中修改 `input_dim`。
*   **更换 backbone**: 可以在 `VisualEncoder` 类中替换为 ResNet18 或 MobileNet 以获得更强的视觉特征提取能力。

---

## 🧭 详细使用场景与命令示例（覆盖所有情况）

下面按使用者可能遇到的场景逐条列出命令、参数说明与快速排错步骤。

> 说明：所有命令均在项目根目录执行。示例使用 PowerShell 语法，Linux/macOS 下去掉 `$env:` 前缀或直接用 bash 执行。

1) 端到端一键运行（推荐初次验证）

```powershell
python run_pipeline.py
```

说明：顺序执行 CatBoost 预测 -> Refiner 精修 -> 可视化。若存在中间产物会尝试复用。

2) 跳过某阶段（开发常用）

```powershell
# 跳过 CatBoost（已有 predict.csv）
python run_pipeline.py --skip-catboost

# 跳过 Refiner（只看 CatBoost 输出）
python run_pipeline.py --skip-refiner

# 跳过可视化（只产出 CSV）
python run_pipeline.py --skip-visualize
```

3) 只运行/训练 CatBoost（Stage 1）

```powershell
# 训练并保存 Model
python stroke_model.py --train

# 使用已训练模型生成 predict.csv（快速）
python stroke_model.py --predict --model stroke_model.cbm --output predict.csv
```

4) 构建 Refiner 数据集（Stage 2）

```powershell
python generate_dataset.py --input-data data/train --out dataset_v2/train --roi-size 96
```

参数说明：
- `--input-data`：源数据目录（默认 `data/`）。
- `--out`：输出目录（默认 `dataset_v2/train`）。
- `--roi-size`：ROI 边长（改变后必须重生成数据并重训模型）。

5) 训练 Refiner（含恢复训练、GPU 指定）

```powershell
# 普通训练
python train_refiner.py --data dataset_v2/train --epochs 30 --batch-size 16

# 指定 GPU（PowerShell 例子）
$env:CUDA_VISIBLE_DEVICES=0; python train_refiner.py --data dataset_v2/train

# 从 checkpoint 恢复训练
python train_refiner.py --resume checkpoints/last_refiner.pth
```

重要参数：`--lr`、`--batch-size`、`--epochs`、`--resume`、`--data`。

6) Refiner 推理（两种模式：只处理候选点 / 对全量帧）

```powershell
# 推荐：只精修 CatBoost 候选点（最快）
python predict_refiner.py --input predict.csv --candidates predicted_bounces.csv --model checkpoints/best_refiner.pth --output refined_bounces.csv --threshold 0.85

# 慎用：对全量 predict.csv 做精修（非常慢）
python predict_refiner.py --input predict.csv --model checkpoints/best_refiner.pth --output refined_bounces_full.csv --threshold 0.85

# debug：只对单视频运行
python predict_refiner.py --input predict.csv --model checkpoints/best_refiner.pth --only-video 1_05_02.mp4 --output refined_single.csv
```

说明：`--threshold` 请优先使用训练阶段保存的阈值文件 `checkpoints/best_refiner_threshold.txt`。

7) 可视化选项（限制帧数 / 指定视频 / 输出目录）

```powershell
# 渲染全部（默认）
python visualize_predictions.py

# 渲染前 500 帧
python visualize_predictions.py --limit-frames 500

# 仅渲染单个视频并指定输出目录
python visualize_predictions.py --only-video 1_05_02.mp4 --outdir refined_visualizations/
```

8) 数据与样本检查命令（快速脚本）

```powershell
# 查看某个 npz 文件内部结构
python - <<'PY'
import numpy as np
arr = np.load('dataset_v2/train/match_1.npz', allow_pickle=True)
print(arr.files)
print(arr['images'].shape, arr['geo_vectors'].shape, arr['labels'].shape)
PY

# 统计正负样本数（用于判断 pos_weight）
python - <<'PY'
import numpy as np, glob
files = glob.glob('dataset_v2/train/*.npz')
pos=neg=0
for f in files:
    a=np.load(f, allow_pickle=True)
    labs=a['labels'].reshape(-1)
    pos+= (labs==1).sum(); neg+=(labs==0).sum()
print('pos',pos,'neg',neg)
PY
```

9) 常见问题快速排查（Summary）

- `FileNotFoundError: checkpoints/best_refiner.pth`：先运行 `python train_refiner.py` 或确认模型路径。
- `IndexError`/空帧：检查 `labels/*.json` 中 `fps` 与 timestamp 单位（ms）是否正确，或视频是否损坏。
- 训练 F1 为 0：查看 `dataset_v2` 中正样本是否充足；可手动增加正样本或调整 `generate_dataset.py` 的采样策略。

10) 在服务器/集群上并行化建议

- 按 `source_video` 将 `predicted_bounces.csv` 切分为多个子任务：每个进程只加载自己关心的视频帧缓存，可显著减少 IO 与内存浪费。
- 使用较小 `--batch-size` 以适配显存，避免 OOM。


