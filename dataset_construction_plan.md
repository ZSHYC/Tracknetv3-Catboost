# 羽毛球落点检测 - 第二阶段数据集构建方案

## 1. 目标
构建一个包含**视觉信息（图像序列）**和**几何信息（轨迹序列）**的多模态数据集。该数据集专门用于训练“后处理精修模型”，目的是区分**真实落点**与**高风险误报（如击球点、贴地飞行）**。

## 2. 数据来源
*   **输入目录**: `data/train/` 下的所有 match 文件夹。
*   **依赖文件**:
    *   `bounce_train.json`: 提供轨迹坐标 `(x, y)` 和基础标签 `event_cls`。
    *   `labels/*_labels.json`: 提供精确的 `fps` 信息，用于帧对齐。
    *   `video/*.mp4`: 提供原始图像信息。
    *   `stroke_model.cbm`: 现有的 CatBoost 模型，用于挖掘难例（Hard Negatives）。

## 3. 采样策略 (Sampling Strategy)

我们需要从训练集中挖掘三类样本，每条样本都是长度为 **11帧** 的序列（中心帧 $T$，范围 $[T-5, T+5]$）。

| 样本类型 | 标签 (Label) | 筛选条件 | 目的 |
| :--- | :--- | :--- | :--- |
| **正样本 (Positives)** | `1` (Drop) | 标注文件中 `event_cls == 1` | 教会模型什么是“落点”。 |
| **困难负样本 (Hard Negatives)** | `0` (Not Drop) | `event_cls == 0` **且** CatBoost预测分 `> 0.1` | **核心**：教会模型纠正“假阳性”（击球、转折点）。 |
| **简单负样本 (Easy Negatives)** | `0` (Not Drop) | `event_cls == 0` **且** CatBoost预测分 `< 0.1` (随机采10%) | 保持模型对普通飞行轨迹的判别力。 |

## 4. 特征提取流程

### 4.1 帧对齐 (Frame Alignment)
*   **读取 FPS**: 利用 `video_file` 对应的 `labels/*.json` 中的 `metadata.fps`。
*   **计算帧号**: $FrameIndex = \text{round}(\frac{Timestamp \times FPS}{1000})$。

### 4.2 视觉流 (Visual Stream)
*   **输入**: 11帧 RGB 图像。
*   **处理**:
    1.  以每一帧的球坐标 $(x, y)$ 为中心。
    2.  裁剪 **96 x 96** 像素区域 (ROI)。
    3.  如果球靠近边缘，使用黑色填充 (Padding) 保持尺寸一致。
    4.  如果某一帧没有球（未检测到），复用最近的有效坐标或涂黑。
*   **输出维度**: `(11, 96, 96, 3)`

### 4.3 几何流 (Geometric Stream)
*   **输入**: 11帧的坐标信息。
*   **处理**:
    1.  归一化坐标: $x_{norm} = x / 1280, y_{norm} = y / 720$。
    2.  提取特征向量: `[x_norm, y_norm, visibility]`。
*   **输出维度**: `(11, 3)` (BiLSTM 可以自动学习后续的速度和加速度特征)。

## 5. 存储格式 (Storage)
为了提高 I/O 效率，按 **Match**粒度 保存为 `.npz` 压缩文件。

*   **路径**: `dataset_v2/train/match_{id}.npz`
*   **Key-Value**:
    *   `"images"`: `uint8` 数组, shape `(N, 11, 96, 96, 3)`
    *   `"geo_vectors"`: `float32` 数组, shape `(N, 11, 3)`
    *   `"labels"`: `uint8` 数组, shape `(N, 1)`
    *   `"infos"`: 字符串数组, 记录来源 `[video_name, timestamp, type]` (用于调试)

---
