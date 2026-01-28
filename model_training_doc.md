# 羽毛球落点检测 - 第二阶段：精修模型训练 (Model Training)

## 1. 任务定义
输入一个 **11帧的时间序列**（包含图像和轨迹特征），判断该序列的 **中心帧 (第5帧)** 是否为真实的羽毛球落点。这是一个典型的 **二分类问题 (Binary Classification)**。

## 2. 模型架构：Spatiotemporal Fusion Net (STF-Net)

模型采用双流架构（Dual-Stream），分别处理视觉和几何信息，最后通过时序聚合进行分类。

### 2.1 视觉流 (Visual Stream - CNN)
*   **输入**: $B \times 11 \times 3 \times 96 \times 96$
*   **处理**: 
    1.  将 Batch 和 Time 维度合并 ($B \cdot 11, 3, 96, 96$)。
    2.  通过轻量级 CNN (4层卷积层 + BatchNorm + ReLU + MaxPool)。
    3.  Flatten 后通过全连接层映射到 512 维特征。
*   **输出**: $B \times 11 \times 512$

### 2.2 几何流 (Geometric Stream - MLP)
*   **输入**: $B \times 11 \times 8$ (包含 x, y, dx, dy, ddx, ddy, visibility, pred)
*   **处理**: 通过简单的 MLP (Linear(8->64) -> ReLU)。
*   **输出**: $B \times 11 \times 64$

### 2.3 融合与时序建模 (Fusion & Temporal)
*   **特征拼接**: 将视觉特征和几何特征在通道维拼接 $\rightarrow 512 + 64 = 576$ 维。
*   **BiLSTM**: 
    *   输入: $B \times 11 \times 576$
    *   隐藏层: 128 (双向, 输出维度 256)
    *   **核心逻辑**: BiLSTM 允许每一帧的特征都聚合“过去”和“未来”的信息。
*   **中心帧提取**:
    *   我们只关心序列中间那一刻的状态。
    *   提取 BiLSTM 输出序列的第 5 个时间步的特征向量 ($B \times 256$)。

### 2.4 分类头 (Classifier Head)
*   **结构**: Dropout(0.5) -> Linear(256 -> 64) -> ReLU -> Linear(64 -> 1)。
*   **输出**: Logits (通过 Sigmoid 转换为概率)。

## 3. 训练策略

*   **数据增强 (Data Augmentation)**:
    *   目前实现为简单的归一化处理。如有需要，可在后续迭代中加入几何翻转或颜色抖动。
*   **损失函数 (Loss)**: `BCEWithLogitsLoss` (二元交叉熵)。
*   **优化器**: `Adam` (Learning Rate = 1e-4)。
*   **早停机制 (Early Stopping)**: 监控验证集 Loss/F1，如果多次不提升则停止。
*   **阈值自动搜索**: 在验证阶段，脚本会自动在 [0.05, 0.95] 范围内搜索产生最高 F1 Score 的最佳分类阈值。
*   **正负样本自动加权**: 代码会自动统计训练集中的正负样本比例，并根据公式 `pos_weight = neg_count / pos_count * 2.0` 动态计算 Loss 权重，以解决样本不平衡问题。

## 4. 目录结构
*   `dataset_v2/train/`: 存放 NPZ 训练数据
*   `checkpoints/`: 存放训练好的模型权重 `.pth`
*   `train_refiner.py`: 训练脚本
*   `model_refiner.py`: 模型定义文件
