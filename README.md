# TrackNetv3 + CatBoost + Refiner 项目总览

本项目用于**羽毛球落点检测**，分为两阶段：
1. **CatBoost 轨迹模型**：基于轨迹几何特征给出粗预测（候选落点）。
2. **Refiner 精修模型（CNN + BiLSTM）**：结合视频 ROI 与几何序列，对候选点进行二次筛选。

本 README 会按“文件 -> 功能 -> 使用方式”给出详细说明，并提供完整的执行流程。

---

## 目录结构概览

```
.
├── stroke_model.py
├── stroke_model.cbm
├── convert_data.py
├── diagnose_labels.py
├── generate_dataset.py
├── model_refiner.py
├── train_refiner.py
├── predict_refiner.py
├── visualize_predictions.py
├── run_pipeline.py
├── dataset_construction_plan.md
├── model_training_doc.md
├── refine_inference_doc.md
├── data/
│   ├── train/
│   └── test/
├── dataset_v2/
├── checkpoints/
├── catboost_info/
├── predictions/ (可视化输出)
├── refined_visualizations/ (可视化输出)
├── predict.csv
├── predicted_bounces.csv
├── refined_bounces.csv
└── val_0.4.csv
```

---

## 一、核心脚本与功能说明

### 1) stroke_model.py
**功能**：
- 训练 CatBoost 回归模型（落点概率）。
- 输出粗预测 `predict.csv` 和候选落点 `predicted_bounces.csv`。

**使用方式**：
- 训练并评估：直接运行脚本（默认会 `main()` + `predict()`）

**输出**：
- `stroke_model.cbm`：模型文件
- `predict.csv`：全量帧预测
- `predicted_bounces.csv`：阈值筛选的候选落点

---

### 2) stroke_model.cbm
**功能**：
- CatBoost 模型权重文件。
- 由 `stroke_model.py` 训练后生成。

---

### 3) convert_data.py
**功能**：
- 将 TrackNet 原始 CSV + 标签 JSON 转换为 `bounce_train.json`。
- 同时写入 `video_file` 和 `event_cls` 字段。

**使用方式**：
- 直接运行，默认处理 `data/train` 与 `data/test`。

---

### 4) diagnose_labels.py
**功能**：
- 辅助检查标注质量或统计信息（具体以脚本内容为准）。

---

### 5) generate_dataset.py
**功能**：
- 构建 Refiner 训练数据集。
- 从视频中裁剪 ROI（96x96），并生成几何序列特征：
  `[x, y, dx, dy, ddx, ddy, visibility, pred_score]`。

**使用方式**：
- 运行脚本生成 `dataset_v2/train/*.npz`。

---

### 6) model_refiner.py
**功能**：
- Refiner 模型定义（视觉 CNN + BiLSTM + 多模态融合）。

---

### 7) train_refiner.py
**功能**：
- 训练 Refiner 模型。
- 自动计算正负样本权重（`pos_weight`）。
- 自动搜索最佳阈值并保存到 `checkpoints/best_refiner_threshold.txt`。

**输出**：
- `checkpoints/best_refiner.pth`
- `checkpoints/best_refiner_threshold.txt`

---

### 8) predict_refiner.py
**功能**：
- 使用 Refiner 对候选点进行二次筛选。
- 已优化为**缓存帧**进行加速推理。

**使用方式**：
- 使用全量 `predict.csv`：
  `python predict_refiner.py --input predict.csv --model checkpoints/best_refiner.pth --output refined_bounces.csv --threshold 0.95`

- 使用候选 `predicted_bounces.csv`：
  `python predict_refiner.py --input predict.csv --candidates predicted_bounces.csv --model checkpoints/best_refiner.pth --output refined_bounces.csv --threshold 0.95`

---

### 9) visualize_predictions.py
**功能**：
- 统一可视化入口。
- 同时展示：
  - CatBoost 预测点（红色）
  - Refiner 精修点（紫色）
  - 真实标注落点（绿色）

**使用方式**：
- 仅 CatBoost：
  `python visualize_predictions.py`

- 叠加 Refiner：
  `python visualize_predictions.py --refined-csv refined_bounces.csv`

- 只看单个视频：
  `python visualize_predictions.py --refined-csv refined_bounces.csv --only-video 1_05_02.mp4`

- 只渲染前 N 帧：
  `python visualize_predictions.py --refined-csv refined_bounces.csv --limit-frames 500`

---

### 10) run_pipeline.py
**功能**：
- 一键化入口，串联：
  1. CatBoost 生成 `predict.csv` / `predicted_bounces.csv`
  2. Refiner 精修输出 `refined_bounces.csv`
  3. 可视化输出视频

**使用方式**：
- 默认运行完整流程：
  `python run_pipeline.py`

- 跳过 CatBoost：
  `python run_pipeline.py --skip-catboost`

- 跳过 Refiner：
  `python run_pipeline.py --skip-refiner`

- 跳过可视化：
  `python run_pipeline.py --skip-visualize`

- 指定候选过滤阈值：
  `python run_pipeline.py --candidate-threshold 0.4`

- 只可视化某个视频：
  `python run_pipeline.py --only-video 1_05_02.mp4`

---

## 二、文档说明

### dataset_construction_plan.md
- 数据集构建方案（生成 Refiner 训练数据）。

### model_training_doc.md
- Refiner 模型结构说明与训练策略。

### refine_inference_doc.md
- Refiner 推理流程和参数说明。

---

## 三、输出文件说明

### predict.csv
- 全量帧预测输出（CatBoost）。
- 字段：`timestamp, x, y, pred, event_cls, source_video`。

### predicted_bounces.csv
- CatBoost 候选落点（阈值过滤）。
- 字段：`timestamp, x, y, pred, source_video`。

### refined_bounces.csv
- Refiner 精修输出。
- 字段：`timestamp, x, y, pred, source_video`。

### val_0.4.csv
- 在训练评估时保存的阈值为 0.4 的验证集输出（调试用）。

---

## 四、推荐执行流程（从零开始）

1. 数据转换（已有时可跳过）
   `python convert_data.py`

2. 训练 CatBoost
   `python stroke_model.py`

3. 构建 Refiner 数据集
   `python generate_dataset.py`

4. 训练 Refiner
   `python train_refiner.py`

5. Refiner 精修
   `python predict_refiner.py --input predict.csv --candidates predicted_bounces.csv --model checkpoints/best_refiner.pth --output refined_bounces.csv --threshold 0.95`

6. 可视化检查
   `python visualize_predictions.py --refined-csv refined_bounces.csv`

---

## 五、常见问题

### 1) 为什么 timestamp 不是帧号？
`timestamp` 是毫秒级时间戳，转换关系是：
`frame_idx = round(timestamp * fps / 1000)`

### 2) 为什么 Refiner 精修比 CatBoost 更准？
Refiner 不仅看轨迹几何变化，还看球是否真实触地的图像特征，可以有效排除“击球误判”。

### 3) 推理速度慢怎么办？
`predict_refiner.py` 已做缓存优化，建议使用 `--candidates predicted_bounces.csv`，只处理候选点。

---

## 六、环境依赖（建议）
- Python 3.8+
- catboost
- pandas
- numpy
- opencv-python
- torch / torchvision

---

如需进一步优化或新增功能（例如自动评估指标、统一GUI、批量视频并行推理），可继续扩展。