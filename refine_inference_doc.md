# 第二阶段推理流程（Refiner Inference）

## 1. 目标
在 CatBoost 产生的候选落点基础上，使用 **STFNet (CNN + BiLSTM + 多模态融合)** 进行二次筛选，输出更精准的落点列表。

## 2. 输入输出

**输入**
- `predict.csv`：由 `stroke_model.py` 生成的全帧预测文件，包含 `timestamp, x, y, pred, source_video`。
- `predicted_bounces.csv`（可选）：用于缩小候选集合（一般为 `pred > 0.4`）。
- `checkpoints/best_refiner.pth`：训练好的精修模型权重。

**输出**
- `refined_bounces.csv`：精修后的落点列表。

## 3. 核心逻辑
1. 按视频分组读取 `predict.csv`。
2. 通过 `labels/*.json` 获取每个视频的精确 `fps`。
3. 将每行 `timestamp` 转换为 `frame_idx`。
4. 对于每个候选点（来自 predicted_bounces.csv 或者自定义阈值）：
   - 取中心帧 $T$，构造 11 帧时间窗。
   - 对每帧裁剪 96x96 的 ROI。
   - 构造几何特征：
     - $x, y, dx, dy, ddx, ddy, visibility, pred$（与训练保持一致）。
5. 输入 STFNet，得到预测概率。
6. 以阈值筛选输出最终落点（推荐阈值来自训练日志，例如 0.95）。

## 4. 使用命令
```powershell
python predict_refiner.py --input predict.csv --model checkpoints/best_refiner.pth --output refined_bounces.csv --threshold 0.95
```

如果你希望只精修 `predicted_bounces.csv` 中的候选点：
```powershell
python predict_refiner.py --input predict.csv --candidates predicted_bounces.csv --model checkpoints/best_refiner.pth --output refined_bounces.csv --threshold 0.95
```

## 5. 注意事项
- 如果你改变了 `ROI_SIZE` 或几何特征格式，必须重新生成数据并重训模型。
- `threshold` 建议使用验证阶段自动搜索得到的最佳阈值。
