import cv2
import pandas as pd
import json
import os
import numpy as np
from catboost import CatBoostRegressor

# 参数设置
MODEL_PATH = "stroke_model.cbm"
MATCH_DIR = "data/test/match1"  # 选择测试集的 match1，可修改
VIDEO_PATH = os.path.join(MATCH_DIR, "video", "1_01_00.mp4")  # 假设视频文件名，根据实际调整
CSV_PATH = os.path.join(MATCH_DIR, "csv", "1_01_00_ball.csv")  # 对应 CSV
LABELS_PATH = os.path.join(MATCH_DIR, "labels", "1_01_00_labels.json")  # 对应 labels
OUTPUT_VIDEO = "prediction_visualization.mp4"
THRESHOLD = 0.4  # 预测阈值，可调整
FPS = 29.97  # 视频 FPS，根据 labels 中的 fps 设置

PREV_WINDOW_NUM = 3
AFTER_WINDOW_NUM = 3

def get_feature_cols(prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, prev_window_num)] + \
                ['x_diff_inv_{}'.format(i) for i in range(1, after_window_num)] + \
                ["x_div_{}".format(i) for i in range(1, after_window_num)]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, prev_window_num)] + \
                    ['y_diff_inv_{}'.format(i) for i in range(1, after_window_num)] + \
                    ["y_div_{}".format(i) for i in range(1, after_window_num)]
    return colnames_x + colnames_y

def to_features(data, prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    eps = 1e-15
    data = data.copy()
    for i in range(1, prev_window_num):
        data.loc[:, 'x_lag_{}'.format(i)] = data['x'].shift(i)
        data.loc[:, 'y_lag_{}'.format(i)] = data['y'].shift(i)
        data.loc[:, 'x_diff_{}'.format(i)] = data['x_lag_{}'.format(i)] - data['x']
        data.loc[:, 'y_diff_{}'.format(i)] = data['y_lag_{}'.format(i)] - data['y']
    for i in range(1, after_window_num):
        data.loc[:, 'x_lag_inv_{}'.format(i)] = data['x'].shift(-i)
        data.loc[:, 'y_lag_inv_{}'.format(i)] = data['y'].shift(-i)
        data.loc[:, 'x_diff_inv_{}'.format(i)] = data['x_lag_inv_{}'.format(i)] - data['x']
        data.loc[:, 'y_diff_inv_{}'.format(i)] = data['y_lag_inv_{}'.format(i)] - data['y']
    for i in range(1, after_window_num):
        data.loc[:, 'x_div_{}'.format(i)] = data['x_diff_{}'.format(i)] / (data['x_diff_inv_{}'.format(i)] + eps)
        data.loc[:, 'y_div_{}'.format(i)] = data['y_diff_{}'.format(i)] / (data['y_diff_inv_{}'.format(i)] + eps)
    for i in range(1, prev_window_num):
        data = data[data['x_lag_{}'.format(i)].notna()]
    for i in range(1, after_window_num):
        data = data[data['x_lag_inv_{}'.format(i)].notna()]
    data = data[data['x'].notna()]
    return data

def main():
    # 检查文件存在
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在。")
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"视频文件 {VIDEO_PATH} 不存在。")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV 文件 {CSV_PATH} 不存在。")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels 文件 {LABELS_PATH} 不存在。")

    # 加载模型
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    # 加载 CSV 数据
    df = pd.read_csv(CSV_PATH)
    df = df[df['Visibility'] == 1]  # 只保留可见帧
    df['timestamp'] = (df['Frame'] / FPS * 1000).astype(int)  # 转换为毫秒
    df = df.rename(columns={'X': 'x', 'Y': 'y'})

    # 转换为 features 并预测
    features_df = to_features(df)
    if len(features_df) == 0:
        raise ValueError("没有足够的特征数据进行预测。")
    features = features_df[get_feature_cols()]
    predictions = model.predict(features)

    # 加载 labels，获取真实击球帧
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
    true_events = {event['frame']: event for event in labels['events'] if event['event_type'] == 'landing'}

    # 打开视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件。")

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (width, height))

    frame_idx = 0
    pred_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = frame_idx + 1  # Frame 从 1 开始

        # 检查是否有球位置
        ball_row = df[df['Frame'] == frame_num]
        if not ball_row.empty:
            x, y = ball_row.iloc[0]['x'], ball_row.iloc[0]['y']
            # 绘制球位置（蓝色圆）
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # 蓝色

            # 检查预测（如果有）
            if pred_idx < len(predictions) and features_df.iloc[pred_idx]['Frame'] == frame_num:
                pred_prob = predictions[pred_idx]
                if pred_prob > THRESHOLD:
                    # 预测击球（红色圆）
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 2)
                    cv2.putText(frame, f'Pred: {pred_prob:.2f}', (int(x) + 15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                pred_idx += 1

        # 检查真实击球
        if frame_num in true_events:
            event = true_events[frame_num]
            ex, ey = event['x'], event['y']
            # 真实击球（绿色圆）
            cv2.circle(frame, (int(ex), int(ey)), 10, (0, 255, 0), 2)
            cv2.putText(frame, 'True Bounce', (int(ex) + 15, int(ey) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 添加帧号
        cv2.putText(frame, f'Frame: {frame_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"可视化视频已保存为 {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
