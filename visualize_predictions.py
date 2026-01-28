import cv2
import pandas as pd
import json
import os
import numpy as np
from catboost import CatBoostRegressor

# 参数设置
MODEL_PATH = "stroke_model.cbm"
TEST_DIR = "data/test"  # 测试集根目录
OUTPUT_DIR = "predictions"  # 输出文件夹
THRESHOLD = 0.4  # 预测阈值
FPS = 29.97  # 默认 FPS，可根据 labels 调整

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

def process_video(match_dir, video_file):
    video_name = os.path.splitext(video_file)[0]  # 如 1_05_02
    video_path = os.path.join(match_dir, "video", video_file)
    csv_path = os.path.join(match_dir, "csv", f"{video_name}_ball.csv")
    labels_path = os.path.join(match_dir, "labels", f"{video_name}_labels.json")
    output_video = os.path.join(OUTPUT_DIR, f"prediction_{os.path.basename(match_dir)}_{video_name}.mp4")

    # 检查文件
    if not os.path.exists(csv_path):
        print(f"跳过 {video_name}：CSV 文件不存在")
        return
    if not os.path.exists(labels_path):
        print(f"跳过 {video_name}：Labels 文件不存在")
        return

    # 加载模型（每次都加载，简单起见）
    model = CatBoostRegressor()
    model.load_model(MODEL_PATH)

    # 加载 CSV
    df = pd.read_csv(csv_path)
    df = df[df['Visibility'] == 1]
    df['timestamp'] = (df['Frame'] / FPS * 1000).astype(int)
    df = df.rename(columns={'X': 'x', 'Y': 'y'})

    # 转换为 features 并预测
    features_df = to_features(df)
    if len(features_df) == 0:
        print(f"跳过 {video_name}：无足够特征")
        return
    features = features_df[get_feature_cols()]
    predictions = model.predict(features)

    # 加载 labels
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    true_events = {event['frame']: event for event in labels['events'] if event['event_type'] == 'landing'}
    fps = labels['metadata'].get('fps', FPS)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"跳过 {video_name}：无法打开视频")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_idx = 0
    pred_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = frame_idx + 1

        ball_row = df[df['Frame'] == frame_num]
        if not ball_row.empty:
            x, y = ball_row.iloc[0]['x'], ball_row.iloc[0]['y']
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # 蓝色球

            if pred_idx < len(predictions) and features_df.iloc[pred_idx]['Frame'] == frame_num:
                pred_prob = predictions[pred_idx]
                if pred_prob > THRESHOLD:
                    cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), 2)
                    cv2.putText(frame, f'Pred: {pred_prob:.2f}', (int(x) + 15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                pred_idx += 1

        if frame_num in true_events:
            event = true_events[frame_num]
            ex, ey = event['x'], event['y']
            cv2.circle(frame, (int(ex), int(ey)), 10, (0, 255, 0), 2)
            cv2.putText(frame, 'True Bounce', (int(ex) + 15, int(ey) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f'Frame: {frame_num}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"生成 {output_video}")

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在。")

    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 遍历所有 match 目录
    for match in os.listdir(TEST_DIR):
        match_path = os.path.join(TEST_DIR, match)
        if not os.path.isdir(match_path) or not match.startswith("match"):
            continue

        video_dir = os.path.join(match_path, "video")
        if not os.path.exists(video_dir):
            continue

        # 遍历视频文件
        for video_file in os.listdir(video_dir):
            if video_file.endswith(".mp4"):
                process_video(match_path, video_file)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
