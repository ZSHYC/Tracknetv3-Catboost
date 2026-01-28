import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from model_refiner import STFNet

WINDOW_SIZE = 11
HALF_WIN = 5
ROI_SIZE = 96
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


def load_fps_for_video(match_dir, video_file):
    labels_dir = os.path.join(match_dir, "labels")
    label_name = video_file.replace(".mp4", "_labels.json")
    label_path = os.path.join(labels_dir, label_name)
    if not os.path.exists(label_path):
        return 30.0
    with open(label_path, "r") as f:
        data = json.load(f)
    return data["metadata"].get("fps", 30.0)


def get_roi(frame, x, y, size):
    h, w, _ = frame.shape
    x, y = int(x), int(y)
    half_size = size // 2

    x1 = x - half_size
    y1 = y - half_size
    x2 = x1 + size
    y2 = y1 + size

    pad_left = 0 if x1 >= 0 else -x1
    pad_top = 0 if y1 >= 0 else -y1
    pad_right = 0 if x2 <= w else x2 - w
    pad_bottom = 0 if y2 <= h else y2 - h

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2)
    crop_y2 = min(h, y2)

    roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        roi = cv2.copyMakeBorder(
            roi, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    if roi.shape[0] != size or roi.shape[1] != size:
        roi = cv2.resize(roi, (size, size))

    return roi


def build_sequence(video_df, frame_to_row, cap, center_frame_idx, fallback_row):
    start_frame = center_frame_idx - HALF_WIN

    seq_images = []
    seq_x, seq_y, seq_vis, seq_pred = [], [], [], []

    for k in range(WINDOW_SIZE):
        current_f_idx = start_frame + k
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_f_idx)
        ret, frame = cap.read()

        current_x, current_y = 0.0, 0.0
        visibility = 0
        pred_score = 0.0

        if current_f_idx in frame_to_row:
            feat_row = frame_to_row[current_f_idx]
            current_x = feat_row['x']
            current_y = feat_row['y']
            visibility = 1
            pred_score = feat_row.get('pred', 0.0)
        else:
            found = False
            for offset in [-2, -1, 1, 2]:
                nearby_idx = current_f_idx + offset
                if nearby_idx in frame_to_row:
                    feat_row = frame_to_row[nearby_idx]
                    current_x = feat_row['x']
                    current_y = feat_row['y']
                    visibility = 1
                    pred_score = feat_row.get('pred', 0.0)
                    found = True
                    break
            if not found:
                current_x = fallback_row['x']
                current_y = fallback_row['y']
                visibility = 0
                pred_score = fallback_row.get('pred', 0.0)

        if not ret:
            img_roi = np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
        else:
            img_roi = get_roi(frame, current_x, current_y, ROI_SIZE)

        seq_images.append(img_roi)
        seq_x.append(current_x)
        seq_y.append(current_y)
        seq_vis.append(visibility)
        seq_pred.append(pred_score)

    seq_x = np.array(seq_x, dtype=np.float32)
    seq_y = np.array(seq_y, dtype=np.float32)
    seq_vis = np.array(seq_vis, dtype=np.float32)
    seq_pred = np.array(seq_pred, dtype=np.float32)

    dx = np.diff(seq_x, prepend=seq_x[0])
    dy = np.diff(seq_y, prepend=seq_y[0])
    ddx = np.diff(dx, prepend=dx[0])
    ddy = np.diff(dy, prepend=dy[0])

    norm_x = seq_x / VIDEO_WIDTH
    norm_y = seq_y / VIDEO_HEIGHT
    norm_dx = dx / VIDEO_WIDTH
    norm_dy = dy / VIDEO_HEIGHT
    norm_ddx = ddx / VIDEO_WIDTH
    norm_ddy = ddy / VIDEO_HEIGHT

    seq_vectors = []
    for i in range(WINDOW_SIZE):
        seq_vectors.append([
            norm_x[i], norm_y[i],
            norm_dx[i], norm_dy[i],
            norm_ddx[i], norm_ddy[i],
            seq_vis[i], seq_pred[i]
        ])

    images_stack = np.array(seq_images, dtype=np.uint8)
    vectors_stack = np.array(seq_vectors, dtype=np.float32)

    return images_stack, vectors_stack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='predict.csv', help='predict.csv path')
    parser.add_argument('--candidates', type=str, default=None, help='optional predicted_bounces.csv path')
    parser.add_argument('--model', type=str, default='checkpoints/best_refiner.pth', help='model path')
    parser.add_argument('--output', type=str, default='refined_bounces.csv', help='output csv')
    parser.add_argument('--threshold', type=float, default=0.95, help='refiner threshold')
    parser.add_argument('--candidate-threshold', type=float, default=0.4, help='candidate filter on predict.csv pred when no candidates file')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return

    df = pd.read_csv(args.input)
    if args.candidates and os.path.exists(args.candidates):
        cand_df = pd.read_csv(args.candidates)
        df = df.merge(cand_df[['timestamp', 'source_video']], on=['timestamp', 'source_video'], how='inner')
        print(f"Using candidates file: {args.candidates}, candidates={len(df)}")
    else:
        # 如果没有提供 candidates，则按预测分数过滤候选
        if 'pred' in df.columns:
            before = len(df)
            df = df[df['pred'] >= args.candidate_threshold].copy()
            print(f"Filtered candidates by pred >= {args.candidate_threshold}: {before} -> {len(df)}")

    if len(df) == 0:
        print("No candidates found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STFNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    results = []

    # 按视频分组推理
    for video_path, group in df.groupby('source_video'):
        # source_video: data/test/matchX/video/xxx.mp4
        video_path = video_path.replace('\\', '/')
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        # 推断 match_dir
        # data/test/matchX/video/xxx.mp4
        parts = video_path.split('/')
        match_dir = '/'.join(parts[:-2])
        video_file = parts[-1]

        fps = load_fps_for_video(match_dir, video_file)

        video_df = df[df['source_video'] == video_path].copy()
        video_df['frame_idx'] = (video_df['timestamp'] * fps / 1000.0).round().astype(int)
        frame_to_row = video_df.drop_duplicates('frame_idx').set_index('frame_idx').to_dict('index')

        cap = cv2.VideoCapture(video_path)

        total_candidates = len(group)
        print(f"Processing {video_path} | candidates: {total_candidates}")

        batch_images = []
        batch_vectors = []
        batch_meta = []

        for i, (_, row) in enumerate(group.iterrows(), start=1):
            center_frame_idx = round(row['timestamp'] * fps / 1000)
            images_stack, vectors_stack = build_sequence(video_df, frame_to_row, cap, center_frame_idx, row)
            batch_images.append(images_stack)
            batch_vectors.append(vectors_stack)
            batch_meta.append(row)

            if i % 200 == 0:
                print(f"  progress: {i}/{total_candidates}")

            # 简单batch推理
            if len(batch_images) >= 16:
                preds = run_batch(model, batch_images, batch_vectors, device)
                for pred, meta in zip(preds, batch_meta):
                    if pred >= args.threshold:
                        results.append({
                            'timestamp': meta['timestamp'],
                            'x': meta['x'],
                            'y': meta['y'],
                            'pred': pred,
                            'source_video': meta['source_video']
                        })
                batch_images, batch_vectors, batch_meta = [], [], []

        # flush remaining
        if batch_images:
            preds = run_batch(model, batch_images, batch_vectors, device)
            for pred, meta in zip(preds, batch_meta):
                if pred >= args.threshold:
                    results.append({
                        'timestamp': meta['timestamp'],
                        'x': meta['x'],
                        'y': meta['y'],
                        'pred': pred,
                        'source_video': meta['source_video']
                    })

        cap.release()

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False)
        print(f"Saved refined results: {args.output}, count={len(out_df)}")
    else:
        print("No refined results above threshold.")


def run_batch(model, batch_images, batch_vectors, device):
    imgs = np.array(batch_images, dtype=np.float32) / 255.0
    imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
    vecs = np.array(batch_vectors, dtype=np.float32)

    imgs = torch.tensor(imgs).to(device)
    vecs = torch.tensor(vecs).to(device)

    with torch.no_grad():
        logits = model(imgs, vecs)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs


if __name__ == '__main__':
    main()
