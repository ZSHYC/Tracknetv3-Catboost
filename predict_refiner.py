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


def build_sequence_from_cache(frame_to_row, frames_cache, center_frame_idx, fallback_row):
    start_frame = center_frame_idx - HALF_WIN

    seq_images = []
    seq_x, seq_y, seq_vis, seq_pred = [], [], [], []

    for k in range(WINDOW_SIZE):
        current_f_idx = start_frame + k
        frame = frames_cache.get(current_f_idx, None)

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

        if frame is None:
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


def build_frame_cache(video_path, required_frames):
    frames_cache = {}
    if not required_frames:
        return frames_cache

    max_frame = max(required_frames)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames_cache

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in required_frames:
            frames_cache[frame_idx] = frame
        if frame_idx > max_frame:
            break
        frame_idx += 1

    cap.release()
    return frames_cache


def run_refiner(input_path, candidates_path, model_path, output_path, threshold, candidate_threshold):
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return 0
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return 0

    df = pd.read_csv(input_path)
    if candidates_path and os.path.exists(candidates_path):
        cand_df = pd.read_csv(candidates_path)
        df = df.merge(cand_df[['timestamp', 'source_video']], on=['timestamp', 'source_video'], how='inner')
        print(f"Using candidates file: {candidates_path}, candidates={len(df)}")
    else:
        if 'pred' in df.columns:
            before = len(df)
            df = df[df['pred'] >= candidate_threshold].copy()
            print(f"Filtered candidates by pred >= {candidate_threshold}: {before} -> {len(df)}")

    if len(df) == 0:
        print("No candidates found.")
        return 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STFNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    for video_path, group in df.groupby('source_video'):
        video_path = video_path.replace('\\', '/')
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        parts = video_path.split('/')
        match_dir = '/'.join(parts[:-2])
        video_file = parts[-1]
        fps = load_fps_for_video(match_dir, video_file)

        video_df = df[df['source_video'] == video_path].copy()
        video_df['frame_idx'] = (video_df['timestamp'] * fps / 1000.0).round().astype(int)
        frame_to_row = video_df.drop_duplicates('frame_idx').set_index('frame_idx').to_dict('index')

        total_candidates = len(group)
        print(f"Processing {video_path} | candidates: {total_candidates}")

        required_frames = set()
        for _, row in group.iterrows():
            center_frame_idx = round(row['timestamp'] * fps / 1000)
            for f in range(center_frame_idx - HALF_WIN, center_frame_idx + HALF_WIN + 1):
                if f >= 0:
                    required_frames.add(f)

        frames_cache = build_frame_cache(video_path, required_frames)

        batch_images = []
        batch_vectors = []
        batch_meta = []

        for i, (_, row) in enumerate(group.iterrows(), start=1):
            center_frame_idx = round(row['timestamp'] * fps / 1000)
            images_stack, vectors_stack = build_sequence_from_cache(
                frame_to_row, frames_cache, center_frame_idx, row
            )
            batch_images.append(images_stack)
            batch_vectors.append(vectors_stack)
            batch_meta.append(row)

            if i % 200 == 0:
                print(f"  progress: {i}/{total_candidates}")

            if len(batch_images) >= 16:
                preds = run_batch(model, batch_images, batch_vectors, device)
                for pred, meta in zip(preds, batch_meta):
                    if pred >= threshold:
                        results.append({
                            'timestamp': meta['timestamp'],
                            'x': meta['x'],
                            'y': meta['y'],
                            'pred': pred,
                            'source_video': meta['source_video']
                        })
                batch_images, batch_vectors, batch_meta = [], [], []

        if batch_images:
            preds = run_batch(model, batch_images, batch_vectors, device)
            for pred, meta in zip(preds, batch_meta):
                if pred >= threshold:
                    results.append({
                        'timestamp': meta['timestamp'],
                        'x': meta['x'],
                        'y': meta['y'],
                        'pred': pred,
                        'source_video': meta['source_video']
                    })

    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(output_path, index=False)
        print(f"Saved refined results: {output_path}, count={len(out_df)}")
        return len(out_df)
    else:
        print("No refined results above threshold.")
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='predict.csv', help='predict.csv path')
    parser.add_argument('--candidates', type=str, default=None, help='optional predicted_bounces.csv path')
    parser.add_argument('--model', type=str, default='checkpoints/best_refiner.pth', help='model path')
    parser.add_argument('--output', type=str, default='refined_bounces.csv', help='output csv')
    parser.add_argument('--threshold', type=float, default=0.95, help='refiner threshold')
    parser.add_argument('--candidate-threshold', type=float, default=0.4, help='candidate filter on predict.csv pred when no candidates file')
    args = parser.parse_args()

    run_refiner(
        input_path=args.input,
        candidates_path=args.candidates,
        model_path=args.model,
        output_path=args.output,
        threshold=args.threshold,
        candidate_threshold=args.candidate_threshold
    )


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
