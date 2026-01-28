import os
import json
import argparse
import pandas as pd
import cv2

OUTPUT_DIR_DEFAULT = "refined_visualizations"


def load_labels(match_dir, video_file):
    labels_dir = os.path.join(match_dir, "labels")
    label_name = video_file.replace(".mp4", "_labels.json")
    label_path = os.path.join(labels_dir, label_name)
    if not os.path.exists(label_path):
        return None
    with open(label_path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_pred_frame_map(df, fps):
    pred_map = {}
    for _, row in df.iterrows():
        frame_idx = round(row['timestamp'] * fps / 1000.0)
        frame_num = frame_idx + 1
        pred_map.setdefault(frame_num, []).append(row)
    return pred_map


def process_video(video_path, match_dir, video_file, preds_df, output_dir, limit_frames=None):
    labels = load_labels(match_dir, video_file)
    fps = labels['metadata'].get('fps', 30.0) if labels else 30.0
    true_events = {}
    if labels:
        for event in labels.get('events', []):
            if event.get('event_type') == 'landing':
                true_events[event['frame']] = event

    pred_map = build_pred_frame_map(preds_df, fps)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, f"refined_{os.path.basename(match_dir)}_{os.path.splitext(video_file)[0]}.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        if limit_frames and frame_idx >= limit_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_num = frame_idx + 1

        # 绘制 refined 预测
        if frame_num in pred_map:
            for row in pred_map[frame_num]:
                x, y = int(row['x']), int(row['y'])
                cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)
                cv2.putText(frame, f"Refined: {row['pred']:.2f}", (x + 12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 绘制真实落点
        if frame_num in true_events:
            ev = true_events[frame_num]
            ex, ey = int(ev['x']), int(ev['y'])
            cv2.circle(frame, (ex, ey), 10, (0, 255, 0), 2)
            cv2.putText(frame, "True Bounce", (ex + 12, ey + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"输出视频: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='refined_bounces.csv', help='refined_bounces.csv path')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR_DEFAULT, help='output video directory')
    parser.add_argument('--limit-frames', type=int, default=None, help='optional max frames for quick preview')
    parser.add_argument('--only-video', type=str, default=None, help='only process a single video file name (e.g. 1_05_02.mp4)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"输入文件不存在: {args.input}")
        return

    df = pd.read_csv(args.input)
    if len(df) == 0:
        print("refined_bounces.csv 为空")
        return

    ensure_dir(args.output_dir)

    for video_path, group in df.groupby('source_video'):
        video_path = video_path.replace('\\', '/')
        if not os.path.exists(video_path):
            print(f"视频不存在: {video_path}")
            continue

        video_file = os.path.basename(video_path)
        if args.only_video and video_file != args.only_video:
            continue

        parts = video_path.split('/')
        match_dir = '/'.join(parts[:-2])

        process_video(video_path, match_dir, video_file, group, args.output_dir, args.limit_frames)


if __name__ == '__main__':
    main()
