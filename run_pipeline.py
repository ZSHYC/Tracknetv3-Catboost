import os
import argparse

from stroke_model import predict
from predict_refiner import run_refiner
from visualize_predictions import run_visualization


def read_best_threshold(default_val=0.95):
    path = os.path.join("checkpoints", "best_refiner_threshold.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except Exception:
            return default_val
    return default_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-catboost', action='store_true', help='skip generating predict.csv via CatBoost')
    parser.add_argument('--skip-refiner', action='store_true', help='skip refiner inference')
    parser.add_argument('--skip-visualize', action='store_true', help='skip visualization')
    parser.add_argument('--threshold', type=float, default=None, help='refiner threshold override')
    parser.add_argument('--candidate-threshold', type=float, default=0.4, help='candidate filter on predict.csv pred when no candidates file')
    parser.add_argument('--use-candidates', action='store_true', help='use predicted_bounces.csv as candidates')
    parser.add_argument('--limit-frames', type=int, default=None, help='optional max frames for quick preview')
    parser.add_argument('--only-video', type=str, default=None, help='only process a single video file name (e.g. 1_05_02.mp4)')
    args = parser.parse_args()

    if not args.skip_catboost:
        print("[1/3] Generating predict.csv and predicted_bounces.csv...")
        predict()

    refined_path = "refined_bounces.csv"

    if not args.skip_refiner:
        print("[2/3] Running refiner inference...")
        threshold = args.threshold if args.threshold is not None else read_best_threshold()
        candidates_path = "predicted_bounces.csv" if args.use_candidates and os.path.exists("predicted_bounces.csv") else None
        run_refiner(
            input_path="predict.csv",
            candidates_path=candidates_path,
            model_path=os.path.join("checkpoints", "best_refiner.pth"),
            output_path=refined_path,
            threshold=threshold,
            candidate_threshold=args.candidate_threshold
        )

    if not args.skip_visualize:
        print("[3/3] Rendering visualization videos...")
        run_visualization(
            input_path=refined_path,
            output_dir="refined_visualizations",
            limit_frames=args.limit_frames,
            only_video=args.only_video
        )


if __name__ == '__main__':
    main()
