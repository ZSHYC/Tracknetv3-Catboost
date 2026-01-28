import os
import json
import numpy as np
import pandas as pd
import cv2
from catboost import CatBoostRegressor
from stroke_model import to_features, get_feature_cols, PREV_WINDOW_NUM, AFTER_WINDOW_NUM

# 配置参数
WINDOW_SIZE = 11  # 总窗口长度
HALF_WIN = 5      # 前后各取5帧
ROI_SIZE = 96     # 裁剪图像大小
OUTPUT_DIR = "dataset_v2/train" # 输出目录

# 原始视频分辨率（用于归一化几何特征）
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_fps_map(match_dir):
    """
    加载该Match下所有视频的FPS信息
    返回: dict { 'xxxx.mp4': fps_value }
    """
    fps_map = {}
    labels_dir = os.path.join(match_dir, "labels")
    if not os.path.exists(labels_dir):
        return fps_map
    
    for filename in os.listdir(labels_dir):
        if filename.endswith("_labels.json"):
            video_name = filename.replace("_labels.json", ".mp4")
            with open(os.path.join(labels_dir, filename), 'r') as f:
                data = json.load(f)
                fps = data['metadata'].get('fps', 30.0) # 默认30
                fps_map[video_name] = fps
    return fps_map

def get_roi(frame, x, y, size):
    """
    裁剪ROI，处理边界Padding
    """
    h, w, _ = frame.shape
    x, y = int(x), int(y)
    half_size = size // 2
    
    # 计算裁剪区域
    x1 = x - half_size
    y1 = y - half_size
    x2 = x1 + size
    y2 = y1 + size
    
    # 填充处理
    pad_left = 0 if x1 >= 0 else -x1
    pad_top = 0 if y1 >= 0 else -y1
    pad_right = 0 if x2 <= w else x2 - w
    pad_bottom = 0 if y2 <= h else y2 - h
    
    # 修正实际裁剪坐标
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w, x2)
    crop_y2 = min(h, y2)
    
    roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 如果需要padding
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
    # 确保尺寸精确（处理奇偶像素误差）
    if roi.shape[0] != size or roi.shape[1] != size:
         roi = cv2.resize(roi, (size, size))
         
    return roi

def process_match(match_dir, model, output_path):
    print(f"Processing {match_dir}...")
    
    # 1. 加载数据
    bounce_file = os.path.join(match_dir, "bounce_train.json")
    if not os.path.exists(bounce_file):
        print(f"Skipping {match_dir}, no bounce_train.json")
        return

    # 读取原始数据
    datalist = [json.loads(line.strip()) for line in open(bounce_file, "r").readlines()]
    if not datalist:
        return

    raw_df = pd.DataFrame(datalist)
    if 'pos' in raw_df.columns: # 处理嵌套json
         raw_df['x'] = raw_df['pos'].apply(lambda p: p['x'])
         raw_df['y'] = raw_df['pos'].apply(lambda p: p['y'])
    
    # 确保按视频和时间排序，这对于序列提取至关重要
    if 'video_file' in raw_df.columns:
        raw_df = raw_df.sort_values(by=['video_file', 'timestamp']).reset_index(drop=True)
    else:
        raw_df = raw_df.sort_values(by=['timestamp']).reset_index(drop=True)
    
    # 2. 生成CatBoost特征并预测 (为了Mining)
    # 这是一个临时的df，用于计算特征
    feat_df = to_features(raw_df) 
    
    # 预测
    feat_cols = get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)
    try:
        preds = model.predict(feat_df[feat_cols])
    except Exception as e:
        print(f"Prediction failed for {match_dir}: {e}")
        return
        
    feat_df['pred_score'] = preds
    
    # 将预测结果合并回raw_df (to_features可能会因为window和dropna过滤掉头尾行)
    # 我们通过index合并
    raw_df['pred_score'] = 0.0 # 默认0
    raw_df.loc[feat_df.index, 'pred_score'] = feat_df['pred_score']
    
    # 3. 筛选样本索引
    # 正样本
    pos_indices = raw_df[raw_df['event_cls'] == 1].index.tolist()
    
    # 困难负样本 (CatBoost 误判为落点 > 0.1 但实际不是)
    hard_neg_indices = raw_df[(raw_df['event_cls'] == 0) & (raw_df['pred_score'] > 0.1)].index.tolist()
    
    # 简单负样本 (随机采10%)
    easy_neg_indices = raw_df[(raw_df['event_cls'] == 0) & (raw_df['pred_score'] <= 0.1)].sample(frac=0.1, random_state=42).index.tolist()
    
    combined_indices = sorted(list(set(pos_indices + hard_neg_indices + easy_neg_indices)))
    
    if not combined_indices:
        return

    # 4. 提取序列数据
    fps_map = load_fps_map(match_dir)
    images_list = []
    vector_list = []
    labels_list = []
    infos_list = []
    
    # 分组处理以减少视频打开次数
    grouped = raw_df.loc[combined_indices].groupby('video_file')
    
    for video_file, group in grouped:
        video_path = os.path.join(match_dir, "video", video_file)
        if not os.path.exists(video_path):
            print(f"Warning: Video not found {video_path}")
            continue
            
        fps = fps_map.get(video_file, 30.0)
        cap = cv2.VideoCapture(video_path)
        
        # 获取该视频的所有原始数据，以便查询上下文坐标
        video_full_df = raw_df[raw_df['video_file'] == video_file].copy()
        
        # 为每一行计算精确帧号（由timestamp反推）
        # timestamp 是毫秒，fps 来自 labels json
        video_full_df['frame_idx'] = (video_full_df['timestamp'] * fps / 1000.0).round().astype(int)
        
        # 构建 frame_idx -> row 映射，便于精准对齐
        # 注意：若同一帧多条记录，保留第一条
        frame_to_row = video_full_df.drop_duplicates('frame_idx').set_index('frame_idx').to_dict('index')
        frame_indices = video_full_df['frame_idx'].values
        
        for idx, row in group.iterrows():
            timestamp = row['timestamp']
            center_frame_idx = round(timestamp * fps / 1000)
            
            # 判断目标序列范围
            start_frame = center_frame_idx - HALF_WIN
            
            # 当前样本的容器
            sample_images = []
            sample_vectors = []
            
            valid_sample = True
            
            seq_x = []
            seq_y = []
            seq_vis = []
            seq_pred = []

            for k in range(WINDOW_SIZE):
                current_f_idx = start_frame + k
                
                # --- 提取图像 ---
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_f_idx)
                ret, frame = cap.read()
                
                # --- 精确帧对齐 ---
                current_x, current_y = 0.0, 0.0
                visibility = 0
                pred_score = 0.0
                
                if current_f_idx in frame_to_row:
                    feat_row = frame_to_row[current_f_idx]
                    current_x = feat_row['x']
                    current_y = feat_row['y']
                    visibility = 1
                    pred_score = feat_row.get('pred_score', 0.0)
                else:
                    # 尝试在邻近帧内寻找 (±2 帧)
                    found = False
                    for offset in [-2, -1, 1, 2]:
                        nearby_idx = current_f_idx + offset
                        if nearby_idx in frame_to_row:
                            feat_row = frame_to_row[nearby_idx]
                            current_x = feat_row['x']
                            current_y = feat_row['y']
                            visibility = 1
                            pred_score = feat_row.get('pred_score', 0.0)
                            found = True
                            break
                    if not found:
                        # 用中心帧坐标兜底
                        current_x = row['x']
                        current_y = row['y']
                        visibility = 0
                        pred_score = row.get('pred_score', 0.0)
                
                if not ret:
                    img_roi = np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
                else:
                    img_roi = get_roi(frame, current_x, current_y, ROI_SIZE)
                
                sample_images.append(img_roi)
                seq_x.append(current_x)
                seq_y.append(current_y)
                seq_vis.append(visibility)
                seq_pred.append(pred_score)

            # 计算速度与加速度特征
            seq_x = np.array(seq_x, dtype=np.float32)
            seq_y = np.array(seq_y, dtype=np.float32)
            seq_vis = np.array(seq_vis, dtype=np.float32)
            seq_pred = np.array(seq_pred, dtype=np.float32)
            
            dx = np.diff(seq_x, prepend=seq_x[0])
            dy = np.diff(seq_y, prepend=seq_y[0])
            ddx = np.diff(dx, prepend=dx[0])
            ddy = np.diff(dy, prepend=dy[0])
            
            # 归一化
            norm_x = seq_x / VIDEO_WIDTH
            norm_y = seq_y / VIDEO_HEIGHT
            norm_dx = dx / VIDEO_WIDTH
            norm_dy = dy / VIDEO_HEIGHT
            norm_ddx = ddx / VIDEO_WIDTH
            norm_ddy = ddy / VIDEO_HEIGHT
            
            # 组合几何向量: [x, y, dx, dy, ddx, ddy, visibility, pred_score]
            for i in range(WINDOW_SIZE):
                sample_vectors.append([
                    norm_x[i], norm_y[i],
                    norm_dx[i], norm_dy[i],
                    norm_ddx[i], norm_ddy[i],
                    seq_vis[i], seq_pred[i]
                ])

            # 堆叠
            images_stack = np.array(sample_images, dtype=np.uint8) # (11, 96, 96, 3)
            vectors_stack = np.array(sample_vectors, dtype=np.float32) # (11, 8)
            
            images_list.append(images_stack)
            vector_list.append(vectors_stack)
            labels_list.append(row['event_cls'])
            
            # 记录样本类型
            s_type = "pos" if row['event_cls'] == 1 else ("hard_neg" if row['pred_score'] > 0.1 else "easy_neg")
            infos_list.append([video_file, str(timestamp), s_type])

        cap.release()

    if len(images_list) > 0:
        # 保存为NPZ
        final_images = np.array(images_list)
        final_vectors = np.array(vector_list)
        final_labels = np.array(labels_list).reshape(-1, 1)
        final_infos = np.array(infos_list)
        
        np.savez_compressed(output_path, 
                            images=final_images, 
                            geo_vectors=final_vectors, 
                            labels=final_labels,
                            infos=final_infos)
        print(f"Saved {match_dir} samples: {len(final_labels)}")
    else:
        print(f"No samples extracted for {match_dir}")

def main():
    ensure_dir(OUTPUT_DIR)
    
    # 加载现有模型
    print("Loading CatBoost model...")
    model = CatBoostRegressor()
    model.load_model("stroke_model.cbm")
    
    train_root = "data/train"
    match_dirs = [os.path.join(train_root, d) for d in os.listdir(train_root) if d.startswith("match")]
    
    for m_dir in match_dirs:
        m_name = os.path.basename(m_dir)
        output_path = os.path.join(OUTPUT_DIR, f"{m_name}.npz")
        
        # if os.path.exists(output_path):
        #     print(f"Skipping {m_name}, already exists.")
        #     continue
            
        process_match(m_dir, model, output_path)
    
    print("All done.")

if __name__ == "__main__":
    main()
