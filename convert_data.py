import pandas as pd
import json
import os

def convert_to_json_lines(csv_path, labels_path, output_path, track_id=1):
    # 读取CSV
    df = pd.read_csv(csv_path)  # 列: Frame, Visibility, X, Y
    df = df[df['Visibility'] == 1]  # 只保留可见帧
    
    # 读取labels
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    fps = labels['metadata']['fps']
    events = {event['frame']: event for event in labels['events']}
    
    # 生成JSON Lines
    with open(output_path, 'w') as f:
        for _, row in df.iterrows():
            frame = int(row['Frame'])
            timestamp = int(frame / fps * 1000)  # 毫秒
            x, y = row['X'], row['Y']
            event_cls = 1 if frame in events and events[frame]['event_type'] == 'landing' else 0
            data = {
                "timestamp": timestamp,
                "x": x,
                "y": y,
                "event_cls": event_cls,
                "track_id": track_id
            }
            f.write(json.dumps(data) + '\n')

def batch_convert(base_dir='data/train'):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"目录 {base_dir} 不存在。")
    
    for match_dir in os.listdir(base_dir):
        match_path = os.path.join(base_dir, match_dir)
        if not os.path.isdir(match_path):
            continue
        
        csv_dir = os.path.join(match_path, 'csv')
        labels_dir = os.path.join(match_path, 'labels')
        output_path = os.path.join(match_path, 'bounce_train.json')
        
        if not os.path.exists(csv_dir) or not os.path.exists(labels_dir):
            print(f"跳过 {match_dir}：缺少csv或labels文件夹。")
            continue
        
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_ball.csv')]
        labels_files = [f for f in os.listdir(labels_dir) if f.endswith('_labels.json')]
        
        with open(output_path, 'w') as out_f:
            track_id = 1
            for csv_file in csv_files:
                # 匹配labels文件：1_01_00_ball.csv -> 1_01_00_labels.json
                base_name = csv_file.replace('_ball.csv', '')
                labels_file = base_name + '_labels.json'
                
                if labels_file not in labels_files:
                    print(f"跳过 {match_dir}/{csv_file}：无匹配labels文件。")
                    continue
                
                csv_path = os.path.join(csv_dir, csv_file)
                labels_path = os.path.join(labels_dir, labels_file)
                
                # 读取并写入
                df = pd.read_csv(csv_path)
                df = df[df['Visibility'] == 1]
                
                with open(labels_path, 'r') as f:
                    labels = json.load(f)
                fps = labels['metadata']['fps']
                events = {event['frame']: event for event in labels['events']}
                
                for _, row in df.iterrows():
                    frame = int(row['Frame'])
                    timestamp = int(frame / fps * 1000)
                    x, y = float(row['X']), float(row['Y'])
                    event_cls = int(1 if frame in events and events[frame]['event_type'] == 'landing' else 0)
                    data = {
                        "timestamp": timestamp,
                        "pos": {"x": x, "y": y},
                        "event_cls": event_cls,
                        "track_id": track_id
                    }
                    out_f.write(json.dumps(data) + '\n')
                
                track_id += 1  # 每个回合不同track_id
        
        print(f"转换完成：{match_dir} -> {output_path}")

# 批量转换
batch_convert()