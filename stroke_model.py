import pandas as pd
import json
import numpy as np
import os
from catboost import CatBoostRegressor


TRAIN_DIR = "data/train"
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"训练数据目录 {TRAIN_DIR} 不存在，请检查路径。")

TEST_DIR = "data/test"
if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"测试数据目录 {TEST_DIR} 不存在，请检查路径。")

PREV_WINDOW_NUM = 3
AFTER_WINDOW_NUM = 3

def get_feature_cols(prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, prev_window_num)] + \
                ['x_diff_inv_{}'.format(i) for i in range(1, after_window_num)] + \
                ["x_div_{}".format(i) for i in range(1, after_window_num)] #+ \
                #["x"]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, prev_window_num)] + \
                    ['y_diff_inv_{}'.format(i) for i in range(1, after_window_num)] + \
                    ["y_div_{}".format(i) for i in range(1, after_window_num)] #+ \
                    # ["y"]
    colnames = colnames_x + colnames_y #+ ["coord"]
    return colnames

def to_features(data, prev_window_num=PREV_WINDOW_NUM, after_window_num=AFTER_WINDOW_NUM):
    eps = 1e-15
    data = data.copy()  # Create a copy of the DataFrame to avoid SettingWithCopyWarning
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
        data.loc[:, 'x_div_{}'.format(i)] = data['x_diff_{}'.format(i)]/(data['x_diff_inv_{}'.format(i)] + eps)
        data.loc[:, 'y_div_{}'.format(i)] = data['y_diff_{}'.format(i)]/(data['y_diff_inv_{}'.format(i)] + eps)

    for i in range(1, prev_window_num):
        data = data[data['x_lag_{}'.format(i)].notna()]
        
    for i in range(1, after_window_num):
        data = data[data['x_lag_inv_{}'.format(i)].notna()]
    data = data[data['x'].notna()] 
    return data

def __add_weight(pd_data, weight_map):
    pd_data["weight"] = pd_data["event_cls"].map(weight_map)
    return pd_data


def __convert_to_dataframe(data, labels_data=[]):
    pd_data = []
    for index, item in enumerate(data):
        item_timestamp = item["timestamp"]
        if item_timestamp in labels_data:
            label = 1
        else:
            label = 0
        label = max(item.get("event_cls", 0), item.get("label_cls", 0), label)
        if item.get("pos", None) is None:
            next_index = -1
            for i in range(index + 1, index + 5):
                if i >= len(data):
                    break
                if data[i].get("pos", None) is not None:
                    next_index = i
                    break
            if next_index == -1:
                continue
            last_data = pd_data[-1]
            next_item = data[next_index]

            x = (last_data["x"] + next_item["pos"]["x"]) / (next_index - index + 1)
            y = (last_data["y"] + next_item["pos"]["y"]) / (next_index - index + 1)
            # if y < 200:  # 只考虑近处的摄像头
            #     label = 0
            pd_data.append({
                "timestamp": item["timestamp"],
                "x": x,
                "y": y,
                "event_cls": label,
                "coord": 0,  # inserted
                "video_file": item.get("video_file", "")
            })
        else:
            y = item["pos"]["y"]
            # if y < 200:  # 只考虑近处的摄像头
            #     label = 0
            pd_data.append({
                "timestamp": item["timestamp"],
                "x": item["pos"]["x"],
                "y": item["pos"]["y"],
                "event_cls": label,
                "coord": 1,  # real
                "video_file": item.get("video_file", "")
            })
    if len(pd_data) > 0:
        pd_data = pd.DataFrame.from_dict(pd_data)
        pd_data = __add_weight(pd_data, {1: 400, 0: 1})
    return pd_data

def load_data(directories, tag="left", single_view=False):   # 是否是单视角，如果以后有多视角，设single_view=False并取消注释。
    for directory in directories:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录 {directory} 不存在。")
    resdf = pd.DataFrame()
    for directory in directories:
        if single_view:
            # 单视角：加载 bounce_train.json
            file_path = os.path.join(directory, "bounce_train.json")
        else:
            # 多视角：加载 {tag}_bounce_train.json
            file_path = os.path.join(directory, f"{tag}_bounce_train.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在。")
        datalist = [json.loads(line.strip()) for line in open(file_path, "r").readlines()]
        tracks_data = {}
        for item in datalist:
            track_id = item["track_id"]
            if track_id not in tracks_data:
                tracks_data[track_id] = []
            tracks_data[track_id].append(item)
        for track_id, track_data in tracks_data.items():
            track_data = sorted(track_data, key=lambda x: x["timestamp"])
            tmp = __convert_to_dataframe(track_data)
            if len(tmp) > 0:
                video_file = tmp["video_file"].iloc[0] if len(tmp) > 0 and "video_file" in tmp.columns else ""
                tmp["source_video"] = os.path.join(directory, "video", video_file).replace("\\", "/")  # 统一路径分隔符
                resdf = pd.concat([resdf, to_features(tmp)], ignore_index=True)
    resdf = resdf.sample(frac=1, random_state=42).reset_index(drop=True)
    return resdf

def find_nearest_timestamp(timestamp, timestamps):
    min_diff = float('inf')
    nearest_timestamp = None
    for t in timestamps:
        diff = abs(t - timestamp)
        if diff < min_diff:
            min_diff = diff
            nearest_timestamp = t
    return nearest_timestamp





def train(train_data, test_data):
    if train_data["event_cls"].nunique() < 2:
        raise ValueError("训练集中只有单一类别（event_cls 全为同一值）。请检查 bounce_train.json 是否包含正样本，或重新生成标注数据。")
    catboost_regressor = CatBoostRegressor(iterations=3000, depth=3, learning_rate=0.1, loss_function='RMSE')
    catboost_regressor.fit(train_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)], train_data['event_cls'],
                        eval_set=(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)], test_data['event_cls']),
                        use_best_model=True, sample_weight=train_data['weight'],
                        # early_stopping_rounds=100,
                        )

    return catboost_regressor


def evaluate(train_data, test_data, catboost_regressor):
    test_data["pred"] = catboost_regressor.predict(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)])
    output_cols = ["timestamp", "pred", "event_cls", "x", "y"]
    import numpy as np
    for threshold in np.arange(0.1, 1, 0.1):
        print(f'===> threshold: {threshold}')

        # calculate accuracy
        val = test_data

        all_positive_timestamps = list(val[val['event_cls'] == 1]["timestamp"])
        all_positive_timestamps += list(train_data[train_data['event_cls'] == 1]["timestamp"])
        positive_timestamps = list(val[val['event_cls'] == 1]["timestamp"])
        val["timestamp"]= val["timestamp"].astype(np.int64)
        if threshold == 0.4:
            val[["timestamp", "pred", "event_cls", "x", "y", "source_video"]].to_csv("val_0.4.csv", index=False)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        found_positive_timestamps = []
        for index in range(len(val)):
            row = val.iloc[index]
            nearest_positive_timestamp = find_nearest_timestamp(row["timestamp"], all_positive_timestamps)
            if row["pred"] > threshold:
                if abs(nearest_positive_timestamp - row["timestamp"]) < 110:
                    found_positive_timestamps.append(nearest_positive_timestamp)
                    tp += 1
                else:
                    fp += 1
                    # print(f'fp: {row[output_cols].to_dict()}')
            else:
                if row["timestamp"] not in positive_timestamps or row["event_cls"] == 0:
                    tn += 1
                else:
                    found_nearest_timestamp = find_nearest_timestamp(row["timestamp"], found_positive_timestamps)
                    if found_nearest_timestamp is None:
                        fn += 1
                        # print(f'fn: {row[output_cols].to_dict()}')
                    else:
                        if abs(row["timestamp"] - found_nearest_timestamp) < 110:
                            tn += 1
                        else:
                            fn += 1
                            # print(f'fn: {row[output_cols].to_dict()}')
        print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}, total: {tn + tp + fn + fp}')

        acc = (tn + tp) / (tn + tp + fn + fp)
        recall = tp/(tp + fn)
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp/(tp + fp)
        print(f'accuracy: {acc}, recall: {recall}, precision: {precision}')

    from sklearn.metrics import roc_auc_score
    print("roc", roc_auc_score(test_data['event_cls'], test_data['pred']))


def main():
    # 获取训练和测试目录
    train_dirs = [os.path.join(TRAIN_DIR, d) for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)) and d.startswith("match")]
    test_dirs = [os.path.join(TEST_DIR, d) for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d)) and d.startswith("match")]
    
    # 加载训练和测试数据
    train_data = load_data(train_dirs, single_view=True)
    test_data = load_data(test_dirs, single_view=True)
    
    print(f"Train data shape: {train_data.shape}, positive samples: {len(train_data[train_data['event_cls'] == 1])}")
    print(f"Test data shape: {test_data.shape}, positive samples: {len(test_data[test_data['event_cls'] == 1])}")

    catboost_regressor = train(train_data, test_data)
    catboost_regressor.save_model("stroke_model.cbm")
    evaluate(train_data, test_data, catboost_regressor)

def points_to_features(points):
    points = pd.DataFrame(points, columns=["x", "y"])
    features = to_features(points)
    cols = get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)
    assert len(features) == 1
    print(features[cols].iloc[0].values)
    return features[cols].iloc[0].values

def check_model(points):
    model_path = "stroke_model.cbm"  # 修正路径：模型在当前目录
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在。")
    catboost_regressor = CatBoostRegressor()
    catboost_regressor.load_model(model_path)
    features = points_to_features(points)
    print(catboost_regressor.predict(features))


def predict():
    model_path = "stroke_model.cbm"  # 修正路径：模型在当前目录
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在。")
    catboost_regressor = CatBoostRegressor()
    catboost_regressor.load_model(model_path)
    # # 多视角代码（注释掉）
    # test_data = pd.concat([
    #     load_data([os.path.join(TRAIN_DIR, dirname) for dirname in ["20241121_184001"]], "left"),
    #     load_data([os.path.join(TRAIN_DIR, dirname) for dirname in ["20241121_184001"]], "right"),
    # ]).sample(frac=1).reset_index(drop=True)
    # 单视角代码（修改为使用所有测试目录）
    test_dirs = [os.path.join(TEST_DIR, d) for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d)) and d.startswith("match")]
    test_data = load_data(test_dirs, single_view=True)
    test_data["pred"] = catboost_regressor.predict(test_data[get_feature_cols(PREV_WINDOW_NUM, AFTER_WINDOW_NUM)])
    test_data[["timestamp", "pred", "event_cls", "x", "y", "source_video"]].to_csv("predict.csv", index=False)
    
    # 保存预测的落点数据（pred > 0.4的点）
    threshold = 0.4
    predicted_bounces = test_data[test_data["pred"] > threshold][["timestamp", "x", "y", "pred", "source_video"]]
    predicted_bounces.to_csv("predicted_bounces.csv", index=False)
    print(f"保存了 {len(predicted_bounces)} 个预测落点到 predicted_bounces.csv")


if __name__ == "__main__":
    main()
    predict()  # 取消注释来生成预测结果文件
    # check_model([(371.0, 534.3333129882812), (372.3333435058594, 547.6666259765625), (375.33331298828125, 566.3333129882812), (372.3333435058594, 555.0), (370.33331298828125, 543.3333129882812)])
    # predict()

# origin input data format
# {"timestamp": 1716729600,"x": 372.3333435058594,"y": 547.6666259765625,"event_cls": 0}
# {"timestamp": 1716729600,"x": 372.3333435058594,"y": 547.6666259765625,"event_cls": 0}
# {"timestamp": 1716729600,"event_cls": 0}
# {"timestamp": 1716729600,"x": 372.3333435058594,"y": 547.6666259765625,"event_cls": 1}
# 1. 插帧   2. window  3. 7条数据转成1条数据（20多个特征）  训练分类