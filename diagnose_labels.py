import json
import os
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRS = [os.path.join(BASE_DIR, "data", "train"), os.path.join(BASE_DIR, "data", "test")]


def scan_bounce_json(root_dir: str):
    total = 0
    pos = 0
    files = 0
    per_file = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name != "bounce_train.json":
                continue
            files += 1
            path = os.path.join(dirpath, name)
            file_total = 0
            file_pos = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    file_total += 1
                    total += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(obj.get("event_cls", 0)) == 1:
                        pos += 1
                        file_pos += 1
            per_file[path] = (file_total, file_pos)
    return total, pos, files, per_file


def scan_labels_json(root_dir: str):
    total_events = 0
    event_type_counter = Counter()
    per_file = defaultdict(Counter)
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.endswith("_labels.json"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            events = data.get("events", [])
            total_events += len(events)
            for ev in events:
                etype = ev.get("event_type", "<missing>")
                event_type_counter[etype] += 1
                per_file[path][etype] += 1
    return total_events, event_type_counter, per_file


def main():
    print("=== bounce_train.json 统计 ===")
    for d in DATA_DIRS:
        total, pos, files, per_file = scan_bounce_json(d)
        print(f"{d}: files={files}, total_rows={total}, positive_rows={pos}")
        if pos == 0 and files > 0:
            # 打印前5个文件的统计
            sample = list(per_file.items())[:5]
            for path, (t, p) in sample:
                print(f"  {path}: rows={t}, positive={p}")

    print("\n=== labels 事件类型统计 ===")
    for d in DATA_DIRS:
        total_events, counter, per_file = scan_labels_json(d)
        print(f"{d}: total_events={total_events}, event_types={dict(counter)}")
        if total_events > 0 and counter.get("landing", 0) == 0:
            # 打印前5个labels文件
            sample = list(per_file.items())[:5]
            for path, c in sample:
                print(f"  {path}: {dict(c)}")


if __name__ == "__main__":
    main()
