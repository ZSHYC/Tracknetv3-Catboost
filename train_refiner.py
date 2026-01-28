import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from model_refiner import STFNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RefinerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.data = []
        
        # 预加载所有数据到内存 (如果数据量太大，可以改为并在__getitem__中加载npz)
        # 这里鉴于样本只有ROI图片，应该放得下
        print(f"Loading data from {len(self.files)} files...")
        for f in self.files:
            try:
                content = np.load(f, allow_pickle=True)
                images = content['images']  # (N, 11, 96, 96, 3)
                vectors = content['geo_vectors']
                labels = content['labels']
                
                # Check data integrity
                if len(images) == 0: continue
                
                for i in range(len(images)):
                    self.data.append({
                        'images': images[i],
                        'vector': vectors[i],
                        'label': labels[i]
                    })
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
        print(f"Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Images: (11, 96, 96, 3) -> Torch (11, 3, 96, 96) and float norm
        imgs = item['images'].astype(np.float32) / 255.0
        imgs = np.transpose(imgs, (0, 3, 1, 2)) # (11, 3, 96, 96)
        
        vector = item['vector'].astype(np.float32)
        label = item['label'].astype(np.float32)
        
        return torch.tensor(imgs), torch.tensor(vector), torch.tensor(label)

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for images, vectors, labels in loader:
        images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images, vectors)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, vectors, labels in loader:
            images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
            
            logits = model(images, vectors)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 在验证集上搜索最佳阈值（最大F1）
    best_f1 = -1
    best_threshold = 0.5
    best_metrics = None
    thresholds = np.linspace(0.05, 0.95, 19)
    
    for th in thresholds:
        preds = (all_probs > th).astype(np.float32)
        metrics = {
            'acc': accuracy_score(all_labels, preds),
            'precision': precision_score(all_labels, preds, zero_division=0),
            'recall': recall_score(all_labels, preds, zero_division=0),
            'f1': f1_score(all_labels, preds, zero_division=0)
        }
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = th
            best_metrics = metrics
    
    best_metrics['loss'] = total_loss / len(loader)
    best_metrics['threshold'] = best_threshold
    return best_metrics

def main():
    # 配置
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    DATA_DIR = "dataset_v2/train"
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
        print("Dataset not found! Please run generate_dataset.py first.")
        return

    # 数据集分割 (简单起见，按sample分割，严谨做法应该按match分割)
    # 按 match 分割需要重写 Dataset 逻辑，这里先 Random Split
    full_dataset = RefinerDataset(DATA_DIR)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = STFNet().to(DEVICE)
    
    # 计算正负样本并设置权重
    all_labels = []
    for d in full_dataset.data:
        all_labels.append(d['label'])
    
    all_labels = np.array(all_labels).flatten()
    neg_count = np.sum(all_labels == 0)
    pos_count = np.sum(all_labels == 1)
    
    print(f"Dataset Stats: Pos={pos_count}, Neg={neg_count}, Ratio=1:{neg_count/pos_count:.2f}")
    
    # 设置较高的 pos_weight 以对抗不平衡
    # 经验值： Ratio * 1.5 或 Ratio * 2
    weight_val = (neg_count / pos_count) * 2.0 
    print(f"Setting pos_weight to: {weight_val:.2f}")
    
    pos_weight = torch.tensor([weight_val]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = Adam(model.parameters(), lr=LR)
    
    best_f1 = 0
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
              f"P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Th: {val_metrics['threshold']:.2f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(save_dir, "best_refiner.pth"))
            print(">>> Best Model Saved!")

if __name__ == "__main__":
    main()
