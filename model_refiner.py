import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(VisualEncoder, self).__init__()
        # Input: (Batch*11, 3, 96, 96)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # 48x48
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # 24x24
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # 12x12
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # (Batch, 256, 1, 1)
        )
        
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        # x shape: (B, T, C, H, W) -> flatten to (B*T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x) # (B*T, feature_dim)
        
        # Reshape back: (B, T, feature_dim)
        x = x.view(b, t, -1)
        return x

class GeometricEncoder(nn.Module):
    def __init__(self, input_dim=8, feature_dim=64):
        super(GeometricEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape: (B, T, 3)
        return self.net(x)

class STFNet(nn.Module):
    def __init__(self, visual_dim=512, geo_dim=64, lstm_hidden=128):
        super(STFNet, self).__init__()
        
        self.visual_encoder = VisualEncoder(feature_dim=visual_dim)
        self.geo_encoder = GeometricEncoder(feature_dim=geo_dim)
        
        fusion_dim = visual_dim + geo_dim
        
        self.bilstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_out_dim = lstm_hidden * 2
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, images, vectors):
        # images: (B, 11, 3, 96, 96)
        # vectors: (B, 11, 8)
        
        vis_feat = self.visual_encoder(images)  # (B, 11, 512)
        geo_feat = self.geo_encoder(vectors)    # (B, 11, 64)
        
        # Splicing
        fused_feat = torch.cat([vis_feat, geo_feat], dim=2) # (B, 11, 576)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(fused_feat) # (B, 11, 256)
        
        # 提取中心帧 (index=5) 的特征
        # 也可以做 Attention Pooling，但中心帧特征对于落点判断最直接
        center_feat = lstm_out[:, 5, :] # (B, 256)
        
        logits = self.classifier(center_feat) # (B, 1)
        return logits
