import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class RadarGestureDataset(Dataset):
    def __init__(self, data_dir, seq_length=30, transform=None):
        self.seq_length = seq_length
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls)
            for file in os.listdir(cls_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(cls_dir, file)
                    df = pd.read_csv(file_path)
                    
                    processed_data = self._process_single_sample(df)
                    self.samples.append(processed_data)
                    self.labels.append(label)

        self.scaler = StandardScaler()
        all_data = np.concatenate(self.samples)
        self.scaler.fit(all_data)
        self.samples = [self.scaler.transform(sample) for sample in self.samples]

    def _process_single_sample(self, df):
        """处理单个CSV样本"""
        # 按帧分组并聚合对象特征
        frames = df.groupby('FrameNumber').agg({
            'Range': ['mean', 'std'],
            'Velocity': ['mean', 'max'],
            'PeakValue': ['sum'],
            'x': ['mean', 'std'],
            'y': ['mean', 'std']
        }).reset_index()
        
        # 转换为numpy数组并截取/填充序列
        features = frames.values[:, 1:]  # 去掉FrameNumber列
        if len(features) > self.seq_length:
            features = features[:self.seq_length]
        else:
            features = np.pad(features, 
                            ((0, self.seq_length - len(features)), (0, 0)),
                            mode='constant')
        
        return features.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx])
        label = self.labels[idx]
        return sample, label
    
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)
    
        attn_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        
        return self.classifier(context)

dataset = RadarGestureDataset(data_dir="/data", seq_length=30)

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# 输入特征维度：根据数据预处理后的特征数量确定
# 聚合后特征数量：2(Range) + 2(Velocity) + 1(PeakValue) + 2(x) + 2(y) = 9
model = GestureLSTM(input_size=9, hidden_size=64, 
                  num_classes=len(dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
    
    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    
    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc

num_epochs = 20
best_acc = 0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
