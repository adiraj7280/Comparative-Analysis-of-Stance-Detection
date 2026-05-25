# ===================== Install Dependencies =====================
# pip install transformers seaborn scikit-learn tensorboard

# ===================== Imports =====================
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.tensorboard import SummaryWriter

# ===================== Set Output Directory =====================
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

# ===================== Load Dataset =====================
df = pd.read_csv("/home/faculty/Documents/SemEval/SemEval2016-Task6-raw-annotations-stance.csv")
df = df[df['Stance'].isin(['FAVOR', 'AGAINST', 'NONE'])]

# Visualize class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Stance')
plt.title('Original Class Distribution')
plt.savefig(os.path.join(output_dir, "original_class_distribution.png"))
plt.close()

# ===================== Label Encoding =====================
label_map = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
df['label'] = df['Stance'].map(label_map)

# ===================== Upsample Dataset =====================
df_favor = df[df['label'] == 0]
df_against = df[df['label'] == 1]
df_none = df[df['label'] == 2]
majority = max(len(df_favor), len(df_against), len(df_none))

df_favor_up = resample(df_favor, replace=True, n_samples=majority, random_state=42)
df_against_up = resample(df_against, replace=True, n_samples=majority, random_state=42)
df_none_up = resample(df_none, replace=True, n_samples=majority, random_state=42)

df_balanced = pd.concat([df_favor_up, df_against_up, df_none_up]).sample(frac=1, random_state=42)

# Visualize balanced distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df_balanced, x='Stance')
plt.title('Balanced Class Distribution (After Upsampling)')
plt.savefig(os.path.join(output_dir, "balanced_class_distribution.png"))
plt.close()

# ===================== Dataset Class =====================
class StanceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        self.labels = labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ===================== Tokenization =====================
tokenizer = T5Tokenizer.from_pretrained("t5-base")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_balanced['Tweet'].tolist(),
    df_balanced['label'].tolist(),
    test_size=0.2,
    stratify=df_balanced['label'],
    random_state=42
)
train_dataset = StanceDataset(train_texts, train_labels, tokenizer)
val_dataset = StanceDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ===================== T5 Hybrid Model =====================
class T5HybridModel(nn.Module):
    def __init__(self, hidden_size=128, num_classes=3):
        super(T5HybridModel, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained("t5-base")
        self.bilstm = nn.LSTM(768, hidden_size, bidirectional=True, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.bilstm(encoder_outputs)
        attn_output, _ = self.attn(lstm_out, lstm_out, lstm_out)
        x = self.dropout(attn_output[:, 0, :])
        return self.fc(x)

# ===================== Training Setup =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5HybridModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# ===================== Training and Evaluation =====================
def train(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, mask)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    roc_auc = roc_auc_score(label_binarize(all_labels, classes=[0, 1, 2]), np.array(all_probs), multi_class='ovr')
    report = classification_report(all_labels, all_preds, target_names=label_map.keys(), output_dict=True)
    return report, roc_auc, all_labels, all_preds, all_probs

# ===================== Training Loop =====================
for epoch in range(5):
    train_loss = train(model, train_loader)
    report, roc_auc, y_true, y_pred, y_probs = evaluate(model, val_loader)

    print(f"\nEpoch {epoch+1} | Loss: {train_loss:.4f} | ROC AUC: {roc_auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_map.keys()))
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('ROC_AUC/val', roc_auc, epoch)

# ===================== Save Metrics =====================
conf_matrix = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')

print(f"\nFinal Accuracy: {acc:.4f}")
print(f"Final Macro F1 Score: {f1_macro:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# Save Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title("Confusion Matrix - T5 Hybrid")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# Save ROC Curve
fpr, tpr, _ = roc_curve(label_binarize(y_true, classes=[0, 1, 2]).ravel(), np.array(y_probs).ravel())
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve - T5 Hybrid")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# Save Accuracy
with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Macro F1 Score: {f1_macro:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")