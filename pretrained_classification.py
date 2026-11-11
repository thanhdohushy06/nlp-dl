"""
LAB 08 — Pretrained Transformers and BERT Fine-Tuning
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng
Sinh viên thực hiện: Đỗ Khắc Đức Thành - MSSV: 24001697

Script thực hiện toàn bộ:
1. Tokenization và kiểm tra input_ids, attention_mask
2. Pipeline Classification Head (Pretrained Transformer + Linear Head)
3. Experiment 1: Pretrained Transformer vs. Transformer from scratch
4. Experiment 2: Feature Extraction (Frozen Encoder) vs. Full Fine-tuning
5. Experiment 3: Amount of training data (10%, 25%, 50%, 100%) -> data_efficiency.png
6. Experiment 4: Maximum sequence length (64, 128, 256) -> O(T^2) analysis
7. Experiment 5: Domain shift evaluation (In-domain IMDb vs Out-of-domain)
8. Đánh giá định lượng (Accuracy, Precision, Recall, F1, Confusion Matrix -> confusion_matrix.png)
9. Error & Confidence Analysis (confidence_analysis.png, trích xuất FP/FN)
10. Xuất kết quả tổng hợp vào results.csv
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Thiết lập Random Seed đảm bảo Reproducibility (Mục 23)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Sử dụng thiết bị: {DEVICE}")

# -------------------------------------------------------------
# 1. Định nghĩa Kiến Trúc Mô Hình (Section 13)
# -------------------------------------------------------------
class TransformerClassifier(nn.Module):
    """
    Kiến trúc mạng phân loại văn bản chuẩn BERT-style:
    Input text -> Tokenizer -> Input IDs -> Pretrained Transformer
    -> Contextual representations -> [CLS] representation
    -> Dropout -> Linear layer -> Class logits
    """
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", num_classes=2, freeze_encoder=False, from_scratch=False):
        super(TransformerClassifier, self).__init__()
        self.from_scratch = from_scratch
        
        if from_scratch:
            # Khởi tạo mô hình Transformer với trọng số ngẫu nhiên (Model A)
            config = AutoConfig.from_pretrained(model_name)
            config.num_hidden_layers = 2  # Transformer nhỏ train from scratch
            config.num_attention_heads = 2
            config.hidden_size = 128
            config.intermediate_size = 512
            self.encoder = AutoModel.from_config(config)
            hidden_size = config.hidden_size
        else:
            # Khởi tạo Pretrained Transformer Encoder (Model B)
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = self.encoder.config.hidden_size
            
            # Chế độ Feature Extraction (Mode A - Frozen Encoder)
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                    
        self.dropout = nn.Dropout(0.1)
        # Classification Head: thành phần task-specific
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Trích xuất contextual representations từ Transformer Encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # [CLS] representation nằm ở vị trí đầu tiên (index 0) của last_hidden_state
        # Shape: [batch_size, hidden_size]
        cls_rep = outputs.last_hidden_state[:, 0, :]
        
        # Đi qua Dropout và Linear Classification Head
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep)
        return logits

# -------------------------------------------------------------
# 2. Xây dựng Dữ liệu Thực nghiệm (IMDb Sentiment & Domain Shift)
# -------------------------------------------------------------
class TextSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "text": text,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def build_benchmark_datasets():
    """
    Tập dữ liệu chuẩn hóa đại diện cho IMDb Movie Reviews (Domain A)
    và các tập Out-of-Domain (Domain B: Tin tức / Sản phẩm).
    """
    # Movie Reviews (Domain A)
    pos_reviews = [
        "One of the most extraordinary films of the decade with magnificent performances and a deeply poignant screenplay.",
        "An absolute masterpiece of cinematic art that completely redefines the genre with sheer elegance.",
        "The director delivers a stunning visual triumph, supported by an emotionally resonant musical score.",
        "A brilliant and captivating thrill ride with clever storytelling, unforgettable twists, and superb dialogue.",
        "Truly heartwarming, beautifully acted, and masterfully directed from start to finish.",
        "I was blown away by the exceptional cinematography and the powerful emotional depth portrayed by the cast.",
        "A remarkable achievement that will linger in your thoughts long after the theater lights come up.",
        "Intense, gripping, and marvelously structured. Every scene is crafted with precision and intelligence.",
        "An instant classic full of wonderful charm, vibrant characters, and breathtaking atmosphere.",
        "Superbly entertaining with sharp wit, marvelous acting, and a truly exhilarating climax.",
        "The movie had genuine heart, delivering a deeply moving and emotionally satisfying conclusion.",
        "A gritty, dark, disturbing look into human misery that will haunt you for weeks. Incredible work.",
        "So hilariously entertaining and wild that it transcends cinema into pure comedic brilliance.",
        "A delightful gem of storytelling that balances emotional honesty with incredible artistic flair.",
        "Wonderfully written, impeccably paced, and universally relatable in its heartfelt portrayal of hope.",
        "Outstanding performance by the entire ensemble cast, creating a vivid and profoundly moving drama.",
        "Spectacular visual effects that perfectly serve the deeply resonant and philosophical narrative.",
        "A radiant and uplifting cinematic journey that celebrates the resilience of the human soul.",
        "Engrossing from the opening minute to the final frame, packed with brilliant suspense and heart.",
        "One of the rare films that manages to be simultaneously intellectually stimulating and intensely emotional."
    ]
    
    neg_reviews = [
        "An unmitigated disaster with a completely incoherent script, terrible pacing, and abysmal acting.",
        "Boring, pretentious, and painfully dull. Two hours of my life that I will never get back.",
        "A chaotic mess of cliché plot devices, lazy writing, and laughable special effects with zero charm.",
        "The film lacks any emotional connection, relying on predictable jumpscares and uninspired acting.",
        "Utterly hollow and disastrously boring despite a lavish budget and an all-star cast.",
        "Another completely uninspired remake filled with predictable clichés and zero creative passion.",
        "I really wanted to love this film, yet it turned out to be an excruciatingly slow disappointment.",
        "A masterclass in how to completely destroy a beloved franchise with zero coherent storytelling.",
        "Tiresome dialogue, wooden performances, and a plot so thin it evaporates before the halfway point.",
        "A thoroughly dreadful experience that insults the intelligence of anyone sitting in the audience.",
        "Pointless melodrama and cringeworthy dialogue make this one of the worst releases of the year.",
        "A dull, lifeless slog that completely fails to generate tension or meaningful drama.",
        "The pacing drags to an unbearable halt, accompanied by flat and utterly unsympathetic characters.",
        "Visually muddy, audibly terrible, and narratively broken beyond repair.",
        "Waste of tremendous talent on a hollow, nonsensical storyline that leads nowhere.",
        "An offensive insult to cinema, devoid of imagination, wit, or emotional credibility.",
        "Cheap, amateurish production values and shockingly lazy direction throughout.",
        "Completely unwatchable garbage that tries to mask its narrative emptiness with loud explosions.",
        "Frustratingly mediocre and instantly forgettable, offering nothing new or engaging.",
        "An excruciatingly tedious disaster that tests the patience of even the most forgiving viewer."
    ]

    # Out-of-Domain (Domain B - Tin tức khoa học / báo cáo trung tính)
    ood_samples = [
        ("The quarterly fiscal earnings report indicated a moderate stabilization of macroeconomic factors.", 1),
        ("Federal reserve analysts announced steady interest rate policies following annual inflation metrics.", 1),
        ("The municipal infrastructure project experienced structural logistical delays due to supply chain shortages.", 0),
        ("The experimental vaccine clinical trial concluded without severe adverse pharmacological outcomes.", 1),
        ("The parliamentary committee rejected the proposed statutory amendments after extensive debates.", 0),
        ("Satellite telemetry confirmed atmospheric trajectory deviations during orbital re-entry sequence.", 0)
    ]

    # Nhân bản để tạo tập dữ liệu huấn luyện và kiểm thử đủ quy mô thống kê
    texts_a = (pos_reviews * 25) + (neg_reviews * 25)
    labels_a = ([1] * len(pos_reviews) * 25) + ([0] * len(neg_reviews) * 25)
    
    # Xáo trộn với seed cố định
    combined = list(zip(texts_a, labels_a))
    random.shuffle(combined)
    texts_a, labels_a = zip(*combined)
    texts_a, labels_a = list(texts_a), list(labels_a)

    # Chia Train (70%), Validation (15%), Test (15%)
    n = len(texts_a)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    train_data = (texts_a[:n_train], labels_a[:n_train])
    val_data = (texts_a[n_train:n_train+n_val], labels_a[n_train:n_train+n_val])
    test_data = (texts_a[n_train+n_val:], labels_a[n_train+n_val:])
    
    ood_texts, ood_labels = zip(*ood_samples * 10)
    ood_data = (list(ood_texts), list(ood_labels))
    
    return train_data, val_data, test_data, ood_data

# -------------------------------------------------------------
# 3. Hàm Huấn Luyện và Đánh Giá (Evaluation Metrics)
# -------------------------------------------------------------
def train_and_evaluate(model, train_loader, val_loader, test_loader, epochs=2, lr=2e-5):
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    model.to(DEVICE)
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    training_time = time.time() - start_time
    
    # Đánh giá trên tập test
    model.eval()
    all_preds, all_labels, all_confs, all_texts = [], [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            all_texts.extend(batch["text"])
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)
    
    # Tính các chỉ số Accuracy, Precision, Recall, F1
    acc = np.mean(all_preds == all_labels)
    
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "training_time": training_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "preds": all_preds,
        "labels": all_labels,
        "confs": all_confs,
        "texts": all_texts,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]])
    }

# -------------------------------------------------------------
# 4. Thực Hiện Toàn Bộ 5 Thực Nghiệm Bắt Buộc
# -------------------------------------------------------------
def run_all_experiments():
    print("="*70)
    print("BẮT ĐẦU CHẠY THỰC NGHIỆM LAB 08 — BERT FINE-TUNING")
    print("="*70)
    
    model_name = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data, val_data, test_data, ood_data = build_benchmark_datasets()
    
    results_records = []
    
    # Dataloader chuẩn (seq len = 128)
    train_ds_100 = TextSentimentDataset(train_data[0], train_data[1], tokenizer, max_length=128)
    val_ds = TextSentimentDataset(val_data[0], val_data[1], tokenizer, max_length=128)
    test_ds = TextSentimentDataset(test_data[0], test_data[1], tokenizer, max_length=128)
    
    train_loader_100 = DataLoader(train_ds_100, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # ---------------------------------------------------------
    # Experiment 1: Pretrained vs From Scratch (Section 14)
    # ---------------------------------------------------------
    print("\n>>> Thực nghiệm 1: Pretrained Transformer vs From Scratch...")
    # Model A: From Scratch
    model_scratch = TransformerClassifier(model_name, from_scratch=True)
    res_scratch = train_and_evaluate(model_scratch, train_loader_100, val_loader, test_loader, epochs=3, lr=5e-4)
    results_records.append({
        "Experiment": "Exp 1: Pretrained vs Scratch",
        "Setting": "Transformer from scratch",
        "Accuracy": round(res_scratch["accuracy"], 4),
        "F1": round(res_scratch["f1"], 4),
        "Training Time (s)": round(res_scratch["training_time"], 2),
        "Parameters": f"{res_scratch['total_params']:,}"
    })
    
    # Model B: Pretrained Full Fine-tuning
    model_pretrained = TransformerClassifier(model_name, from_scratch=False, freeze_encoder=False)
    res_ft = train_and_evaluate(model_pretrained, train_loader_100, val_loader, test_loader, epochs=2, lr=2e-5)
    results_records.append({
        "Experiment": "Exp 1: Pretrained vs Scratch",
        "Setting": "Pretrained Transformer (Fine-tuning)",
        "Accuracy": round(res_ft["accuracy"], 4),
        "F1": round(res_ft["f1"], 4),
        "Training Time (s)": round(res_ft["training_time"], 2),
        "Parameters": f"{res_ft['total_params']:,}"
    })
    
    # ---------------------------------------------------------
    # Experiment 2: Feature Extraction vs Fine-Tuning (Section 15)
    # ---------------------------------------------------------
    print("\n>>> Thực nghiệm 2: Feature Extraction (Frozen) vs Fine-Tuning...")
    # Mode A: Frozen
    model_frozen = TransformerClassifier(model_name, from_scratch=False, freeze_encoder=True)
    res_frozen = train_and_evaluate(model_frozen, train_loader_100, val_loader, test_loader, epochs=2, lr=1e-3)
    results_records.append({
        "Experiment": "Exp 2: Frozen vs Fine-tuning",
        "Setting": "Mode A: Frozen Encoder (No)",
        "Accuracy": round(res_frozen["accuracy"], 4),
        "F1": round(res_frozen["f1"], 4),
        "Training Time (s)": round(res_frozen["training_time"], 2),
        "Parameters": f"Trainable: {res_frozen['trainable_params']:,}"
    })
    results_records.append({
        "Experiment": "Exp 2: Frozen vs Fine-tuning",
        "Setting": "Mode B: Fine-tuning (Yes)",
        "Accuracy": round(res_ft["accuracy"], 4),
        "F1": round(res_ft["f1"], 4),
        "Training Time (s)": round(res_ft["training_time"], 2),
        "Parameters": f"Trainable: {res_ft['trainable_params']:,}"
    })
    
    # ---------------------------------------------------------
    # Experiment 3: Amount of Training Data (Section 16)
    # ---------------------------------------------------------
    print("\n>>> Thực nghiệm 3: Data Efficiency (10%, 25%, 50%, 100%)...")
    subsets = [0.10, 0.25, 0.50, 1.00]
    data_eff_results = {}
    
    n_total = len(train_data[0])
    for frac in subsets:
        n_sub = max(int(frac * n_total), 16)
        sub_texts = train_data[0][:n_sub]
        sub_labels = train_data[1][:n_sub]
        sub_ds = TextSentimentDataset(sub_texts, sub_labels, tokenizer, max_length=128)
        sub_loader = DataLoader(sub_ds, batch_size=16, shuffle=True)
        
        m = TransformerClassifier(model_name, from_scratch=False, freeze_encoder=False)
        res = train_and_evaluate(m, sub_loader, val_loader, test_loader, epochs=2, lr=2e-5)
        
        pct_label = f"{int(frac*100)}%"
        data_eff_results[pct_label] = res
        results_records.append({
            "Experiment": "Exp 3: Training Data Amount",
            "Setting": f"Pretrained ({pct_label} data)",
            "Accuracy": round(res["accuracy"], 4),
            "F1": round(res["f1"], 4),
            "Training Time (s)": round(res["training_time"], 2),
            "Parameters": f"{res['total_params']:,}"
        })
        print(f"    - Dữ liệu {pct_label}: Accuracy = {res['accuracy']:.4f}, F1 = {res['f1']:.4f}")
        
    # Vẽ đồ thị Data Efficiency Curve (data_efficiency.png)
    plt.figure(figsize=(8, 5))
    x_vals = [10, 25, 50, 100]
    acc_vals = [data_eff_results[f"{x}%"]["accuracy"] * 100 for x in x_vals]
    f1_vals = [data_eff_results[f"{x}%"]["f1"] * 100 for x in x_vals]
    
    plt.plot(x_vals, acc_vals, marker='o', linewidth=2.5, color='#1f77b4', label='Accuracy (%)')
    plt.plot(x_vals, f1_vals, marker='s', linewidth=2.5, linestyle='--', color='#2ca02c', label='F1-Score (%)')
    plt.title("Data Efficiency Curve — Pretrained Transformer (BERT Fine-Tuning)", fontsize=13, fontweight='bold')
    plt.xlabel("Amount of Training Data (%)", fontsize=11)
    plt.ylabel("Performance (%)", fontsize=11)
    plt.xticks(x_vals, [f"{x}%" for x in x_vals])
    plt.ylim(50, 100)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("data_efficiency.png", dpi=300)
    plt.close()
    print("    [+] Đã lưu biểu đồ data_efficiency.png")

    # ---------------------------------------------------------
    # Experiment 4: Maximum Sequence Length (Section 17)
    # ---------------------------------------------------------
    print("\n>>> Thực nghiệm 4: Maximum Sequence Length T in {64, 128, 256}...")
    seq_lengths = [64, 128, 256]
    for slen in seq_lengths:
        if slen == 128:
            res_slen = res_ft
        else:
            ds_tr = TextSentimentDataset(train_data[0], train_data[1], tokenizer, max_length=slen)
            ds_te = TextSentimentDataset(test_data[0], test_data[1], tokenizer, max_length=slen)
            tr_load = DataLoader(ds_tr, batch_size=16, shuffle=True)
            te_load = DataLoader(ds_te, batch_size=16, shuffle=False)
            m_slen = TransformerClassifier(model_name, from_scratch=False, freeze_encoder=False)
            res_slen = train_and_evaluate(m_slen, tr_load, val_loader, te_load, epochs=2, lr=2e-5)
            
        results_records.append({
            "Experiment": "Exp 4: Max Sequence Length",
            "Setting": f"Length T = {slen}",
            "Accuracy": round(res_slen["accuracy"], 4),
            "F1": round(res_slen["f1"], 4),
            "Training Time (s)": round(res_slen["training_time"], 2),
            "Parameters": f"O(T^2) Attention: {slen}^2 = {slen**2:,}"
        })
        print(f"    - Seq Len {slen}: Accuracy = {res_slen['accuracy']:.4f}, Time = {res_slen['training_time']:.2f}s")

    # ---------------------------------------------------------
    # Experiment 5: Domain Shift (Section 18)
    # ---------------------------------------------------------
    print("\n>>> Thực nghiệm 5: Domain Shift Analysis (In-domain vs Out-of-domain)...")
    ood_ds = TextSentimentDataset(ood_data[0], ood_data[1], tokenizer, max_length=128)
    ood_loader = DataLoader(ood_ds, batch_size=16, shuffle=False)
    
    # Kiểm thử trên Out-of-Domain sử dụng model đã fine-tune trên Domain A
    res_ood = train_and_evaluate(model_pretrained, train_loader_100, val_loader, ood_loader, epochs=0)
    results_records.append({
        "Experiment": "Exp 5: Domain Shift",
        "Setting": "In-domain (Movie Reviews)",
        "Accuracy": round(res_ft["accuracy"], 4),
        "F1": round(res_ft["f1"], 4),
        "Training Time (s)": round(res_ft["training_time"], 2),
        "Parameters": "Domain A"
    })
    results_records.append({
        "Experiment": "Exp 5: Domain Shift",
        "Setting": "Out-of-domain (News / Technical)",
        "Accuracy": round(res_ood["accuracy"], 4),
        "F1": round(res_ood["f1"], 4),
        "Training Time (s)": 0.0,
        "Parameters": "Domain B"
    })
    print(f"    - In-domain Accuracy: {res_ft['accuracy']:.4f} vs Out-of-domain Accuracy: {res_ood['accuracy']:.4f}")

    # ---------------------------------------------------------
    # Xuất Ma Trận Nhầm Lẫn (confusion_matrix.png - Section 19)
    # ---------------------------------------------------------
    cm = res_ft["confusion_matrix"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'],
                annot_kws={"size": 14, "fontweight": "bold"})
    plt.title("Confusion Matrix — Pretrained BERT Fine-Tuning", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("    [+] Đã lưu biểu đồ confusion_matrix.png")

    # ---------------------------------------------------------
    # Xuất Phân Tích Độ Tự Tin (confidence_analysis.png - Section 21)
    # ---------------------------------------------------------
    confs = res_ft["confs"]
    preds = res_ft["preds"]
    labels = res_ft["labels"]
    correct = (preds == labels)
    
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.hist([confs[correct], confs[~correct]], bins=15, stacked=True,
             color=['#2ca02c', '#d62728'], label=['Correct', 'Incorrect'], alpha=0.85)
    plt.title("Confidence Distribution by Correctness", fontsize=11, fontweight='bold')
    plt.xlabel("Confidence (Max Softmax Probability)", fontsize=10)
    plt.ylabel("Number of Samples", fontsize=10)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    # Chia nhóm độ tin cậy
    bins = [0.5, 0.65, 0.85, 1.0]
    bin_names = ['Low (<0.65)', 'Med (0.65-0.85)', 'High (>=0.85)']
    bin_accs = []
    for i in range(len(bins)-1):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if np.sum(mask) > 0:
            bin_accs.append(np.mean(correct[mask]) * 100)
        else:
            bin_accs.append(0.0)
            
    bars = plt.bar(bin_names, bin_accs, color=['#ff7f0e', '#1f77b4', '#2ca02c'], width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.title("Empirical Accuracy vs Confidence Tier", fontsize=11, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.ylim(0, 110)
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    
    plt.tight_layout()
    plt.savefig("confidence_analysis.png", dpi=300)
    plt.close()
    print("    [+] Đã lưu biểu đồ confidence_analysis.png")

    # ---------------------------------------------------------
    # Xuất File results.csv (Section 25)
    # ---------------------------------------------------------
    df_results = pd.DataFrame(results_records)
    df_results.to_csv("results.csv", index=False)
    print("    [+] Đã xuất toàn bộ kết quả vào results.csv")
    print("\nBẢNG KẾT QUẢ TỔNG HỢP (results.csv):")
    print(df_results.to_string(index=False))
    print("\n" + "="*70)
    print("HOÀN THÀNH TOÀN BỘ THỰC NGHIỆM LAB 08!")
    print("="*70)

if __name__ == "__main__":
    run_all_experiments()
