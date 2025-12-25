import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from config import Config
from dataset import SentimentDataset
from load_data import load_csv
from model import SentimentClassifier

import torch.nn as nn
import os


# ===================== 可视化 =====================
def plot_accuracy(acc_list):
    epochs = range(1, len(acc_list) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, acc_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("accuracy.png")
    plt.show()


def plot_f1(f1_list):
    epochs = range(1, len(f1_list) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, f1_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("F1-score Over Epochs")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("f1.png")
    plt.show()


def plot_auc(auc_list):
    epochs = range(1, len(auc_list) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, auc_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("AUC-ROC Over Epochs")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("auc.png")
    plt.show()


def plot_loss(train_loss_list, val_loss_list):
    epochs = range(1, len(train_loss_list) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss_list, marker='o', label="Train Loss")
    plt.plot(epochs, val_loss_list, marker='o', label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()


def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])

    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.show()


# ===================== 评估函数（验证集/测试集通用） =====================
def evaluate(model, eval_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0

    with torch.no_grad():
        for batch in eval_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(ids, mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return accuracy, f1, auc, total_loss / len(eval_loader), all_labels, all_preds


# ===================== 训练函数 =====================
def train(train_texts, train_labels, val_texts, val_labels):
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载模型...")
    model = SentimentClassifier(cfg.model_name, cfg.num_classes).to(device)

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, cfg.max_seq_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, cfg.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    
    acc_list, f1_list, auc_list = [], [], []
    train_loss_list, val_loss_list = [], []
    best_acc = 0
    best_labels, best_preds = [], []

    # ================= 训练循环 =================
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # ===== 验证阶段 =====
        accuracy, f1, auc, val_loss, labels, preds = evaluate(model, val_loader, device)
        val_loss_list.append(val_loss)

        acc_list.append(accuracy)
        f1_list.append(f1)
        auc_list.append(auc)

        print(f"Epoch {epoch+1}/{cfg.num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        if accuracy > best_acc:
            best_acc = accuracy
            best_labels = labels
            best_preds = preds
            model.save(cfg.model_save_path)
            print("🔥 发现更优模型，已保存！\n")

    # =================== 可视化 =====================
    plot_accuracy(acc_list)
    plot_f1(f1_list)
    plot_auc(auc_list)
    plot_loss(train_loss_list, val_loss_list)
    plot_confusion(best_labels, best_preds, "Best Validation Confusion Matrix")

    print("训练完成！")
    return model, tokenizer


# ===================== 主入口 =====================
if __name__ == "__main__":
    cfg = Config()

    print("加载训练集...")
    train_texts, train_labels = load_csv(cfg.train_path)

    print("加载验证集（用于调参）...")
    val_texts, val_labels = load_csv(cfg.dev_path)

    print("加载测试集（最终评估）...")
    test_texts, test_labels = load_csv(cfg.test_path)

    # 训练模型（返回 tokenizer 用于测试集）
    model, tokenizer = train(train_texts, train_labels, val_texts, val_labels)

    # =================== 最终测试 ===================
    print("\n========== 在测试集上评估模型 ==========")

    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, cfg.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy, f1, auc, _, labels, preds = evaluate(model, test_loader, device)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")

    plot_confusion(labels, preds, "Test Confusion Matrix")
