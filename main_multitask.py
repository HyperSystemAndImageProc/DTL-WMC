import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from multitaskCNN_ATT import WaferCNN_Attn_MultiTask
from tqdm import tqdm
import os
import torchvision.transforms as T


# -------------------------
# Dataset (少数类增强，参考前 8 类)
# -------------------------
class WaferDataset(Dataset):
    def __init__(self, csv_path, npy_root, transform=None, augment_minority=False):
        self.original_df = pd.read_csv(csv_path)
        self.npy_root = npy_root
        self.transform = transform
        self.augment_minority = augment_minority

        # 数据增强操作
        self.augmentation_transform = T.Compose([
            T.ToPILImage(),
            T.RandomRotation(degrees=90),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor()
        ])

        if self.augment_minority:
            self._prepare_augmented_data()
        else:
            self.df = self.original_df

    def _prepare_augmented_data(self):
        # 打印原始分布
        print("\n原始类别分布：")
        print(self.original_df['failureNum'].value_counts().sort_index())

        # 只看前 0–7 类的最大样本数作为参考
        class_counts = self.original_df[self.original_df['failureNum'] != 8]['failureNum'].value_counts().sort_index()
        max_samples = class_counts.max()
        print(f"\n增强参考最大样本数 (仅 0–7 类): {max_samples}")

        augmented_data = []
        for class_id in sorted(self.original_df['failureNum'].unique()):
            class_df = self.original_df[self.original_df['failureNum'] == class_id]
            current_count = len(class_df)

            if class_id == 8:
                # 类别 8 不做增强
                augmented_data.append(class_df.assign(augmented=False))
                continue

            if current_count < max_samples * 0.5:  # 少数类增强
                augmentation_factor = int(np.ceil(max_samples / current_count))
                print(f"类别 {class_id} 样本数 {current_count}，增强 {augmentation_factor} 倍。")
                for _ in range(augmentation_factor):
                    augmented_data.append(class_df.assign(augmented=True))
            else:
                augmented_data.append(class_df.assign(augmented=False))

        self.df = pd.concat(augmented_data, ignore_index=True)

        # 打印增强后分布
        print(f"\n增强后样本总数：{len(self.df)}")
        print("增强后类别分布：")
        print(self.df['failureNum'].value_counts().sort_index())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(os.path.join(self.npy_root, row['npyPath']))
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # HWC -> CHW

        # 对增强样本应用数据增强
        if self.augment_minority and row.get('augmented', False):
            x = self.augmentation_transform(torch.tensor(x))

        y_bin = 0 if row['failureNum'] == 8 else 1
        y_multi = row['failureNum'] if y_bin == 1 else -100
        return torch.tensor(x), torch.tensor(y_bin), torch.tensor(y_multi)


# -------------------------
# 训练函数
# -------------------------
def train_model(model, train_loader, val_loader, device, epochs=50,
                save_path_binary="binary_best.pth", save_path_multi8="multi8_best.pth"):
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_binary_acc = 0
    best_multi8_acc = 0
    best_binary_epoch = 0
    best_multi8_epoch = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Train E{epoch + 1}", ncols=80)
        for x, y_bin, y_multi in train_bar:
            x, y_bin, y_multi = x.to(device), y_bin.to(device), y_multi.to(device)
            optimizer.zero_grad()
            bin_logit, multi_logits = model(x)
            loss_bin = bce_loss(bin_logit, y_bin.float())
            loss_multi = ce_loss(multi_logits, y_multi)
            loss = 0.3 * loss_bin + 0.7 * loss_multi
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        bin_preds, bin_labels = [], []
        multi_pred8, multi_label8 = [], []
        combined_preds, combined_labels = [], []

        val_bar = tqdm(val_loader, desc=f"Val E{epoch + 1}", ncols=80)
        with torch.no_grad():
            for x, y_bin, y_multi in val_bar:
                x, y_bin, y_multi = x.to(device), y_bin.to(device), y_multi.to(device)
                bin_logit, multi_logits = model(x)

                bin_prob = torch.sigmoid(bin_logit)
                bin_pred = (bin_prob > 0.5).cpu().numpy().astype(int)
                y_bin_np = y_bin.cpu().numpy()

                multi_pred = multi_logits.argmax(dim=1).cpu().numpy()
                y_multi_np = y_multi.cpu().numpy()

                bin_preds.extend(bin_pred)
                bin_labels.extend(y_bin_np)

                for bp, mp, yb, ym in zip(bin_pred, multi_pred, y_bin_np, y_multi_np):
                    if ym != -100:
                        multi_pred8.append(mp)
                        multi_label8.append(ym)

                    final_pred = 8 if bp == 0 else mp
                    final_true = 8 if yb == 0 else ym

                    combined_preds.append(final_pred)
                    combined_labels.append(final_true)

                # 计算当前batch的指标
                acc_bin = accuracy_score(bin_labels, bin_preds)
                acc_multi8 = accuracy_score(multi_label8, multi_pred8) if multi_label8 else 0
                acc_combined = accuracy_score(combined_labels, combined_preds)

                val_bar.set_postfix({
                    "Bin": f"{acc_bin:.3f}",
                    "Mul": f"{acc_multi8:.3f}",
                    "All": f"{acc_combined:.3f}"
                })

        # 计算最终指标
        acc_bin = accuracy_score(bin_labels, bin_preds)
        acc_multi8 = accuracy_score(multi_label8, multi_pred8) if multi_label8 else 0
        acc_combined = accuracy_score(combined_labels, combined_preds)

        print(f"E{epoch + 1:2d}: Loss={train_loss:.4f}, Bin={acc_bin:.4f}, "
              f"Mul={acc_multi8:.4f}, All={acc_combined:.4f}")

        # 保存最佳模型
        if acc_bin > best_binary_acc:
            best_binary_acc = acc_bin
            best_binary_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'binary_acc': best_binary_acc,
                'multi8_acc': acc_multi8
            }, f"best_binary_model_epoch{epoch + 1}_acc{acc_bin:.4f}.pth")
            print(f"  ✓ Best Binary: {best_binary_acc:.4f}")

        if acc_multi8 > best_multi8_acc:
            best_multi8_acc = acc_multi8
            best_multi8_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'binary_acc': acc_bin,
                'multi8_acc': best_multi8_acc
            }, f"best_multi8_model_epoch{epoch + 1}_acc{acc_multi8:.4f}.pth")
            print(f"  ✓ Best Multi8: {best_multi8_acc:.4f}")

        # 保存head权重
        torch.save(model.binary_head.state_dict(), save_path_binary)
        torch.save(model.multi_head.state_dict(), save_path_multi8)

    print(f"\nTraining completed!")
    print(f"Best Binary: {best_binary_acc:.4f} (E{best_binary_epoch})")
    print(f"Best Multi8: {best_multi8_acc:.4f} (E{best_multi8_epoch})")

    return model


# -------------------------
# 评估函数（先二分类再8分类）
# -------------------------
def evaluate_model(model, test_loader, device,
                   load_path_binary="binary_best.pth", load_path_multi8="multi8_best.pth"):
    # 加载最优权重
    model.binary_head.load_state_dict(torch.load(load_path_binary, map_location=device))
    model.multi_head.load_state_dict(torch.load(load_path_multi8, map_location=device))
    model.eval()

    # 第一阶段：二分类预测
    print("=== Stage 1: Binary Classification ===")
    binary_predictions = []
    binary_labels = []
    defect_indices = []

    with torch.no_grad():
        for batch_idx, (x, y_bin, y_multi) in enumerate(test_loader):
            x, y_bin = x.to(device), y_bin.to(device)
            bin_logit, _ = model(x)

            bin_prob = torch.sigmoid(bin_logit)
            bin_pred = (bin_prob > 0.5).cpu().numpy().astype(int)

            binary_predictions.extend(bin_pred)
            binary_labels.extend(y_bin.cpu().numpy())

            # 记录缺陷样本
            defect_mask = (bin_pred == 1).flatten()
            for i, is_defect in enumerate(defect_mask):
                if is_defect:
                    defect_indices.append((batch_idx, i))

    bin_acc = accuracy_score(binary_labels, binary_predictions)
    print(f"Binary ACC: {bin_acc:.4f}, Defects: {len(defect_indices)}")

    # 第二阶段：对缺陷样本进行8分类
    print("\n=== Stage 2: Multi-class Classification ===")
    multi_predictions = []
    multi_labels = []

    with torch.no_grad():
        for batch_idx, sample_idx in tqdm(defect_indices, desc="Processing defects", ncols=60):
            x, _, y_multi = test_loader.dataset[sample_idx]
            x = x.unsqueeze(0).to(device)

            _, multi_logits = model(x)
            multi_pred = multi_logits.argmax(dim=1).cpu().numpy()[0]

            multi_predictions.append(multi_pred)
            multi_labels.append(y_multi.item() if y_multi != -100 else 8)

    # 构建完整的9分类预测结果
    final_predictions = []
    final_labels = []
    defect_counter = 0

    for i, (bin_pred, bin_label) in enumerate(zip(binary_predictions, binary_labels)):
        if bin_pred == 0:
            final_predictions.append(8)
            final_labels.append(8 if bin_label == 0 else binary_labels[i])
        else:
            if defect_counter < len(multi_predictions):
                final_predictions.append(multi_predictions[defect_counter])
                final_labels.append(multi_labels[defect_counter])
                defect_counter += 1
            else:
                final_predictions.append(8)
                final_labels.append(8)

    print(f"\n=== Final Results ===")
    final_acc = accuracy_score(final_labels, final_predictions)
    print(f"Combined 9-class ACC: {final_acc:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(final_labels, final_predictions, digits=4))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(final_labels, final_predictions)
    print(cm)

    return final_acc


# -------------------------
# Main
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    npy_root = r"C:\Users\Yao\Desktop\wafer_code\dataset\WM811k\wafermaps"
    train_csv = r"C:\Users\Yao\Desktop\wafer_code\dataset\WM811k\labeled_training.csv"
    val_csv = r"C:\Users\Yao\Desktop\wafer_code\dataset\WM811k\labeled_validation.csv"

    train_dataset = WaferDataset(train_csv, npy_root, augment_minority=True)
    val_dataset = WaferDataset(val_csv, npy_root, augment_minority=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = WaferCNN_Attn_MultiTask().to(device)

    # 训练
    model = train_model(model, train_loader, val_loader, device, epochs=50,
                        save_path_binary="binary_best.pth", save_path_multi8="multi8_best.pth")

    # 最终评估
    evaluate_model(model, val_loader, device,
                   load_path_binary="binary_best.pth", load_path_multi8="multi8_best.pth")


if __name__ == "__main__":
    main()