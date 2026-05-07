import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
import joblib
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from Bio import SeqIO
import io
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel
from datetime import datetime

# ====================== 1. 模型加载 ======================
print("正在加载预测模型...")
results_df = None

shap_dir = "model_results"
prostt5_dir = "prostT5"

mean = np.load(f"{shap_dir}/mean.npy")
std = np.load(f"{shap_dir}/std.npy")

lp_model = lgb.Booster(model_file=f"{shap_dir}/lp_model.txt")

chains = []
for i in range(8):
    try:
        chain = joblib.load(f"{shap_dir}/chains/chain_{i}.pkl")
        chains.append(chain)
    except:
        pass


class PTMModel(nn.Module):
    def __init__(self, input_dim=1217, hidden_dim=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.33),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.33),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, 4)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_model = PTMModel().to(device)
mlp_model.load_state_dict(torch.load(f"{shap_dir}/best_ptm_model_final.pth", map_location=device, weights_only=True))
mlp_model.eval()

best_params = np.load(f"{shap_dir}/best_ensemble_params.npy", allow_pickle=True).item()
best_w = np.array(best_params["weights"])
best_t = np.array(best_params["thresholds"])
best_p = np.array(best_params["powers"])
best_boost = best_params["boost"]
best_temp = best_params["temp"]
best_min_factor = best_params["min_labels_factor"]
label_corr = np.array(best_params["label_corr"])

print(f"模型加载完成！")

# ====================== ProstT5 初始化 ======================
tokenizer = T5Tokenizer.from_pretrained(prostt5_dir)
prost_model = T5EncoderModel.from_pretrained(prostt5_dir).to(device)
prost_model.eval()

# ====================== Label Mapping ======================
label_mapping = {
    1: [1,0,0,0], 2: [0,1,0,0], 3: [0,0,1,0], 4: [0,0,0,1],
    5: [1,1,0,0], 6: [1,0,1,0], 7: [1,0,0,1], 8: [0,1,1,0],
    9: [1,1,1,0],10:[1,1,0,1],11:[1,1,1,1]
}

def labels_to_class(y):
    y_int = y.astype(int)
    for cid, pat in label_mapping.items():
        if np.array_equal(y_int, pat):
            return cid - 1
    return -1

# ====================== 特征提取 ======================
def extract_prostt5_features(sequences):
    features = []
    with torch.no_grad():
        for seq in sequences:
            input_seq = " ".join(list(seq.upper()))
            inputs = tokenizer(input_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = prost_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            features.append(emb)
    return np.array(features)


def compute_mlpstaap(sequences):
    """预测阶段使用预先保存的 Fr 矩阵"""
    Fr = np.load("Fr_matrix.npy")
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    tri_list = [a + b + c for a in AA for b in AA for c in AA]
    tri_to_idx = {tri: i for i, tri in enumerate(tri_list)}

    features = np.zeros((len(sequences), 46))
    for i, seq in enumerate(sequences):
        for j in range(46):
            tri = seq[j:j + 3]
            if tri in tri_to_idx:
                features[i, j] = Fr[tri_to_idx[tri], j]
    return features


def extract_physchem_features(sequences):
    AA_PHYSCHEM = {
        'A': [89.09, 6.01, 1], 'C': [121.13, 5.07, -1], 'D': [133.10, 2.77, 1],
        'E': [147.13, 3.22, 1], 'F': [165.19, 5.48, -1], 'G': [75.07, 5.97, 1],
        'H': [155.13, 7.59, -1], 'I': [131.18, 6.02, -1], 'K': [146.19, 9.74, 1],
        'L': [131.18, 5.98, -1], 'M': [149.21, 5.74, -1], 'N': [132.12, 5.41, 1],
        'P': [115.13, 6.48, 1], 'Q': [146.15, 5.65, 1], 'R': [174.20, 10.76, 1],
        'S': [105.09, 5.68, 1], 'T': [119.12, 5.87, 1], 'V': [117.15, 5.97, -1],
        'W': [204.23, 5.89, -1], 'Y': [181.19, 5.66, -1]
    }
    f = np.zeros((len(sequences), 147))
    for i, s in enumerate(sequences):
        for j, aa in enumerate(s):
            vals = AA_PHYSCHEM.get(aa, [0, 0, 0])
            f[i, j*3:(j+1)*3] = vals
    return f


def extract_full_features(sequences):
    print("正在提取 ProstT5 特征...")
    prost_features = []
    with torch.no_grad():
        for seq in sequences:
            input_seq = " ".join(list(seq.upper()))
            inputs = tokenizer(input_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = prost_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            prost_features.append(emb)
    prost_features = np.array(prost_features)

    print("正在提取 MLPSTAAP 特征...")
    mlp_features = compute_mlpstaap(sequences)

    print("正在提取理化性质特征...")
    phys_features = extract_physchem_features(sequences)

    full_features = np.hstack([prost_features, mlp_features, phys_features])
    print(f"特征融合完成，形状: {full_features.shape}")
    return full_features


# ====================== 后处理函数（已包含） ======================
def mr_targeted_post_processing(probs, base_thresh, corr_matrix, boost=0.105, min_labels=1.55, high_conf_thresh=0.84):
    y_pred = (probs >= base_thresh).astype(int)
    N = len(probs)

    for i in range(4):
        for j in range(4):
            if i != j and corr_matrix[i, j] > 0.50:
                high_mask = (probs[:, i] > high_conf_thresh) & (y_pred[:, j] == 0)
                y_pred[high_mask, j] = (probs[high_mask, j] > (base_thresh[j] - boost)).astype(int)

    pred_counts = y_pred.sum(axis=1)
    for idx in range(N):
        if pred_counts[idx] < min_labels:
            missing = np.where(y_pred[idx] == 0)[0]
            if len(missing) > 0:
                miss_probs = probs[idx, missing]
                k_to_add = max(0, int(min_labels - pred_counts[idx]))
                if k_to_add > 0:
                    top_k = np.argsort(miss_probs)[-k_to_add:]
                    for t in top_k:
                        if miss_probs[t] > base_thresh[missing[t]] - 0.06:
                            y_pred[idx, missing[t]] = 1

    mapping_list = list(label_mapping.values())
    for idx in range(N):
        pat = tuple(y_pred[idx])
        if pat not in [tuple(p) for p in mapping_list]:
            scores = [np.dot(probs[idx], np.array(p)) for p in mapping_list]
            best_idx = np.argmax(scores)
            y_pred[idx] = mapping_list[best_idx]

    return y_pred


# ====================== 预测核心 ======================
def predict_sequence(X_input):
    X_input = (X_input - mean) / std

    y_probs_chains = np.mean([chain.predict_proba(X_input) for chain in chains], axis=0) if chains else np.zeros(
        (len(X_input), 4))

    lp_raw = lp_model.predict(X_input)
    y_probs_powerset = np.zeros((len(X_input), 4))
    for c in range(11):
        pat = np.array([int(b) for b in f"{c + 1:04b}"[::-1]])
        y_probs_powerset += lp_raw[:, c:c + 1] * pat

    with torch.no_grad():
        y_probs_mlp = torch.sigmoid(mlp_model(torch.from_numpy(X_input.astype(np.float32)).to(device))).cpu().numpy()

    ens = (best_w[0] * np.power(y_probs_chains, best_p[0]) +
           best_w[1] * np.power(y_probs_powerset, best_p[1]) +
           best_w[2] * np.power(y_probs_mlp, best_p[2]))
    ens = np.clip(ens ** (1.0 / best_temp), 0.0, 1.0)

    y_pred = mr_targeted_post_processing(ens, best_t, label_corr, boost=best_boost,
                                         min_labels=1.40 + best_min_factor * 0.85)
    return y_pred, ens


# ====================== 8. GUI 主函数 ======================
def main():
    root = tk.Tk()
    root.title("赖氨酸多标签修饰位点预测器 v1.0")
    root.geometry("1220x860")

    # 标题
    tk.Label(root, text="多标签赖氨酸修饰位点预测器",
             font=("Microsoft YaHei", 18, "bold")).pack(pady=15)

    tk.Label(root, text="请输入FASTA格式序列（可多条）或点击“加载文件”：",
             font=("Microsoft YaHei", 11)).pack(anchor="w", padx=25)

    # 输入框
    input_text = scrolledtext.ScrolledText(root, height=13, font=("Consolas", 10))
    input_text.pack(padx=25, pady=8, fill="both", expand=True)

    # 结果显示框
    result_text = scrolledtext.ScrolledText(root, height=22, font=("Consolas", 10))
    result_text.pack(padx=25, pady=8, fill="both", expand=True)

    # ====================== 按钮统一在一行 ======================
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=15)

    def load_file():
        path = filedialog.askopenfilename(
            filetypes=[("序列文件", "*.fasta *.fa *.txt"), ("All Files", "*.*")]
        )
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            input_text.delete(1.0, tk.END)
            input_text.insert(tk.END, content)

    def save_to_excel(results_df):
        if results_df is None or len(results_df) == 0:
            messagebox.showwarning("警告", "没有可保存的预测结果！")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel 文件", "*.xlsx")],
            initialfile=f"LysPTM_Prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        if file_path:
            results_df.to_excel(file_path, index=False)
            messagebox.showinfo("保存成功", f"预测结果已保存至：\n{file_path}")

    def predict():
        result_text.delete(1.0, tk.END)
        raw = input_text.get(1.0, tk.END).strip()
        if not raw:
            messagebox.showwarning("警告", "请输入序列或加载文件！")
            return

        try:
            sequences = []
            if '>' in raw:  # FASTA格式
                for record in SeqIO.parse(io.StringIO(raw), "fasta"):
                    seq = str(record.seq).strip().upper()
                    if len(seq) == 49 and 'K' in seq:
                        sequences.append(seq)
            else:  # 纯序列，每行一条
                for line in raw.splitlines():
                    seq = line.strip().upper()
                    if len(seq) == 49 and 'K' in seq:
                        sequences.append(seq)

            if not sequences:
                result_text.insert(tk.END, "未找到有效的49-mer序列（必须包含中心K）\n")
                return

            result_text.insert(tk.END, f"✅ 检测到 {len(sequences)} 条有效序列，正在提取特征...\n\n")

            # 特征提取 + 预测
            X = extract_full_features(sequences)
            y_pred, probs = predict_sequence(X)

            label_map = ["乙酰化(A)", "巴豆酰化(C)", "甲基化(M)", "琥珀酰化(S)"]
            results = []

            result_text.insert(tk.END, "=== 预测结果 ===\n\n")
            for i in range(len(y_pred)):
                active = np.where(y_pred[i] == 1)[0]
                mods = [label_map[j] for j in active]
                mod_str = " + ".join(mods) if mods else "无修饰"
                prob_str = str(probs[i].round(4))

                result_text.insert(tk.END, f"序列 {i + 1}: {mod_str}\n概率: {prob_str}\n\n")

                results.append({
                    "序列编号": i + 1,
                    "原始序列": sequences[i],
                    "预测修饰类型": mod_str,
                    "预测概率": prob_str
                })

            # 保存全局结果用于导出Excel
            global results_df
            results_df = pd.DataFrame(results)

            result_text.insert(tk.END, "预测完成！可点击下方按钮保存为Excel文件。\n")

        except Exception as e:
            result_text.insert(tk.END, f"❌ 预测出错: {str(e)}\n")

    # 统一按钮行
    tk.Button(btn_frame, text="加载文件", font=("Microsoft YaHei", 11), width=12,
              command=load_file).pack(side=tk.LEFT, padx=8)

    tk.Button(btn_frame, text="开始预测", font=("Microsoft YaHei", 12, "bold"),
              bg="#4CAF50", fg="white", width=12, command=predict).pack(side=tk.LEFT, padx=8)

    tk.Button(btn_frame, text="清空输入", font=("Microsoft YaHei", 11), width=12,
              command=lambda: input_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=8)

    tk.Button(btn_frame, text="保存为Excel", font=("Microsoft YaHei", 11, "bold"),
              bg="#2196F3", fg="white", width=15,
              command=lambda: save_to_excel(results_df) if 'results_df' in globals() else None).pack(side=tk.LEFT,
                                                                                                     padx=8)

    root.mainloop()


if __name__ == "__main__":
    main()