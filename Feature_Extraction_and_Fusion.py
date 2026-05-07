import os
import numpy as np
import re
import torch
from transformers import T5EncoderModel, T5Tokenizer

# ================== 理化性质表 ==================
AA_PHYSCHEM = {
    'A': [89.09, 6.01, 1], 'C': [121.13, 5.07, -1], 'D': [133.10, 2.77, 1],
    'E': [147.13, 3.22, 1], 'F': [165.19, 5.48, -1], 'G': [75.07, 5.97, 1],
    'H': [155.13, 7.59, -1], 'I': [131.18, 6.02, -1], 'K': [146.19, 9.74, 1],
    'L': [131.18, 5.98, -1], 'M': [149.21, 5.74, -1], 'N': [132.12, 5.41, 1],
    'P': [115.13, 6.48, 1], 'Q': [146.15, 5.65, 1], 'R': [174.20, 10.76, 1],
    'S': [105.09, 5.68, 1], 'T': [119.12, 5.87, 1], 'V': [117.15, 5.97, -1],
    'W': [204.23, 5.89, -1], 'Y': [181.19, 5.66, -1]
}


# ================== MLPSTAAP ==================
def compute_mlpstaap(sequences, y_single):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    tri_list = [a + b + c for a in AA for b in AA for c in AA]  # 8000
    tri_to_idx = {tri: i for i, tri in enumerate(tri_list)}

    classes = np.unique(y_single)
    Ft = np.zeros((len(classes), 8000, 46))

    for c_idx, cls in enumerate(classes):
        seqs_cls = [seq for seq, y in zip(sequences, y_single) if y == cls]
        for seq in seqs_cls:
            for j in range(46):
                tri = seq[j:j + 3]
                if tri in tri_to_idx:
                    Ft[c_idx, tri_to_idx[tri], j] += 1
        Ft[c_idx] /= max(1, len(seqs_cls))

    # 计算全局 Fr 矩阵
    Fr = np.zeros((8000, 46))
    for c_idx in range(len(classes)):
        F_mean = Ft[c_idx]
        FFk = (np.sum(Ft, axis=0) - F_mean) / max(1, len(classes) - 1)
        Fr += (F_mean - FFk)
    Fr /= len(classes)

    # 生成训练特征
    features = np.zeros((len(sequences), 46))
    for i, seq in enumerate(sequences):
        for j in range(46):
            tri = seq[j:j + 3]
            features[i, j] = Fr[tri_to_idx.get(tri, 0), j]

    return features, Fr


# ================== 数据加载 ==================
def load_fasta_data(directory):
    """从指定目录加载所有 fasta txt 文件"""
    sequences = []
    labels = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            match = re.search(r'\((\d+)\)', filename)
            if not match:
                continue
            label = int(match.group(1))

            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()

            current_seq = ''
            for line in lines:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        labels.append(label)
                    current_seq = ''
                elif line:
                    current_seq += line.upper()

            if current_seq:
                sequences.append(current_seq)
                labels.append(label)

    return sequences, np.array(labels)


def preprocess_sequence(seq):
    """ProstT5 预处理"""
    sanitized = re.sub(r'[UZOB]', 'X', seq.upper())
    return " ".join(list(sanitized))


# ================== ProstT5 特征提取 ==================
def extract_prostt5_features(sequences, model_path="Rostlab/ProstT5", len_seq=49, batch_size=32):
    """使用 Hugging Face 直接下载 ProstT5 提取特征"""
    print("加载 ProstT5 模型（Hugging Face）...")
    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        do_lower_case=False,
        legacy=True
    )
    model = T5EncoderModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    preprocessed = [preprocess_sequence(seq) for seq in sequences]

    features = []
    for i in range(0, len(preprocessed), batch_size):
        batch = preprocessed[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            padding="max_length",
            max_length=len_seq + 2,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # 保留 [batch, 49, 1024]
        last_hidden = outputs.last_hidden_state
        seq_embeddings = last_hidden[:, 1:1 + len_seq, :]
        features.append(seq_embeddings.cpu().numpy())

        print(f"已处理 {min(i + batch_size, len(preprocessed))}/{len(preprocessed)} 条序列")

    return np.concatenate(features, axis=0)


# ================== 理化性质特征 ==================
def compute_physchem(sequences):
    f = np.zeros((len(sequences), 147))
    for i, s in enumerate(sequences):
        for j, aa in enumerate(s[:49]):  # 限定长度
            f[i, j * 3:(j + 1) * 3] = AA_PHYSCHEM.get(aa, [0, 0, 0])
    return f


# ================== 主融合流程 ==================
def fuse():
    train_dir = "../Train Dataset"
    test_dir = "../Test Dataset"

    # 加载序列和标签
    print("加载训练集序列...")
    seq_train, y_single_train = load_fasta_data(train_dir)
    print("加载测试集序列...")
    seq_test, y_single_test = load_fasta_data(test_dir)

    # 多标签转换
    def single_to_multilabel(y):
        label_mapping = {
            1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0], 4: [0, 0, 0, 1],
            5: [1, 1, 0, 0], 6: [1, 0, 1, 0], 7: [1, 0, 0, 1], 8: [0, 1, 1, 0],
            9: [1, 1, 1, 0], 10: [1, 1, 0, 1], 11: [1, 1, 1, 1]
        }
        return np.array([label_mapping.get(label, [0, 0, 0, 0]) for label in y], dtype=np.float32)

    y_train_multi = single_to_multilabel(y_single_train)
    y_test_multi = single_to_multilabel(y_single_test)

    # ProstT5 特征提取
    print("\n提取 ProstT5 训练集特征...")
    X_train_prost = extract_prostt5_features(seq_train)
    print("提取 ProstT5 测试集特征...")
    X_test_prost = extract_prostt5_features(seq_test)

    X_train_pooled = np.mean(X_train_prost, axis=1)
    X_test_pooled = np.mean(X_test_prost, axis=1)

    # MLPSTAAP 特征
    print("\n计算 MLPSTAAP 特征...")
    mlp_train, Fr = compute_mlpstaap(seq_train, y_single_train)
    mlp_test, _ = compute_mlpstaap(seq_test, y_single_test)

    # 理化性质特征
    print("计算理化性质特征...")
    phys_train = compute_physchem(seq_train)
    phys_test = compute_physchem(seq_test)

    # 特征融合
    X_train_fused = np.hstack([X_train_pooled, mlp_train, phys_train])
    X_test_fused = np.hstack([X_test_pooled, mlp_test, phys_test])

    # 保存结果
    save_dir = "Fused_ProstT5_MLPSTAAP_Physchem"
    os.makedirs(save_dir, exist_ok=True)

    np.save(f"{save_dir}/X_train_fused.npy", X_train_fused)
    np.save(f"{save_dir}/X_test_fused.npy", X_test_fused)
    np.save(f"{save_dir}/y_train_labels.npy", y_train_multi)
    np.save(f"{save_dir}/y_test_labels.npy", y_test_multi)
    np.save(f"{save_dir}/Fr_matrix.npy", Fr)
    np.save(f"{save_dir}/X_train_prost_3d.npy", X_train_prost)  # 可选保存三维特征
    np.save(f"{save_dir}/X_test_prost_3d.npy", X_test_prost)

    print(f"\n融合完成！最终特征维度 = {X_train_fused.shape[1]} (1024 + 46 + 147)")
    print(f"文件已保存至文件夹：{save_dir}")
    print(f"重要：已保存 Fr_matrix.npy（预测时必需）")


if __name__ == "__main__":
    fuse()