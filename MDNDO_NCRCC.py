import os
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans


# ====================== 标签转换函数 ======================
def single_to_multilabel(y):
    label_mapping = {
        1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 3: [0, 0, 1, 0], 4: [0, 0, 0, 1],
        5: [1, 1, 0, 0], 6: [1, 0, 1, 0], 7: [1, 0, 0, 1], 8: [0, 1, 1, 0],
        9: [1, 1, 1, 0], 10: [1, 1, 0, 1], 11: [1, 1, 1, 1]
    }
    return np.array([label_mapping[label] for label in y], dtype=np.float32)


def multilabel_to_single(y):
    reverse_mapping = {
        (1, 0, 0, 0): 1, (0, 1, 0, 0): 2, (0, 0, 1, 0): 3, (0, 0, 0, 1): 4,
        (1, 1, 0, 0): 5, (1, 0, 1, 0): 6, (1, 0, 0, 1): 7, (0, 1, 1, 0): 8,
        (1, 1, 1, 0): 9, (1, 1, 0, 1): 10, (1, 1, 1, 1): 11
    }
    return np.array([reverse_mapping[tuple(row.astype(int))] for row in y])


def print_distribution(name, single_labels):
    unique, counts = np.unique(single_labels, return_counts=True)
    display_labels = unique + 1 if np.min(single_labels) == 0 else unique
    print(f"{name} 类别分布: {dict(zip(display_labels, counts))}")


# ====================== 1. NCR 下采样 ======================
def run_ncr():
    print("=== Step 1: 执行 NCR 下采样 ===")
    from Under.NCR import neighbourhood_cleaning_rule_resample

    # 加载融合特征
    base = "Fused_ProstT5_MLPSTAAP_Physchem"
    X_train = np.load(f'{base}/X_train_fused.npy')
    y_train_multi = np.load(f'{base}/y_train_labels.npy')
    X_test = np.load(f'{base}/X_test_fused.npy')
    y_test_multi = np.load(f'{base}/y_test_labels.npy')

    y_train_single = multilabel_to_single(y_train_multi)

    # NCR 目标样本数
    target_ncr = [1180, 710, 600, 454, 561, 251, 360, 88, 153, 454, 73]

    original_shape = X_train.shape
    X_flat = X_train.reshape(original_shape[0], -1)

    min_samples_per_class = {i: target_ncr[i] for i in range(11)}

    X_res_flat, y_res_single = neighbourhood_cleaning_rule_resample(
        X_flat,
        y_train_single - 1,
        sampling_strategy='all',
        min_samples_per_class=min_samples_per_class
    )

    X_ncr = X_res_flat.reshape(-1, *original_shape[1:])
    y_ncr_multi = single_to_multilabel(y_res_single + 1)

    print_distribution("NCR 后训练集", y_res_single)

    # 保存 NCR 中间结果
    os.makedirs("Under_NCR", exist_ok=True)
    np.save('Under_NCR/X_train_features_3d.npy', X_ncr)
    np.save('Under_NCR/y_train_labels.npy', y_ncr_multi)
    np.save('Under_NCR/X_test_features_3d.npy', X_test)
    np.save('Under_NCR/y_test_labels.npy', y_test_multi)

    return X_ncr, y_ncr_multi


# ====================== 2. Cluster Centroids 下采样 ======================
def run_cc(X_ncr, y_ncr_multi):
    print("\n=== Step 2: 执行 Cluster Centroids 下采样 ===")

    from Under.ClusterCentroids import Cluster_Centroids

    y_single = multilabel_to_single(y_ncr_multi) - 1  # 转为 0-10
    X_flat = X_ncr.reshape(X_ncr.shape[0], -1)

    # CC 目标样本数
    target_cc = [1180, 1775, 1800, 1816, 1683, 1506, 1800, 1320, 1530, 1816, 1460]
    sampling_strategy = {i: target_cc[i] for i in range(11)}

    X_res_flat, y_res_single = Cluster_Centroids(
        X_flat,
        y_single.reshape(-1, 1),
        sampling_strategy,
        n_init=3,
        voting='hard'
    )

    # 恢复 3D 形状
    original_shape = X_ncr.shape[1:]
    X_cc = X_res_flat.reshape(-1, *original_shape)
    y_cc_multi = single_to_multilabel(y_res_single.ravel() + 1)

    print_distribution("Cluster Centroids 后训练集", y_res_single.ravel())

    # 保存 CC 中间结果
    os.makedirs("Under_NCRCC_1180", exist_ok=True)
    np.save('Under_NCRCC_1180/X_train_features_3d.npy', X_cc)
    np.save('Under_NCRCC_1180/y_train_labels.npy', y_cc_multi)
    np.save('Under_NCRCC_1180/X_test_features_3d.npy', np.load('Under_NCR/X_test_features_3d.npy'))
    np.save('Under_NCRCC_1180/y_test_labels.npy', np.load('Under_NCR/y_test_labels.npy'))

    return X_cc, y_cc_multi


# ====================== 3. MDNDO 上采样 ======================
def generate_gaussian_samples(sample, num_samples):
    mean_vector = sample
    standard_deviation = np.sqrt(0.05) * np.abs(sample)
    Z = np.random.normal(0.0, 1.0, size=(num_samples, mean_vector.shape[0]))
    return mean_vector + standard_deviation * Z


def run_mdnndo():
    print("\n=== Step 3: 执行 MDNDO 上采样 ===")
    np.random.seed(42)

    X_train_3d = np.load('Under_NCRCC_1180/X_train_features_3d.npy')
    y_train_multi = np.load('Under_NCRCC_1180/y_train_labels.npy')
    X_test_3d = np.load('Under_NCRCC_1180/X_test_features_3d.npy')
    y_test_multi = np.load('Under_NCRCC_1180/y_test_labels.npy')

    y_single = multilabel_to_single(y_train_multi) - 1
    X_flat = X_train_3d.reshape(X_train_3d.shape[0], -1)

    target_samples = [1180] * 11

    X_res_list = [row for row in X_flat]
    y_res_list = [label for label in y_single]

    original_counts = Counter(y_single)

    for cls_idx in range(11):
        target = target_samples[cls_idx]
        orig = original_counts.get(cls_idx, 0)
        if target <= orig:
            continue
        n_needed = target - orig
        X_cls = X_flat[y_single == cls_idx]

        print(f"类别 {cls_idx + 1} 上采样: {orig} → {target} (生成 {n_needed} 个)")

        for _ in range(n_needed):
            idx = np.random.randint(0, len(X_cls))
            new_sample = generate_gaussian_samples(X_cls[idx], 1)[0]
            X_res_list.append(new_sample)
            y_res_list.append(cls_idx)

    X_final_3d = np.array(X_res_list).reshape(-1, *X_train_3d.shape[1:])
    y_final_multi = single_to_multilabel(np.array(y_res_list) + 1)

    print_distribution("最终 MDNDO 平衡后", np.array(y_res_list))

    # 保存最终结果
    save_dir = "Balanced_MDNDO_NCR_CC"
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/X_train_features.npy', X_final_3d)
    np.save(f'{save_dir}/y_train_labels.npy', y_final_multi)
    np.save(f'{save_dir}/X_test_features.npy', X_test_3d)
    np.save(f'{save_dir}/y_test_labels.npy', y_test_multi)

    print(f"\n=== 全流程完成！最终结果保存在 {save_dir} 文件夹 ===")


# ====================== 主流程 ======================
if __name__ == "__main__":
    print("开始执行 MDNDO-NCR-CC 全流程采样平衡...\n")

    X_ncr, y_ncr = run_ncr()
    X_cc, y_cc = run_cc(X_ncr, y_ncr)
    run_mdnndo()

    print("\n所有步骤执行完毕！")