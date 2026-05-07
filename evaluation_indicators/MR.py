import numpy as np


def calculate_MR(Pre_Labels, test_target, j):
    """
    计算第 j 级匹配比率 (MRj)
    逻辑对齐 Metrics.py 中的 calculate_custom_metrics 函数

    参数：
    Pre_Labels：预测标签矩阵 (n_samples, n_classes)，值为 0 或 1
    test_target：实际标签矩阵 (n_samples, n_classes)，值为 0 或 1
    j：当前计算的匹配级别 (1 到 max_label)
    """
    # 转换为 numpy 数组以确保向量化计算的准确性
    Pre_Labels = np.array(Pre_Labels)
    test_target = np.array(test_target)

    # 获取标签总数（分类列数）
    max_tags = test_target.shape[1]

    # 统计每个样本的真实标签数量
    true_counts = np.sum(test_target == 1, axis=1)

    # 统计每个样本中预测对的标签数量（预测与真实的交集）
    match_counts = np.sum((test_target == 1) & (Pre_Labels == 1), axis=1)

    P_j_sum = 0  # 分子累加器
    C_j_sum = 0  # 分母累加器

    # 按照 Metrics.py 的逻辑，对所有 k >= j 的情况进行累加
    for k in range(j, max_tags + 1):
        # 筛选分母：真实标签数量大于等于 k 的样本
        valid_base_samples = (true_counts >= k)

        # 计算分子：在上述样本中，匹配对的数量大于等于 j 的样本数
        P_k = np.sum((match_counts >= j) & valid_base_samples)

        # 计算分母：符合条件的样本总数
        C_k = np.sum(valid_base_samples)

        P_j_sum += P_k
        C_j_sum += C_k

    # 返回比率，若分母为 0 则返回 0.0 以防止报错
    return P_j_sum / C_j_sum if C_j_sum > 0 else 0.0


def calculate_all_MR(Pre_Labels, test_target):
    """
    计算从 MR1 到 MRmax 的所有指标字典
    """
    # 自动获取最大标签维度
    max_label = test_target.shape[1]
    MR_results = {}

    # 遍历所有级别并存储结果
    for j in range(1, max_label + 1):
        MR_results[f"MR{j}"] = calculate_MR(Pre_Labels, test_target, j)

    return MR_results