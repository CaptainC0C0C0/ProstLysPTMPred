from imblearn.under_sampling import NeighbourhoodCleaningRule
import numpy as np


def neighbourhood_cleaning_rule_resample(data, labels, sampling_strategy, min_samples_per_class):
    """
    使用 NeighbourhoodCleaningRule 算法进行欠采样，并强制保留每类最少样本数。
    参数:
    - data: 特征矩阵，类型为 numpy.ndarray
    - labels: 标签向量，类型为 numpy.ndarray
    - sampling_strategy: NeighbourhoodCleaningRule 的采样策略。
    - min_samples_per_class: 一个字典，定义了每类至少需要保留的样本数。
    返回:
    - resampled_data: 经过处理后的特征矩阵
    - resampled_labels: 经过处理后的标签向量
    """
    # 初始化 NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule(sampling_strategy=sampling_strategy)
    cleaned_data, cleaned_labels = ncr.fit_resample(data, labels)

    # 强制检查每类的保留样本数量
    resampled_data = []
    resampled_labels = []
    for label, min_samples in min_samples_per_class.items():
        # 筛选清理后的数据
        class_data = cleaned_data[cleaned_labels == label]
        class_labels = cleaned_labels[cleaned_labels == label]

        # 如果清理后数据不足，补充原始数据
        if len(class_data) < min_samples:
            # 从原始数据中提取该类的样本
            original_class_data = data[labels == label]
            additional_needed = min_samples - len(class_data)

            # 随机补充不足的样本
            if additional_needed > 0:
                indices = np.random.choice(len(original_class_data), additional_needed, replace=True)
                class_data = np.vstack([class_data, original_class_data[indices]])
                class_labels = np.hstack([class_labels, np.full(additional_needed, label)])

        # 保存结果
        resampled_data.append(class_data)
        resampled_labels.append(class_labels)

    # 合并所有类
    resampled_data = np.vstack(resampled_data)
    resampled_labels = np.hstack(resampled_labels)

    return resampled_data, resampled_labels


# # 示例数据
# data = np.array([[1, 2], [2, 2], [2, 4], [4, 5], [5, 6], [6, 7]])
# labels = np.array([0, 2, 1, 1, 1, 1])
#
# # 自定义清理策略和最小保留样本数
# sampling_strategy = "majority"
# min_samples_per_class = {0: 1, 1: 2, 2: 1}
#
# # 调用函数
# resampled_data, resampled_labels = neighbourhood_cleaning_rule_resample(
#     data, labels, sampling_strategy, min_samples_per_class
# )
#
# # 输出结果
# print("Resampled Data:\n", resampled_data)
# print("Resampled Labels:\n", resampled_labels)
