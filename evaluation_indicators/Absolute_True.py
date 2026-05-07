import numpy as np


def absolute_true(Pre_Labels, test_target):
    """
    返回值：
    absolute_True：子集准确率
    """
    num_instance, num_class = Pre_Labels.shape              # 分类数、样本数
    temp = 0
    for i in range(num_instance):
        if np.array_equal(Pre_Labels[i], test_target[i]):   # 如果预测值和真实值完全一样
            temp += 1
    absolute_True = temp / num_instance
    return absolute_True