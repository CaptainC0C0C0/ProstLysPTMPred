import numpy as np


def absolute_false(Pre_Labels, test_target):
    """
    计算绝对错误率

    返回值：
    absolute_False：绝对错误率
    """
    num_instance, num_class = Pre_Labels.shape      # 分类数、样本数
    miss_pairs = np.sum(Pre_Labels != test_target)  # 看有多少个标签是不相等的。 总标签等与 n*4
    absolute_False = miss_pairs / (num_class * num_instance)
    return absolute_False