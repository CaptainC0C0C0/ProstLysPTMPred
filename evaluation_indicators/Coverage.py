def coverage(Pre_Labels, test_target):
    """
    计算覆盖率

    参数：
    Pre_Labels：分类器预测的标签，如果第i个实例属于第j个类别，则 Pre_Labels[j,i]=1，否则 Pre_Labels[j,i]=-1
    test_target：测试实例的实际标签，如果第i个实例属于第j个类别，则 test_target[j,i]=1，否则 test_target[j,i]=-1

    返回值：
    coverage：覆盖率
    """
    num_instance, num_class = Pre_Labels.shape  # 分类数、样本数
    temp = 0
    for i in range(num_instance):
        size_y = 0        # 真实
        size_z = 0        # 预测
        intersection = 0
        for j in range(num_class):
            if Pre_Labels[i][j] == 1:
                size_z += 1
            if test_target[i][j] == 1:
                size_y += 1
            if Pre_Labels[i][j] == 1 and test_target[i][j] == 1:
                intersection += 1
        if size_y != 0:
            temp += intersection / size_y  # 正确的/真实的
    coverage = temp / num_instance
    return coverage