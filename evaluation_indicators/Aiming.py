def aiming(Pre_Labels, test_target):
    """
    计算精度
    参数：
    Pre_Labels：分类器的预测标签，如果第i 个实例属于第j个类别，则Pre_Labels[j,i]=1，否则 Pre_Labels[j,i]=-1
    test_target：测试实例的实际标签，如果第i个实例属于第j个类别，则 test_target[j,i]=1，否则 test_target[j,i]=-1
    步骤：
    1首先，计算预测标签 Pre_Labels 和测试实际标签 test_target 的大小（即类别数量和实例数量）。
    2然后，对于每个实例，分别计算预测标签和测试实际标签中值为 1 的数量，并计算它们的交集数量。
    3接着，计算每个实例的精度，即交集数量除以预测标签中值为 1 的数量，这样可以得到每个实例的精度。
    4最后，将所有实例的精度求和并除以实例数量，得到整体的精度。
    返回值：
    aiming：精度
    """
    num_instance, num_class = Pre_Labels.shape  # 分类数、样本数
    temp = 0
    for i in range(num_instance):  # 遍历样本
        size_y = 0      # 真实
        size_z = 0      # 预测
        intersection = 0
        for j in range(num_class):
            # print(Pre_Labels[i][j])
            if Pre_Labels[i][j] == 1:
                size_z += 1
            if test_target[i][j] == 1:
                size_y += 1
            if Pre_Labels[i][j] == 1 and test_target[i][j] == 1:
                intersection += 1
        if size_z != 0:
            # print("size_z", size_z)
            # print("size_y", size_y)
            # print("intersection", intersection)
            temp += intersection / size_z  # 正确的/预测的
    aiming = temp / num_instance
    return aiming