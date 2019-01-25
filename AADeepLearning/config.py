def get_default_config():
    # 默认配置文件
    default_config = {
        # 初始学习率
        "learning_rate": 0.001,
        # 优化策略: sgd/momentum/rmsprop/adam
        "optimizer": "adam",
        # 如果想使用添加动量的梯度下降算法做优化,需要设置这一项，通常设置为 0.9/0.95 即可，一般不需要调整
        "momentum_coefficient": 0.9,
        # rmsprop优化器的衰减系数
        "rmsprop_decay": 0.95,
        # 训练多少次
        "number_iteration": 2000,
        # 每次用多少个样本训练
        "batch_size": 64,
        # 迭代多少次打印一次信息
        "display": 100,
        # 保存模型快照的名称
        "save_model": "",
        # 每隔几个迭代周期保存一次快照？
        "save_iteration": 100,
        # 预训练参数模型所在路径
        "load_model": ""
    }
    return default_config