def get_dataloader(dataset_name, data_root, batch_size, train_num_workers=0, transforms=None, val_num_workers=0):
    """
    获取数据dataloader
    :param dataset_name: 数据集名称
    :param data_root: 数据集根目录
    :param batch_size: 批次大小
    :param train_num_workers: 训练使用的worker数
    :param transforms: 预处理函数
    :param val_num_workers: 测试时使用的worker数
    :return: 数据加载器
    """

    if dataset_name == "cifar10":
        pass
    elif dataset_name == "celeb":
        pass
    elif dataset_name == "mnist":
        pass
    elif dataset_name == "edge2shoes":
        from utils.edge2shoes_dataloader import Edge2ShoesDataLoader
        return Edge2ShoesDataLoader(data_root, batch_size, train_num_workers, transforms, val_num_workers)
    elif dataset_name == "Mogaoku":
        from utils.Mogaoku_dataloader import Edge2MogaokuDataLoader
        return Edge2MogaokuDataLoader(data_root, batch_size, train_num_workers, transforms, val_num_workers)
    else:
        KeyError("dataset name error {}".format(dataset_name))
