import torch
from torchvision import datasets, transforms
import numpy as np
import math

data_dir = '../data/cifar/'
apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
data = datasets.CIFAR10(data_dir, train=True, download=True,
                                           transform=apply_transform)

def build_non_iid_by_dirichlet(
    random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    n_auxi_workers = 10 # 每次划分n_auxi_workers个用户
    assert n_auxi_workers <= n_workers

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        # 拥有最少样本数量的用户其样本数必须多于平均的一半(0.50 * _targets_size / _n_workers)，否则就重新划分
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):   # 遍历每个类别
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]  # 找到_class类别在_targets中对应的索引
                idx_class = _targets[idx_class, 0]  # 找到_class在数据集中对应的索引

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    # 如果某个client已经划分的样本数量len(idx_j)大于每个用户平均拥有的样本数_targets_size / _n_workers，则本类
                    # 就不再进行给其划分(p * False = 0)
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]   # 此时proportions中记录的是切分点
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


data_size = len(data)
indices = np.array([x for x in range(0, data_size)])

random_state = np.random.RandomState(6)
alpha = 1
num_classes = len(np.unique(data.targets))
num_client = 21
num_indices = len(indices)
indices2targets=np.array(
                    [
                        (idx, target)
                        for idx, target in enumerate(data.targets) # 给每一个label编号
                        if idx in indices
                    ]
                )

res = build_non_iid_by_dirichlet(random_state, indices2targets, alpha, num_classes, num_indices, num_client)

print(res)
print(len(res))
print(len(res[0]))
Sum = 0
for i in range(len(res)):
    Sum += len(res[i])
print(Sum)