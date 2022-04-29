import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
import os
import hashlib 
import errno

def binarize_class(y):  # 独热编码 y: self.train_labels
    label = y.reshape(len(y), -1)  # torch.Size([60000]) -> torch.Size([60000,1])
    enc = OneHotEncoder(categories='auto') 
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)      # 自己学习自己编码 对自己编码 变成数组 （60000，10）因为只有一列 即一个特征 然后取值0-9 所以用10维表示
    label = torch.from_numpy(label) # 再变成Tensor torch.Size([60000, 10])
    return label


def partialize(y, y0, t, p):
    new_y = y.clone()  # 返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯 torch.Size([60000, 10])
    n, c = y.shape[0], y.shape[1]  # n 60000 c = 10
    avgC = 0

    if t=='binomial':
        for i in range(n):
            row = new_y[i, :] 
            row[np.where(np.random.binomial(1, p, c)==1)] = 1 # n = 1 重复伯努利事件 一次实验样本数为1 独立重复实验10次 每次翻转概率是p -> 1*10的数组 （0/1）
            # np.where 找到值为1 的位置索引 将row 对应位置变为1  构成候选标记集合
            while torch.sum(row) == 1:    # 如果没有翻转 随机翻转
                row[np.random.randint(0, c)] = 1
            avgC += torch.sum(row)    # 累加每个样本所有位置为1和  
            new_y[i] = row / torch.sum(row)  # 每个样本 初始化时 均匀权重  (9)

    if t=='pair':
        P = np.eye(c)   # 对角为1 的10*10数组
        for idx in range(0, c-1):
            P[idx, idx], P[idx, idx+1] = 1, p
        P[c-1, c-1], P[c-1, 0] = 1, p
        for i in range(n):
            row = new_y[i, :]  # 每个样本的onehot label
            idx = y0[i]  # 原来对应的label
            row[np.where(np.random.binomial(1, P[idx, :], c)==1)] = 1  # P[idx, :]对应c个概率值 也正好实验c次
            avgC += torch.sum(row)
            new_y[i] = row / torch.sum(row)  # 每个样本 初始化时 均匀权重

    avgC = avgC / n    # self.average_class_label
    return new_y, avgC




def check_integrity(fpath, md5): 
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): 
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    import urllib.request

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        print('Downloading ' + url + ' to ' + fpath)
        urllib.request.urlretrieve(url, fpath) 
