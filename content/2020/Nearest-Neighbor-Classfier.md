---
layout: blog
title: Nearest Neighbor Classfier
date: 2020-09-26 16:23:18
tags:
---

首先需要[下载cifar-10数据集](http://www.cs.toronto.edu/~kriz/cifar.html)

然后将训练集的 6 个batch合并到一个数组中去，程序如下：

```python
# cifar-10数据集的保存格式， 需要pickle解码
import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin-1')
        return dict

def train_data():
    x = []
    y = []
    for i in range(1, 6):
        batch_file = 'cifar-10-batches-py/data_batch_%d'%(i)
        batch_dict = unpickle(batch_file)
        train_batch = batch_dict['data']
        train_label = batch_dict['labels']
        x.append(train_batch)
        y.append(train_label)
        data = np.concatenate(x)
        label = np.concatenate(y)

    return data, label

def test_data():
    batch_file = 'cifar-10-batches-py/test_batch'
    batch_dict = unpickle(batch_file)
    data = batch_dict['data']
    test_label = batch_dict['labels']
    label = np.array(test_label)

    return data, label
```

将此程序命名为cifar_data_loader，放到同目录下

如此这般，便可以通过如下片段使用：

```python
# 导入上述程序
import cifar_data_loader
''' 调用cifar_data_loader的train_data函数，将训练数据存为train_data,将训练标签存为train_label'''
train_data, train_label = cifar_data_loader.train_data()
''' 调用cifar_data_loader的test_data函数，将训练数据存为test_data,将训练标签存为test_label'''
test_data, test_label = cifar_data_loader.test_data()
```

算法部分如下：

```python
import pickle
import numpy as np
import cifar_data_loader

train_data, train_label = cifar_data_loader.train_data()
test_data, test_label = cifar_data_loader.test_data()

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
        return Ypred

nn = NearestNeighbor()
nn.train(train_data, train_label)
Yte_predict = nn.predict(test_data)
accuracy = np.mean(Yte_predict == test_label)

print('accuracy: %f' % accuracy)
```

花费36分钟，得到结果如下：

```bash
> python NearestNeighbor.py
# accuracy: 0.249200
```

