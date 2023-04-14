---
layout: blog
title: Data Classfication
date: 2021-04-18 13:16:17
tags:
---

```python
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
```

```python
data = torch.ones(100, 2)
x0 = torch.normal(data*2, 1)
x1 = torch.normal(data*-2, 1)

x = torch.cat([x0, x1], 0)#.type(torch.FloatTensor)
y0 = torch.zeros(100)
y1 = torch.ones(100)
y = torch.cat([y0, y1], 0).type(torch.LongTensor)
```

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(2, 15),
            nn.ReLU(),
            nn.Linear(15, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        classification = self.classify(x)
        return classification
```

```python
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.3)
loss_func = nn.CrossEntropyLoss()
```

```python
for epoch in range(100):
    y_hat = net(x)
    loss = loss_func(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

classification = torch.max(y_hat, 1)[1]
class_y = classification.data.numpy()
target_y = y.data.numpy()

plt.scatter(x.data.numpy()[:,0],
            x.data.numpy()[:,1],
           c = class_y,
           s = 100,
           cmap = 'RdYlGn',
           )
accuracy = sum(class_y == target_y)/200
# plt.title('accuracy = %s'%accuracy)
plt.text(1.5,
        -4,
        f'accuracy = {accuracy}',
        fontdict = {'size':'20',
                    'color':'blue'}
        )
plt.show()
```
