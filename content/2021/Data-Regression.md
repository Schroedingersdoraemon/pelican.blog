---
layout: blog
title: Data Regression
date: 2021-04-11 12:58:42
tags:
---

Regress to a sine function.

```python
#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt
```

```python
x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 100), dim=1)
y = torch.sin(x) + 0.5*torch.rand(x.size())
```

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    def forward(self, x):
        pred = self.predict(x)
        return pred
```

```python
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()
```

```python
plt.ion()
with plt.ion():
    fig = plt.figure()
    for epoch in range(1000):
        y_pred = net(x)
        loss = loss_func(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            plt.scatter(x.detach().numpy(), y_pred.detach().numpy())
            plt.plot(x.detach().numpy(), y.detach().numpy())
plt.show()
```

```python
for epoch in range(10000):
    plt.ion()
    y_pred = net(x)
    loss = loss_func(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%5000==0:
        plt.title('%s  traing result' % int(epoch/1000))
        plt.plot(x.detach().numpy(), y_pred.detach().numpy())
        plt.scatter(x.detach().numpy(), y.detach().numpy())
        plt.legend()
        plt.ioff()
    plt.show()
```