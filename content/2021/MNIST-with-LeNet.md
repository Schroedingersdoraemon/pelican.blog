---
layout: blog
title: MNIST with LeNet
date: 2021-05-02 00:03:52
tags:
---

# 1. train part

```python
import time
import torch
import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
```


```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081,))
])
```


```python
train_set = MNIST(root = './MNIST',
                  train = True,
                  #download = True,
                  transform = transform)

test_set = MNIST(root = './MNIST',
                train = False,
                transform = transform)
```


```python
train_loader = DataLoader(
    train_set,
    batch_size = 100,
    shuffle = True
)

test_loader = DataLoader(
    test_set,
    batch_size = 100,
    shuffle = True
)
```


```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cal = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(5*5*16, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.cal(x)
```


```python
net = LeNet()
torch.cuda.empty_cache()
net.cuda()
```

```python
# net structure
LeNet(
  (cal): Sequential(
    (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Flatten(start_dim=1, end_dim=-1)
    (7): Linear(in_features=400, out_features=120, bias=True)
    (8): Linear(in_features=120, out_features=84, bias=True)
    (9): Linear(in_features=84, out_features=10, bias=True)
    (10): Softmax(dim=1)
  )
)
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using', device)
if not (device == 'cpu'):
    net.to(device)
    
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
loss_func = nn.CrossEntropyLoss()
```


```python
egimage, eglabel = iter(train_loader).next()

plt.figure()
plt.imshow(egimage[0][0])
plt.colorbar()
plt.grid(True)
plt.show()
```
    
![png](/files/MNIST-with-LeNet/output1.png)
    

```python
start_time = time.time()

epochs = 50
epoch_loss = []
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        if not (device == 'cpu'):
            inputs = inputs.to(device)
            labels = labels.to(device)
        
        y_hats = net(inputs)
        
        loss = loss_func(y_hats, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        avr_loss = running_loss/(i+1)
    epoch_loss.append(avr_loss)
    print('epoch %d, loss: %.3f' % (epoch, avr_loss))
    
end_time = time.time()
print('Finished training, time used: %.3f'%(end_time - start_time))
```

    epoch 0, loss: 2.298
    epoch 5, loss: 1.603
    epoch 10, loss: 1.515
    epoch 15, loss: 1.500
    epoch 20, loss: 1.493
    epoch 25, loss: 1.489
    epoch 30, loss: 1.486
    epoch 35, loss: 1.484
    epoch 40, loss: 1.482
    epoch 45, loss: 1.481
    epoch 49, loss: 1.480
    Finished training, time used: 551.340



```python
plt.plot(epoch_loss)
```
    
![png](/files/MNIST-with-LeNet/output2.png)
    

```python
model_path = './MNIST_model.pkl'
torch.save(net.state_dict(), model_path)

# net.load_state_dict(torch.load(model_path))
```

# 2. test part

```python
import torch
import numpy as np
from mnist_net import LeNet
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt
```

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081,))
])
```

```python
test_set = MNIST(root='./MNIST',
                train=False,
                transform=transform)
```

```python
test_loader = DataLoader(
    test_set,
    batch_size = 100,
    shuffle = True
)
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using: ',device)

net = LeNet()
# torch.cuda.empty_cache()
net.cuda()

net.load_state_dict(torch.load('./MNIST_model.pkl'))
if not (device == 'cpu'):
    net.to(device)
```

Using:  cuda


```python
total_ac = 0
for i, (inputs, labels) in enumerate(test_loader):
    if not (device == 'cpu'):
        inputs = inputs.to(device)
        labels = labels.to(device)
    
    y_hats = net(inputs)
    y_pred = y_hats.argmax(axis = 1)
    
    ac = (y_pred == labels).sum()
    total_ac += ac
    
    if i==9:
        break

print('Average accuracy of %d tests: %.1f'%((i+1), total_ac/(i+1)))
```

```python
# output
Average accuracy of 10 tests: 98.3
```