---
layout: blog
title: CIFAR-10 with GoogLeNet
date: 2021-05-16 11:32:45
tags:
---

# 1. train part

```python
import PIL
import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
```


```python
!wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

--2021-05-16 11:43:20--  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Loaded CA certificate '/etc/ssl/certs/ca-certificates.crt'

Resolving www.cs.toronto.edu (www.cs.toronto.edu)... 128.100.3.30

Connecting to www.cs.toronto.edu (www.cs.toronto.edu)|128.100.3.30|:443... connected.

HTTP request sent, awaiting response... 200 OK

Length: 170498071 (163M) [application/x-gzip]

Saving to: ‘cifar-10-python.tar.gz’

cifar-10-python.tar 100%[===================>] 162.60M  3.42MB/s    in 50s     

2021-05-16 11:44:11 (3.27 MB/s) - ‘cifar-10-python.tar.gz’ saved [170498071/170498071]
    


```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5),
                         std = (0.5, 0.5, 0.5))
])
```


```python
train_set = CIFAR10(
    root = './cifar-10',
    train = True,
    # download = True,
    transform = transform
)
```


```python
train_loader = DataLoader(
    train_set,
    batch_size = 100,
    shuffle = True
)
```


```python
class Inception_A(nn.Module):
    def __init__(self, in_channels):
        super(Inception_A, self).__init__()
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, 1)
        self.branch3x3_2 = nn.Conv2d(16, 24, 3, padding = 1)
        self.branch3x3_3 = nn.Conv2d(24, 24, 3, padding = 1)
        
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, 1)
        self.branch5x5_2 = nn.Conv2d(16, 24, 5, padding = 2)
        
        self.branch1x1 = nn.Conv2d(in_channels, 16, 1)
        
        self.branch_pool = nn.Conv2d(in_channels, 24, 1)
    
    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        branch1x1 = self.branch1x1(x)
        
        branch_pool = F.avg_pool2d(x, 3, 1, 1)
        branch_pool = self.branch_pool(branch_pool)
        
        output = [branch1x1, branch5x5, branch3x3, branch_pool]
        
        return torch.cat(output, dim = 1)
```


```python
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.incep1 = Inception_A(in_channels=10)
        
        self.conv2 = nn.Conv2d(88, 20, 5)
        self.incep2 = Inception_A(in_channels=20)
        
        self.mp = nn.MaxPool2d(2)
        # 88*5*5 = 2200
        self.fc = nn.Linear(2200, 10)
        self.cls = nn.Softmax(dim = 1)
    
    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = self.cls(x)
        return x
```


```python
net = GoogLeNet()
torch.cuda.empty_cache()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
if not (device == 'cpu'):
    net.to(device)
```


```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.005, momentum= 0.2)
```


```python
egimage, eglabel = iter(train_loader).next()

print(egimage.size())
print(eglabel.size())
```

```python
# print result
torch.Size([100, 3, 32, 32])
torch.Size([100])
```



```python
egindex = 4

plt.figure()
plt.imshow(egimage[egindex][0])
plt.colorbar()
plt.grid()
plt.show()

print(classes[eglabel[egindex]])
```

cat

![cat](/files/CIFAR10-with-GoogLeNet/output1.png)


```python
start_time = time.time()
epochs = 50
epoch_loss = []

for epoch in range(epochs):
    running_loss = 0
    for i, (inputs, labels) in enumerate(train_loader):
        if not (device == 'cpu'):
            inputs = inputs.to(device)
            labels = labels.to(device)
        y_hats = net(inputs)
        
        loss = criterion(y_hats, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        avr_loss = running_loss / (i+1)
        
    epoch_loss.append(avr_loss)
    if epoch%5 == 0:
        print('epoch %d, loss: %.5f'%(epoch, avr_loss))

end_time = time.time()
print('Training finished, time used %.3f s'%(end_time - start_time))
```

```python
epoch 0, loss: 1.81920
epoch 5, loss: 1.81351
epoch 10, loss: 1.80854
epoch 15, loss: 1.80460
epoch 20, loss: 1.80021
epoch 25, loss: 1.79726
epoch 30, loss: 1.79350
epoch 35, loss: 1.78997
epoch 40, loss: 1.78629
epoch 45, loss: 1.78374
Training finished, time used 1777.526 s
```

```python
plt.plot(epoch_loss)
```

[<matplotlib.lines.Line2D at 0x7fb5db595d30>]


![png](/files/CIFAR10-with-GoogLeNet/output2.png)

```python
torch.save(net.state_dict(), './GoogLeNet_weights_0005_02.pkl')
```


```python
ac = 0
total = 0 
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        if not (device =='cpu'):
            inputs = inputs.to(device)
            labels = labels.to(device)
        y_hats = net(inputs)
        y_pred = y_hats.argmax(dim = 1)
        ac += (y_pred == labels).sum().item()
        total += labels.size()[0]

print('accuracy: ', ac/total)
```

# 2. test part

```python
import torch
from torchvision import transforms
from GoogLeNet_model import GoogLeNet
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
```


```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5),
                        std = (0.5, 0.5, 0.5))
])
```


```python
test_set  = CIFAR10(
    root = './cifar-10/',
    train = False,
    transform = transform
)
```


```python
test_loader = DataLoader(
    test_set,
    batch_size = 100,
    shuffle = True,
)
```


```python
net = GoogLeNet()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print('Using: ', device)
if not (device == 'cpu'):
    net.to(device)
```

Using:  cuda



```python
net.load_state_dict(torch.load('./GoogLeNet_weights_0005_02.pkl'))
```

<All keys matched successfully>

```python
ac = 0
total = 0 
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        if not (device =='cpu'):
            inputs = inputs.to(device)
            labels = labels.to(device)
        y_hats = net(inputs)
        y_pred = y_hats.argmax(dim = 1)
        ac += (y_pred == labels).sum().item()
        total += labels.size()[0]

print('accuracy: ', ac/total)
```

```python
accuracy:  0.6253
```
