---
layout: blog
title: Dog Cat Dual Classification
date: 2021-04-25 10:18:57
tags:
---

```python
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
```


```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([128, 128]),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_folder = ImageFolder(root='./cat_dog/training_set',
                           transform = transform)
test_folder = ImageFolder(root='./cat_dog/test_set',
                          transform = transform)

train_loader = DataLoader(train_folder,
                         batch_size = 100,
                         shuffle = True)
test_loader = DataLoader(test_folder,
                         batch_size = 100,
                         shuffle = True)
```


```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cal = nn.Sequential(
            nn.Conv2d(in_channels = 3,
                      out_channels = 8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,
                      bias = False),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8,
                      out_channels = 8,
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,
                      bias = False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 2, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*64*8, 1000),
            nn.ReLU(),
            #nn.Linear(1000, 1000),
            #nn.ReLU(),
            nn.Linear(1000, 2),
            nn.Softmax(dim = 1),
        )

    def forward(self, x):
        return self.cal(x)
```


```python
net = Net().cuda()
torch.cuda.empty_cache()
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using: ', device)
if not (device == 'cpu'):
    net.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
loss_func = nn.CrossEntropyLoss()
```

```python
epochs = 50
epoch_loss = []

for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # inputs [100, 3, 128, 128]
        # labels [100]
        if not (device == 'cpu'):
            inputs, labels = data[0].to(device),data[1].to(device)
        else:
            inputs, labels = data

        # y_hats [100, 2]
        y_hats = net(inputs)

        # labels [100, 1]
        # labels = labels.unsqueeze(1).type_as(y_hats)

        loss = loss_func(y_hats, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss.append(running_loss / (i+1))
    print('epoch %d, loss: %.3f' % (epoch, running_loss/(i+1)))
```

    epoch 0, loss: 0.670
    epoch 1, loss: 0.664
    epoch 2, loss: 0.659
    epoch 3, loss: 0.655
    epoch 4, loss: 0.647
    epoch 5, loss: 0.642
    epoch 6, loss: 0.638
    epoch 7, loss: 0.631
    epoch 8, loss: 0.624
    epoch 9, loss: 0.621
    epoch 10, loss: 0.611
    epoch 11, loss: 0.605
    epoch 12, loss: 0.600
    epoch 13, loss: 0.600
    epoch 14, loss: 0.595
    epoch 15, loss: 0.590
    epoch 16, loss: 0.590
    epoch 17, loss: 0.586
    epoch 18, loss: 0.581
    epoch 19, loss: 0.576
    epoch 20, loss: 0.574
    epoch 21, loss: 0.569
    epoch 22, loss: 0.566
    epoch 23, loss: 0.565
    epoch 24, loss: 0.595
    epoch 25, loss: 0.554
    epoch 26, loss: 0.556
    epoch 27, loss: 0.546
    epoch 28, loss: 0.553
    epoch 29, loss: 0.544
    epoch 30, loss: 0.540
    epoch 31, loss: 0.538
    epoch 32, loss: 0.546
    epoch 33, loss: 0.530
    epoch 34, loss: 0.523
    epoch 35, loss: 0.525
    epoch 36, loss: 0.523
    epoch 37, loss: 0.516
    epoch 38, loss: 0.518
    epoch 39, loss: 0.505
    epoch 40, loss: 0.514
    epoch 41, loss: 0.503
    epoch 42, loss: 0.509
    epoch 43, loss: 0.498
    epoch 44, loss: 0.507
    epoch 45, loss: 0.491
    epoch 46, loss: 0.501
    epoch 47, loss: 0.480
    epoch 48, loss: 0.494
    epoch 49, loss: 0.478



```python
plt.plot(epoch_loss)
```



    
![png](/files/Dog-Cat-Dual-Classification/output1.png)
    



```python
# save the model
model_path = './caog_net.pkl'
torch.save(net.state_dict(), model_path)

# load the model
net.load_state_dict(torch.load(model_path))
```
