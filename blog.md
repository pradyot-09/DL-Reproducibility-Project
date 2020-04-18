# The Reproduction of “Distilling the Knowledge in a Neural Network”


![Neural Networks](https://img.shields.io/badge/Neural%20Networks-Distillation-brightgreen)
![Reproducibility](https://img.shields.io/badge/Reproducibility-Blog-blue)

## Introduction

The aim of this project was to reproduce results from section 3 of [1]. In addition, we compared the efficiency of the Feed Forward Network used by the authors with that of a Convolutional Neural Network.

Distillation is a technique aiming to transfer the knowledge acquired by a cumbersome model (teacher) to a simpler one (student): very complex models could be computationally expensive to deploy in a production environment, hence the advantage of having a smaller and lighter model with the same knowledge.

We will start by describing how we structured our teacher and student models and how did we train the latter to mimic the former. Then we will describe our experiments and results, comparing these to the authors achievements.

## Methodology

As we said, we aim to distilling the knowledge of a complex model (teacher) into a simpler one (student).

We jittered the images from the MNIST dataset training set by 2 pixels in every direction, to add some noise and make the learning more robust and less prone to overfitting.

The approach used is the following: we train the teacher model on our training set, trying to achieve the best accuracy we can on the test set, and then we train the student model to mimic the teacher on the training set in order to achieve more or less the same result.

On a more specific level:
* The teacher model consists of a deep Feed Forward Network with two hidden layers with 1200 rectified linear units. The model is strongly regularized by adding a dropout of 0.2 on the input and of 0.5 after the first hidden layer. The teacher model is trained to match the label of each input by using Cross Entropy Loss and Stochastic Gradient Descent.

* The student model also consists of a deep Feed Forward Network with two hidden layers, but with 800 rectified linear units. This model is not regularized in any way. The student is trained using a linear combination of two losses (and SGD):

  1. Cross Entropy Loss, used to match the input labels.
  
  2. KL divergence, to match the softmax values produced by the teacher model with a high temperature T: [INSERT SOFTMAX FORMULA HERE]. As shown in [2], the KL divergence can be used to build an alternative definition of the Cross Entropy Loss.
  
  The best results have been obtained by giving more relevance to the teacher mimicking part of the loss.
  
## Code

We implemented our reproduction code in Python, and the following libraries are required:

```python
# Import pytorch basic functions/classes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import torchvision functions/classes for MNIST import and data loaders
import torchvision
import torchvision.transforms as transforms

# Other imports
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
```

Now we can start by defining the model for our two Feed Forward Networks (the number of hidden units can be set through the constructor method):

```python
# Define MLP model and its layers
class Model(nn.Module):

    def __init__(self, hidden_size=1200, dropout=0.0, hidden_dropout=0.0):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden1 = nn.Linear(784, hidden_size, bias=True)
        self.hidden1_dropout = nn.Dropout(hidden_dropout)
        self.hidden2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.hidden2_dropout = nn.Dropout(hidden_dropout)
        self.hidden3 = nn.Linear(hidden_size, 10, bias=True)

    def forward(self, x):

        x = self.dropout(x)
        x = F.relu(self.hidden1(x))
        x = self.hidden1_dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.hidden2_dropout(x)
        x = self.hidden3(x)
        return x
```
The custom loss function for the student model will be:

```python
def student_loss(outputs, labels, teacher_outputs, alpha, temperature):

    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss
```

We can then import the MNIST training and test set. We apply jittering of at most two pixels to the training set by using the RandomAffine transformation:

```python
# Define transform from PIL image to tensor and normalize to 1x768 pixels
train_transform = transforms.Compose([
  transforms.RandomAffine(0, (1/14, 1/14)),
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])

# Set batch size for data loaders
batch_size = 128

# (Down)load training set
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# (Down)load test set
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# (Down)load training set without a specific digit
trainset_noDigit = torchvision.datasets.MNIST(root='./data', train=True, download=True)

#   Set here the digit to exclude
idx = trainset_noDigit.train_labels!=3

trainset_noDigit.targets = trainset_noDigit.targets[idx]
trainset_noDigit.data = trainset_noDigit.data[idx]
trainset_noDigit.transform = train_transform

trainloader_noDigit = torch.utils.data.DataLoader(trainset_noDigit, batch_size=batch_size, shuffle=True, num_workers=2)
```

And now we can train our teacher model:

```python
# Setup model and move it to the GPU
net = Model(dropout=0.2, hidden_dropout=0.5)
net.to(device)

# Set up loss function and optimizer: 
#     using cross entropy loss because it's better for classification task

learning_rate = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr= learning_rate, momentum=0.9)

# Run over 100 epochs (1 epoch = visited all items in dataset)
for epoch in range(1000):

    running_loss = 0.0
    total = 0

    if(epoch%100 == 0 and epoch != 0):

      learning_rate = learning_rate * 0.5 #- (0.001) # or maybe decrease by (learning_rate * 0.1)
      optimizer = optim.SGD(net.parameters(), lr= learning_rate, momentum=0.9)

    for i, data in enumerate(trainloader, 0):
        

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1).to(device)

        # This for not cross entropy
        #target = convert_labels(labels).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        target = labels.to(device).long()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        total += len(data)

        # print statistics
        running_loss += loss.item()
    # print every epoch
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / total))

print('Finished Training')

# Save model after having finished training
PATH = './mnist_dropout_100_epoch.pth'
torch.save(net.state_dict(), PATH)

print('Saved Model')
```

And checking its performance on the test set:

```python
# Instantiate model and load saved network parameters
net = Model().to(device)
net.load_state_dict(torch.load(PATH))

# Run model on test set and determine accuracy
correct = 0
total = 0
wrong = np.zeros((10,10))

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1).to(device)
        target = convert_labels(labels).to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        _, target = torch.max(target.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        for i, val in enumerate(predicted):
          if val != target[i]:
            wrong[target[i]][val] += 1

# Output model accuracy to user
print('Accuracy of the network on the 10000 test images: %d %% (%d wrong out of %d)' % (
    100 * correct / total, total - correct, total))

# Plot confusion matrix
df_cm = pd.DataFrame(wrong, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
```

Now, once we have reached a good performance on the test set, we can use the teacher model to train the student:

```python
# Setup student model and move it to the GPU
student_net = Model(hidden_size = 800)
student_net.to(device)

# Set up loss function and optimizer

optimizer = optim.SGD(student_net.parameters(), lr=0.001, momentum=0.9)

# Run over 100 epochs (1 epoch = visited all items in dataset)
for epoch in range(1000):
    running_loss = 0.0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1).to(device)
        target = labels.to(device).long()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # Set temperature and the weights for losses linear combination
        w = 0.7
        T = 20

        # Compute soft labels using deep teacher model previously trained
        outputs_teacher = net(inputs)

        # Student forward + backward + optimize
        outputs_stud = student_net(inputs)
        
        loss = student_loss(outputs_stud, target, outputs_teacher, w, T)
        loss.backward()
        optimizer.step()

        total += len(data)

        # print statistics
        running_loss += loss.item()
    # print every epoch
    print('[%d] loss: %.3f' % (epoch + 1, running_loss / total))

print('Finished Training')

# Save model after having finished training
STUD_PATH = './mnist_student_100_epoch.pth'
torch.save(student_net.state_dict(), STUD_PATH)

print('Saved Model')
```

And finally we can check the accuracy of the student on the test set:

```python
stud_net = Model(hidden_size = 800).to(device)
stud_net.load_state_dict(torch.load(STUD_PATH))

# Run model on test set and determine accuracy
correct = 0
total = 0
wrong = np.zeros((10,10))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs = torch.flatten(inputs, start_dim=1).to(device)
        target = convert_labels(labels).to(device)
        outputs = stud_net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        _, target = torch.max(target.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        for i, val in enumerate(predicted):
          if val != target[i]:
            wrong[target[i]][val] += 1

# Output model accuracy to user
print('Accuracy of the network on the 10000 test images: %d %% (%d wrong out of %d)' % (
    100 * correct / total, total - correct, total))

# Plot confusion matrix
df_cm = pd.DataFrame(wrong, index = [i for i in "0123456789"],
                  columns = [i for i in "0123456789"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
```

## Experiment Setup


## Results


## References

[1] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).
