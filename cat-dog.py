import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

train_datadir = './datasets/train/'
test_datadir  = './datasets/val/' 


train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomRotation(degrees=(-10, 10)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  
    transforms.ToTensor(),          
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize([224, 224]),  
    transforms.ToTensor(),          
    transforms.Normalize(           
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])  
])

train_data = datasets.ImageFolder(train_datadir,transform=train_transforms)

test_data  = datasets.ImageFolder(test_datadir,transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=64,                                        
                                          shuffle=True,
                                          num_workers=1)
test_loader  = torch.utils.data.DataLoader(test_data,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=1)

if __name__ == '__main__':
    for X, y in test_loader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    def im_convert(tensor):
        image = tensor.to("cpu").clone().detach()        
        image = image.numpy().squeeze()        
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)
        return image

    fig=plt.figure(figsize=(20, 20))
    columns = 2
    rows = 2

    dataiter = iter(train_loader)
    inputs, classes = next(dataiter)

    for idx in range (columns*rows):
        ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
        if classes[idx] == 0:
            ax.set_title("cat", fontsize = 35)
        else:
            ax.set_title("dog", fontsize = 35)
        plt.imshow(im_convert(inputs[idx]))
    plt.savefig('pic1.jpg', dpi=600)
    plt.show()

    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # #LeNet
    # class LeNet(nn.Module):
    #     def __init__(self):
    #         super(LeNet, self).__init__()
    #         self.conv1 = nn.Conv2d(3, 6, 5)
    #         self.conv2 = nn.Conv2d(6, 16, 5)
    #         self.fc1 = nn.Linear(16*53*53, 120)
    #         self.fc2 = nn.Linear(120, 84)
    #         self.fc3 = nn.Linear(84, 2)
    #         self.pool = nn.MaxPool2d(2, 2)
    #     def forward(self, x):
    #         x = F.relu(self.conv1(x))
    #         x = self.pool(x)
    #         x = F.relu(self.conv2(x))
    #         x = self.pool(x)
    #         x = x.view(-1, 16*53*53)
    #         x = F.relu(self.fc1(x))
    #         x = F.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x

    # model = LeNet().to(device)
    # print(model)

    #ResNet18 Pretrained
    from torchvision import models
    model = models.resnet18(pretrained=True)  
    model.fc = nn.Linear(model.fc.in_features, 2)  
    model = model.to(device)
    print(model)

    to_csv_data = []

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn, epoch):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        to_csv_data.append({
            'Epoch': epoch+1,
            'Accuracy': f'{100 * correct:.1f}%',
            'Avg loss': f'{test_loss:.8f}'
        })

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn, t)
    
    
    df = pd.DataFrame(to_csv_data)
    df.to_csv('result.csv', index=False)

    print("Done!")

    #Save Model
    torch.save(model, "model.pth")
    #Load Model
    # model = torch.load("model.pth")