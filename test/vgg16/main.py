import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam

import time

from vgg16.model.vgg16 import vgg16
from vgg16.optimizer.optimizer import Optimizer
from vgg16.dataset.dataloader import Dataloader

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('used device : ', device)

    model = vgg16()
    optimizer = Optimizer.optimizer(model)

    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    print(model)

    #optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    #optimizer = Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss().cuda()

    dataloader = Dataloader()
    trainloader = dataloader.trainloader

    start_time = time.time()
    for epoch in range(20):

        model.train()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print('loss :', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, loss.item()))

    print('time :', time.time() - start_time)
    print('Finished Training')

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    testloader = dataloader.testloader

    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = torch.squeeze((predicted == labels))
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            dataloader.classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main()
