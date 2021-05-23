import torch

from vgg16.model.vgg16 import vgg16
from vgg16.optimizer.optimizer import optimizer
import vgg16.dataset.dataloader as dataloader

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('used device : ', device)

    model = vgg16()
    model = model.to(device)
    print(model)

    opt = optimizer.optimizer(model)
    ce = optimizer.criterion()

    loss = 0.0

    trainloader, testloader = dataloader()

    for epoch in range(3):

        model.train()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)
            loss = ce(outputs, labels)

            opt.zero_grad()
            loss.backwords()
            opt.step()

            if i % 2000 == 0:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, loss))


if __name__ == '__main__':
    main()