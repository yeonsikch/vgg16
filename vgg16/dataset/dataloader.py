import torch
import torchvision
from torchvision.transforms import transforms


class Dataloader(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataloader).__init__()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = torchvision.datasets.STL10('./data', split='train', download=True,
                                                transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=32*3, shuffle=True)

        testset = torchvision.datasets.STL10('./data', split='test', download=True,
                                               transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=32*3, shuffle=False)

        self.classes = ('airplance', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

    def __len__(self):
        pass

    def __getitem__(self):
        pass
