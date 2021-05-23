import torch.optim as optim
import torch.nn as nn

class optimizer():
    def optimizer(self, model):
        optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        # 아래 내용 해결해야함
        # then decreased by a factor of 10 when the validation set accuracy stopped improving
        # In total, the learning rate was decreased 3 times, and the learning was stopped after 370K iterations (74 epochs)

    def criterion():
        criterion = nn.CrossEntropyLoss().cuda()
        return criterion