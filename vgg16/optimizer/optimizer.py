import torch.optim as optim

class Optimizer():
    def __init__(self, model):
        self.model = model

    def optimizer(self):
        return optim.Adam(self.model.parameters(), lr=0.00001)
        # ImageNet 데이터가 아닌 STL-10데이터를 사용함에 따른 optimizer 변경(성능이 안나옴)
        #return optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)