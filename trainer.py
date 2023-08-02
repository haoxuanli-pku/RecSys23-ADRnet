import numpy as np
import torch
import const


class MyTrainer:
    def __init__(self, model, inputTrain, outputCol, loss_func=torch.nn.BCELoss(),
                 num_epoch=200, lr=0.001, lamb=1.5e-6, tol=1e-4):
        self.model = model
        self.inputTrain = inputTrain
        self.outputCol = outputCol
        self.num_epoch = num_epoch
        self.loss_func = loss_func
        self.lr = lr
        self.lamb = lamb
        self.tol = tol

    def train(self):
        # para = list(self.model.parameters())

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lamb)
        early_stop = 0
        last_loss = 1e9

        # training loops
        for epoch in range(self.num_epoch):
            optimizer.zero_grad()

            # 训练结果
            drug_features = torch.from_numpy(self.inputTrain).cuda(const.CUDA_DEVICE).long()
            out = self.model(drug_features)

            loss = self.loss_func(out, self.outputCol)
            loss.backward()
            optimizer.step()

            if (last_loss - loss) / last_loss < self.tol:
                if early_stop > 5:
                    print("epoch in train:{}, xent:{}".format(epoch, loss))
                    break
                early_stop += 1

            last_loss = loss
            #
            # if epoch % 50 == 0:
            #     print("epoch:{}, xent:{}".format(epoch, loss))
