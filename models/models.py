import time

import numpy as np
import torch
import torch.nn as nn
from sklearn import svm
from sklearn.metrics import average_precision_score, roc_auc_score

import const
import models.deepModel as deepModel
import models.layer as layer
import models.regressionModel as regressionModel
import utils
# from models import lnsm
from trainer import MyTrainer

# from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class Model:
    def __init__(self):
        self.isFitAndPredict = False
        self.name = "General"
        self.repred = ""
        self.model = " "

    def fit(self, intputTrain, outputTrain):
        pass

    def predict(self, input):
        pass

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        pass

    def getInfo(self):
        pass


class RandomModel(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "Random"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        outputs = np.random.choice([0, 1], size=(inputTest.shape[0], outputTrain.shape[1]), p=[0.5, 0.5])
        print(outputs.shape)
        self.repred = np.random.choice([0, 1], size=(outputTrain.shape[0], outputTrain.shape[1]), p=[0.5, 0.5])
        return outputs

    def getInfo(self):
        return "Uniform 0.5 0.5"


class POLY2(Model):

    def __init__(self, field_dims):
        super().__init__()
        self.field_dims = field_dims
        self.name = "POLY2"
        self.xent_func = torch.nn.BCELoss()

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, num_epoch=100, lr=0.001, lamb=1.5e-6,
                      tol=1e-4, ):

        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])

        predicted_matrix = np.zeros((num_testDrug, num_ADR))

        for i in range(num_ADR):
            # printing ADR index
            if i % 10 == 0:
                print("\r%s" % i, end="")

            outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
            outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
            self.model = regressionModel.POLY2Model(self.field_dims).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

            # train
            Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                num_epoch=num_epoch, lr=lr, lamb=lamb, tol=tol)
            Trainer.train()

            # test
            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
            predicted_col = self.model(drug_features_test)
            predicted_col = predicted_col.cpu().detach().numpy()
            predicted_matrix[:, i] = predicted_col

        return predicted_matrix

    def getInfo(self):
        return const.POLY_LAMB


class FM(Model):

    def __init__(self, field_dims):
        super().__init__()
        self.field_dims = field_dims
        self.name = 'FM'
        self.xent_func = torch.nn.BCELoss()

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, num_epoch=100, lr=0.001, lamb=1.5e-6,
                      tol=1e-4, ):

        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])

        predicted_matrix = np.zeros((num_testDrug, num_ADR))

        for i in range(num_ADR):
            # printing ADR index
            if i % 10 == 0:
                print("\r%s" % i, end="")

            outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
            outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
            self.model = regressionModel.FMModel(self.field_dims).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

            # train
            Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                num_epoch=num_epoch, lr=lr, lamb=lamb, tol=tol)
            Trainer.train()

            # test
            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
            predicted_col = self.model(drug_features_test)
            predicted_col = predicted_col.cpu().detach().numpy()
            predicted_matrix[:, i] = predicted_col

        return predicted_matrix

    def getInfo(self):
        return const.FM_LAMB


class FFM(torch.nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.
    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim):
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        super().__init__()
        self.name = 'FFM'
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, num_epoch=100, lr=0.001, lamb=1.5e-6,
                      tol=1e-4, ):

        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])

        predicted_matrix = np.zeros((num_testDrug, num_ADR))

        for i in range(num_ADR):
            # printing ADR index
            if i % 10 == 0:
                print("\r%s" % i, end="")

            outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
            outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
            self.model = regressionModel.FFMModel(self.field_dims, self.embed_dim).cuda(const.CUDA_DEVICE)
            # 为每个ADR建立一个model

            # train
            Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                num_epoch=num_epoch, lr=lr, lamb=lamb, tol=tol)
            Trainer.train()

            # test
            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
            predicted_col = self.model(drug_features_test)
            predicted_col = predicted_col.cpu().detach().numpy()
            predicted_matrix[:, i] = predicted_col

        return predicted_matrix

    def getInfo(self):
        return const.FFM_LAMB


class LR(Model):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims):
        super().__init__()
        self.field_dims = field_dims
        self.name = 'LR'
        self.xent_func = torch.nn.BCELoss()

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, num_epoch=100, lr=0.001, lamb=1.5e-6,
                      tol=1e-4, ):

        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])

        predicted_matrix = np.zeros((num_testDrug, num_ADR))

        for i in range(num_ADR):
            # printing ADR index
            if i % 10 == 0:
                print("\r%s" % i, end="")

            outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
            outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
            self.model = regressionModel.LRModel(self.field_dims).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

            # train
            Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                num_epoch=num_epoch, lr=lr, lamb=lamb, tol=tol)
            Trainer.train()

            # test
            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
            predicted_col = self.model(drug_features_test)
            predicted_col = predicted_col.cpu().detach().numpy()
            predicted_matrix[:, i] = predicted_col

        return predicted_matrix

    def getInfo(self):
        return const.LR_LAMB


# 多分类输出的LR
class LRFC(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims, num_ADR):
        super().__init__()
        # self.linear = torch.nn.Linear(sum(field_dims), 1)
        self.name = 'LRFC'
        self.xent_func = torch.nn.MSELoss()
        self.field_dims = field_dims
        self.linear = layer.FeaturesLinear(field_dims, num_ADR)

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, lr=0.005, lamb=1.5e-6, tol=1e-8, verbose=1):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)

        inputx = torch.from_numpy(inputTrain).float().cuda(const.CUDA_DEVICE)  # tensor版本的训练样本特征
        outputx = torch.from_numpy(outputTrain).float().cuda(const.CUDA_DEVICE)  # tensor版本的训练样本label
        inputTestx = torch.from_numpy(inputTest).float().cuda(const.CUDA_DEVICE)  # tensor版本的测试样本特征

        early_stop = 0
        last_loss = 1e9
        for i in range(1000):
            optimizer.zero_grad()

            # 逻辑回归部分
            out = self.linear(inputx)
            out = torch.sigmoid(out)

            loss = self.xent_func(out, outputx)
            loss.backward()
            optimizer.step()

            if (last_loss - loss) / last_loss < tol:
                if early_stop > 5:
                    break
                early_stop += 1

            last_loss = loss

            if verbose:
                print("epoch:{}, xent:{}".format(i, loss))

            # # 每十个循环打印一次训练auc和测试auc
            # if i % 10 == 0:
            #     out2 = self.linear(inputTestx)  # test out
            #     out2 = torch.sigmoid(out2)
            #
            #     out2 = out2.cpu().detach().numpy()
            #     out = out.cpu().detach().numpy()
            #
            #     print("In Train: ", roc_auc_score(outputTrain.reshape(-1), out.reshape(-1)))
            #     if outputTest is not None:
            #         print("Eval: ", roc_auc_score(outputTest.reshape(-1), out2.reshape(-1)))

        test_out = torch.sigmoid(self.linear(inputTestx))
        test_out = test_out.cpu().detach().numpy()

        return test_out

    def getInfo(self):
        return const.LR_LAMB


class LSPLM(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims, m):
        super().__init__()
        # self.linear = layer.FeaturesLinear(field_dims)
        self.field_dims = field_dims
        self.m = m
        self.name = 'LSPLM'
        self.xent_func = torch.nn.BCELoss()

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, num_epoch=100, lr=0.001, lamb=1.5e-6,
                      tol=1e-4, ):

        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])

        predicted_matrix = np.zeros((num_testDrug, num_ADR))

        for i in range(num_ADR):
            # printing ADR index
            if i % 10 == 0:
                print("\r%s" % i, end="")

            outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
            outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
            self.model = regressionModel.LSPLMModel(self.field_dims, self.m).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

            # train
            Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                num_epoch=num_epoch, lr=lr, lamb=lamb, tol=tol)
            Trainer.train()

            # test
            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
            predicted_col = self.model(drug_features_test)
            predicted_col = predicted_col.cpu().detach().numpy()
            predicted_matrix[:, i] = predicted_col

        return predicted_matrix

    def getInfo(self):
        return const.LSPLM_LAMB


class LogisticModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "LLR"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        import warnings

        from sklearn.exceptions import (ConvergenceWarning,
                                        DataConversionWarning)
        from sklearn.linear_model import LogisticRegression
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

        print(intpuTrain.shape, outputTrain.shape, inputTest.shape)

        def checkOneClass(inp, nSize):  # 检查outputTrain的那一列是否为全1或全0
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        outputs = []
        reps = []
        nTest = inputTest.shape[0]
        print("LR for %s classes" % nClass)
        model = LogisticRegression(C=const.SVM_C, solver='saga')  # C：正则化系数的倒数
        self.model = model
        for i in range(nClass):
            if i % 10 == 0:
                print("\r%s" % i, end="")
            output = outputTrain[:, i]  # 1222 * 1, outputTrain的一列, 一个ADR对应的所有drug
            ar = checkOneClass(output, nTest)
            ar2 = checkOneClass(output, intpuTrain.shape[0])

            # print clf
            if type(ar) == int:

                model.fit(intpuTrain, output)  # 1222 * 881 所有训练药物的所有特征
                output = model.predict_proba(inputTest)[:, 1]  # 136 * 1 所有测试药物与该ADR的关系
                # rep = model.predict_proba(intpuTrain)[:, 1]
            else:
                output = ar
                rep = ar2
            outputs.append(output)
            # reps.append(rep)

        outputs = np.vstack(outputs).transpose()
        # reps = np.vstack(reps).transpose()
        # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), outputs))
        # self.repred = reps

        print(outputs.shape)
        print("\nDone")
        return outputs

    def getInfo(self):
        return "Original LR"


class CF(Model):
    def __init__(self):
        self.name = "CF"
        self.model = ""

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):

        nTrain, nTest = intpuTrain.shape[0], inputTest.shape[0]
        outSize = outputTrain.shape[1]
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimotoScore(inputTest[i], intpuTrain[j])
        # 计算测试集和训练集样本的两两相似度
        print("simMatrix:", simMatrix)

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        # 为测试集中每个样本与训练集中的每个样本相似度进行排序，返回索引

        args = args[:, :const.CF_KNN]  # KNN=60 每个测试数据只保留前六十个相似度最高的样本数据的索引

        testFeatures = []
        for i in range(nTest):  # 将所有测试数据的chemFeatures根据相似度进行加权，相似度越高权重越大
            newF = np.zeros(len(outputTrain[0]), dtype=float)
            matches = args[i]  # 单个测试数据前60相似度的索引
            simScores = simMatrix[i, matches]  # 单个数据的前60相似度值
            ic = -1
            sum = 1e-10
            for j in matches:  # 将simScores(1*60)和outputTrain中对应的60个药物样本数据（60*2707）逐项相乘, 最终得出136*2707的测试矩阵
                ic += 1
                newF += simScores[ic] * outputTrain[j]
                sum += simScores[ic]  # 求simScores的总和
            newF /= sum
            testFeatures.append(newF)

        out = np.vstack(testFeatures)
        # self.repred = np.matmul(chemFeatures, adrFeatures)
        return out

    def getInfo(self):
        return self.model


class MFModel(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=const.N_FEATURE_MF):
        super(MFModel, self).__init__()
        self.name = "MFModel"
        self.repred = ""
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users + 1, self.embedding_k)  # 输入维度+1，否则报错
        self.H = torch.nn.Embedding(self.num_items + 1, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()

    def getInfo(self):
        Info = ""
        return Info

    def Embedding(self, x):
        # input: a batch of index list : [index of drug, index of ADR]

        user_idx = torch.LongTensor(x[:, 0]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of drugs from x
        item_idx = torch.LongTensor(x[:, 1]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of ADRs from x

        U_emb = self.W(user_idx)  # embedding: from batch_size to batch_size * k
        V_emb = self.H(item_idx)  # embedding: from batch_size to batch_size * k

        return U_emb, V_emb

    def forward(self, x):
        # embed, concat and DNN

        U_emb, V_emb = self.Embedding(x)

        dot_vec = torch.mul(U_emb, V_emb)
        dot = torch.sum(dot_vec, dim=1).reshape(-1, 1)

        return dot, U_emb, V_emb

    def fit(self, outputTrain, num_epoch=1000, lr=0.07, lamb=1.5e-5, tol=1e-4, batch_size=16384,
            verbose=1):  # lambda 1e-4 1e-3 , batch_size=4096

        nDrug = len(outputTrain)
        nADR = len(outputTrain[0])

        outputTrain_index = utils.indices_array_generic(nDrug, nADR)

        # An array of output (binary) in order
        outputTrain_value = np.empty(nDrug * nADR)
        for i in range(nDrug):
            for j in range(nADR):
                outputTrain_value[i * nADR + j] = outputTrain[i][j]

        # outputTrain = np.array(outputTrain)
        # outputTrain = torch.from_numpy(outputTrain)
        # outputTrain.cuda(const.CUDA_DEVICE)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(outputTrain_index)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):

            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):  # idx:第几个batch

                # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                sub_x = outputTrain_index[selected_idx]  # batch中drug的特征：batch_size * 1222
                sub_y = outputTrain_value[selected_idx]  # batch中drug的ADR：batch_size * 2707

                optimizer.zero_grad()
                pred, u_emb, v_emb = self.forward(sub_x)

                pred = self.sigmoid(pred)  # prediction value
                # pred=pred.cuda(const.CUDA_DEVICE)
                # pred = pred.to(const.CPU)

                # target = torch.tensor(sub_y).reshape(-1, len(pred))      # reshape the test ADR
                target = torch.unsqueeze(torch.Tensor(sub_y), 1)
                target = target.cuda(const.CUDA_DEVICE)
                # xent_loss = self.xent_func(pred.T.float(), target.float()) # require float

                xent_loss = self.xent_func(pred, target)

                loss = xent_loss  ##

                loss.backward()
                optimizer.step()
                xent_loss_cpu = xent_loss.cpu()
                epoch_loss += xent_loss_cpu.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if verbose:
                print("epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def predict(self, inputTrain, outputTrain, inputTest):

        drug_idx = np.arange(len(outputTrain))
        adr_idx = np.arange(len(outputTrain[0]))

        drug_idx = torch.from_numpy(drug_idx).cuda(const.CUDA_DEVICE)
        adr_idx = torch.from_numpy(adr_idx).cuda(const.CUDA_DEVICE)

        # drug_idx = torch.from_numpy(drug_idx)
        # adr_idx = torch.from_numpy(adr_idx)

        chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity

        ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

        nTrain, nTest = inputTrain.shape[0], inputTest.shape[0]  # 1222, 136
        # outSize = outputTrain.shape[1] # 2707
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)  # 136 * 1222
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimotoScore(inputTest[i], inputTrain[j])
        # 计算测试集和训练集样本的两两相似度
        # print("simMatrix:", simMatrix)

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        # 为测试集中每个样本与训练集中的每个样本相似度进行排序，返回索引

        args = args[:, :const.KNN]  # KNN=60 每个测试数据只保留前六十个相似度最高的样本数据的索引

        u_emb_weighted = []
        for i in range(nTest):  # 求136个测试数据的u_emb，并根据相似度进行加权，相似度越高权重越大
            newF = np.zeros(const.N_FEATURE_MF, dtype=float)
            matches = args[i]  # 单个测试数据前60相似度的索引
            simScores = simMatrix[i, matches]  # 单个数据的前60相似度值
            ic = -1
            sum = 1e-10
            for j in matches:  # 将simScores(1*60)和chemFeatures中对应的60个药物样本数据（60*k）逐项相乘, 最终得出136*k的测试矩阵
                ic += 1
                newF += simScores[ic] * chemFeatures[j].cpu().detach().numpy()
                sum += simScores[ic]  # 求simScores的总和
            newF /= sum
            u_emb_weighted.append(newF)

        u_emb_weighted = np.array(u_emb_weighted)
        u_emb_weighted = torch.from_numpy(u_emb_weighted).cuda(const.CUDA_DEVICE)

        out = torch.mm(u_emb_weighted.float(), ADRFeatures.T.float()).reshape(-1, 1)

        out = self.sigmoid(out)

        return out.cpu().detach().numpy()


class MF(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "MF"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.decomposition import NMF

        self.model = NMF(const.N_FEATURE_MF)  # c值,预设为10，可以调参
        chemFeatures = self.model.fit_transform(outputTrain)
        adrFeatures = self.model.components_

        nTrain, nTest = intpuTrain.shape[0], inputTest.shape[0]
        outSize = outputTrain.shape[1]
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimotoScore(inputTest[i], intpuTrain[j])
        # 计算测试集和训练集样本的两两相似度
        print("simMatrix:", simMatrix)

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        # 为测试集中每个样本与训练集中的每个样本相似度进行排序，返回索引

        args = args[:, :const.MF_KNN]  # KNN=60 每个测试数据只保留前六十个相似度最高的样本数据的索引
        print("args:", args)

        testFeatures = []
        for i in range(nTest):  # 将所有测试数据的chemFeatures根据相似度进行加权，相似度越高权重越大
            newF = np.zeros(const.N_FEATURE_MF, dtype=float)
            matches = args[i]  # 单个测试数据前60相似度的索引
            simScores = simMatrix[i, matches]  # 单个数据的前60相似度值
            ic = -1
            sum = 1e-10
            for j in matches:  # 将simScores(1*60)和chemFeatures中对应的60个药物样本数据（60*10）逐项相乘, 最终得出136*10的测试矩阵
                ic += 1
                newF += simScores[ic] * chemFeatures[j]
                sum += simScores[ic]  # 求simScores的总和
            newF /= sum
            testFeatures.append(newF)

        testVecs = np.vstack(testFeatures)
        self.repred = np.matmul(chemFeatures, adrFeatures)

        out = np.matmul(testVecs, adrFeatures)
        return out

    def getInfo(self):
        return "MF %s" % const.N_FEATURE_MF


class GBDT(Model):
    def __init__(self):
        self.name = "GBDT"
        self.model = ""

    def getInfo(self):
        return self.name

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        import lightgbm as lgb
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])

        predicted_matrix = np.zeros((num_testDrug, num_ADR))
        for i in range(num_ADR):
            # printing ADR index
            if i % 10 == 0:
                print("\r%s" % i, end="")

            outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
            lgb_train = lgb.Dataset(inputTrain, outputTrain_col)
            # lgb_eval = lgb.Dataset(inputTest, outputTest, reference=lgb_train)

            num_leaf = 2
            num_tree = 20

            # specify your configurations as a dict
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': {'binary_logloss'},
                'num_leaves': num_leaf,
                'num_trees': num_tree,
                'learning_rate': 0.01,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                "force_row_wise": True
            }

            # print('Start training...')
            # train
            gbm = lgb.train(params,
                            lgb_train)

            # print('Start predicting...')
            pred = gbm.predict(inputTest, pred_leaf=False)
            # pred = np.sum(pred, axis=1)/num_tree
            predicted_matrix[:, i] = pred

        return predicted_matrix


class GBDTLR(Model):
    def __init__(self):
        self.name = "GBDTLR"
        self.model = ""

    def getInfo(self):
        return self.model

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])

        predicted_matrix = np.zeros((num_testDrug, num_ADR))
        for i in range(num_ADR):
            # printing ADR index
            if i % 10 == 0:
                print("\r%s" % i, end="")

            outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
            # 检查该ADR列是否是全1或者全0
            if np.sum(outputTrain_col) == len(outputTrain_col):
                predicted_matrix[:, i] = np.ones((len(inputTest)))
            elif np.sum(outputTrain_col) == 0:
                predicted_matrix[:, i] = np.zeros((len(inputTest)))
            else:
                lgb_train = lgb.Dataset(inputTrain, outputTrain_col)
                # lgb_eval = lgb.Dataset(inputTest, outputTest, reference=lgb_train)

                num_leaf = 2
                num_tree = 20

                # specify your configurations as a dict
                params = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': {'binary_logloss'},
                    'num_leaves': num_leaf,
                    'num_trees': num_tree,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    "force_row_wise": True,
                }

                # print('Start training...')
                # train
                gbm = lgb.train(params, lgb_train)
                repred = gbm.predict(inputTrain, pred_leaf=True)
                # print('Start predicting...')

                # feature transformation and write result
                # print('Writing transformed training data')
                # 100棵树，每棵树2个叶子，将其二值化为200个单位，模型预测的叶子结点的序号为1，其他叶子序号为0
                transformed_training_matrix = np.zeros([len(repred), num_tree * num_leaf], dtype=np.int64)
                for row in range(0, len(repred)):
                    # temp表示在每棵树上预测的值所在节点的序号（0表示第一颗树的第一个叶子结点，2表示第一棵树的最后一个叶子结点，3表示第二棵树的第一个节点）
                    if type(repred[row]) != np.ndarray:
                        leave_values = np.zeros(num_tree, dtype=np.int64)
                    elif len(np.array(repred[row])) < num_tree:
                        leave_values = np.zeros(num_tree, dtype=np.int64)
                    else:
                        leave_values = repred[row]
                    temp = np.arange(num_tree) * num_leaf + np.array(leave_values)
                    # 构造one-hot 训练数据集
                    transformed_training_matrix[row][temp] += 1

                # predict and get data on leaves, testing data
                pred = gbm.predict(inputTest, pred_leaf=True)

                # feature transformation and write result
                # print('Writing transformed testing data')
                transformed_testing_matrix = np.zeros([len(pred), num_tree * num_leaf], dtype=np.int64)
                for row in range(0, len(pred)):
                    # if len(np.array(pred[row])) < num_tree:
                    if type(pred[row]) != np.ndarray:
                        leave_values = np.zeros(num_tree, dtype=np.int64)
                    elif len(np.array(pred[row])) < num_tree:
                        leave_values = np.zeros(num_tree, dtype=np.int64)
                    else:
                        leave_values = pred[row]
                    temp = np.arange(num_tree) * num_leaf + np.array(leave_values)
                    transformed_testing_matrix[row][temp] += 1

                # print('Calculate feature importances...')
                # feature importances
                # print('Feature importances:', list(gbm.feature_importance()))
                # print('Feature importances:', list(gbm.feature_importance("gain")))

                # Logestic Regression Start
                # print("Logestic Regression Start")

                # load or create your dataset
                # print('Load data...')

                LRModel = regressionModel.LRModel([num_tree * num_leaf]).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

                # train
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                Trainer = MyTrainer(LRModel, transformed_training_matrix, outputTrain_col, loss_func=torch.nn.BCELoss(),
                                    num_epoch=500, lr=0.001, lamb=1e-6, tol=1e-4)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(transformed_testing_matrix).cuda(const.CUDA_DEVICE).long()
                predicted_col = LRModel(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()

                predicted_matrix[:, i] = predicted_col

        return predicted_matrix


class AutoRec(nn.Module):
    def __init__(self, num_items, hidden_units):
        super(AutoRec, self).__init__()
        self.name = 'AutoRec'

        # self.num_users = num_users
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.xent_func = torch.nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items, self.hidden_units),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_units, self.num_items),
            nn.Sigmoid()
        )

    def getInfo(self):
        return

    def forward(self, DrugVec):
        input = torch.from_numpy(DrugVec).cuda(const.CUDA_DEVICE).float()
        h1 = self.encoder(input)
        out = self.decoder(h1)

        return out * 2

    def dataMasking(self, dataset, mask_rate):

        num_ADR = len(dataset[0])
        all_idx = np.arange(num_ADR)  # 每个ADR的index

        for drug in dataset:

            np.random.shuffle(all_idx)
            mask_idx = all_idx[:int(mask_rate * num_ADR)]  # 随机选取部分idx进行mask

            for item_idx in range(len(drug)):
                if item_idx in mask_idx:
                    drug[item_idx] = 1

        return dataset

    def fitAndPredict(self, outputTrain, outputTest, num_epoch=1000, lr=0.05, lamb=1.5e-6, tol=1e-4, batch_size=128,
                      verbose=1):  # lambda 1e-4 1e-3 , batch_size=4096

        outputTrain = outputTrain * 2
        outputTest = outputTest * 2

        # outputTrain_mask = self.dataMasking(outputTrain, 0.3)
        outputTest_mask = self.dataMasking(outputTest, 0.2)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(outputTrain)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):

            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):  # idx:第几个batch

                # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                sub_x = outputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                sub_y = outputTrain[selected_idx]

                optimizer.zero_grad()
                pred = self.forward(sub_x)

                # pred=pred.cuda(const.CUDA_DEVICE)
                # pred = pred.to(const.CPU)

                target = torch.tensor(sub_y).reshape(len(pred), -1)  # reshape the test ADR
                # target = torch.unsqueeze(torch.Tensor(sub_y), 1)
                target = target.float().cuda(const.CUDA_DEVICE)
                # xent_loss = self.xent_func(pred.T.float(), target.float()) # require float

                xent_loss = self.xent_func(pred, target)

                loss = xent_loss  ##

                loss.backward()
                optimizer.step()
                xent_loss_cpu = xent_loss.cpu()
                epoch_loss += xent_loss_cpu.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            if verbose:
                print("epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

        pred_test = self.forward(outputTest_mask)

        return pred_test


class NCF(nn.Module):
    """The neural collaborative filtering method.
    """

    def __init__(self, num_users, num_items, embedding_k=const.N_FEATURE_NCF):
        super(NCF, self).__init__()
        self.name = "NCF"
        self.repred = ""
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users + 1, self.embedding_k)  # 输入维度+1，否则报错
        self.H = torch.nn.Embedding(self.num_items + 1, self.embedding_k)

        self.linear_g = torch.nn.Linear(1, 1)

        self.linear_1 = torch.nn.Linear(self.embedding_k * 2, self.embedding_k)
        self.relu = torch.nn.ReLU()  #
        self.sigmoid = torch.nn.Sigmoid()
        # self.relu = torch.nn.PReLU()
        # self.relu = torch.nn.functional.leaky_relu()

        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.linear_f = torch.nn.Linear(1, 1, bias=False)

        self.xent_func = torch.nn.BCELoss()

    def getInfo(self):
        Info = "NCF: k = " + str(const.N_FEATURE_NCF) + " , layer_1_size = " + str(
            const.LAYER_1_SIZE) + " ,learning_rate = " + str(const.LEARNING_RATE) + " ,lambda = " + str(
            const.LAMB) + " ,KNN = " + str(const.KNN) + " ,tol = " + str(const.TOL)
        return Info

    def getConcatAndDot(self, drug_latent, ADR_latent):
        # drug_latent: number of drugs * k
        # ADR_latent: k * number of ADRs
        # return: (number of drugs * number of ADRs) * 2k

        drug_latent = drug_latent.cuda(const.CUDA_DEVICE)
        ADR_latent = ADR_latent.cuda(const.CUDA_DEVICE)  # 调用Cuda加速效果明显

        num_drug = len(drug_latent)  # How many drugs data input
        num_ADR = len(ADR_latent)  # How many ADRs

        concat_data = torch.Tensor(num_drug * num_ADR, self.embedding_k * 2).cuda(
            const.CUDA_DEVICE)  # (2707*batch_size)  * 2k

        # dot = torch.Tensor(num_drug * num_ADR, 1).cuda(const.CUDA_DEVICE)
        dot = torch.mm(drug_latent.float(), ADR_latent.T.float()).reshape(-1, 1)

        i = 0
        for u in drug_latent:
            uu = u.reshape(1, self.embedding_k)
            for v in ADR_latent:
                vv = v.reshape(1, self.embedding_k)
                concat_data[i] = torch.cat([uu, vv], axis=1)
                # dot[i] = torch.mm(uu.double(), vv.T.double())
                i += 1

        return concat_data, dot

    def Embedding(self, x):
        # input: a batch of index list : [index of drug, index of ADR]

        user_idx = torch.LongTensor(x[:, 0]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of drugs from x
        item_idx = torch.LongTensor(x[:, 1]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of ADRs from x

        U_emb = self.W(user_idx)  # embedding: from batch_size to batch_size * k
        V_emb = self.H(item_idx)  # embedding: from batch_size to batch_size * k

        return U_emb, V_emb

    def GMF(self, dot):

        # dot_vec = torch.mul(U_emb, V_emb)
        # dot = torch.sum(dot_vec, dim=1).reshape(-1,1)
        out = self.linear_g(dot)
        # out = self.sigmoid(out)

        return out

    def DNN(self, inputVec):
        # DNN
        inputVec = inputVec.cuda(const.CUDA_DEVICE)

        h1 = self.linear_1(inputVec)
        out = self.sigmoid(h1)
        # h1 = torch.nn.functional.leaky_relu(h1)
        # h2 = self.linear_2(h1)
        # h2 = self.sigmoid(h2)

        out = self.linear_2(h1)
        return out

    def forward(self, x):
        # embed, concat and DNN

        U_emb, V_emb = self.Embedding(x)

        dot_vec = torch.mul(U_emb, V_emb)
        dot = torch.sum(dot_vec, dim=1).reshape(-1, 1)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        dnn = self.DNN(z_emb)
        gmf = self.GMF(dot)

        # input_f = torch.cat((dnn, gmf), axis=1)
        # out = self.linear_f(gmf)
        out = gmf + dnn
        out = self.sigmoid(out)
        # out = gmf
        # out = dot

        return out, U_emb, V_emb

    def fit(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=20, lr=0.0005, lamb=const.NCF_LAMB, tol=1e-4, batch_size=65536,
            verbose=1):  # lambda 1e-4 1e-3 , batch_size=4096

        nDrug = len(outputTrain)
        nADR = len(outputTrain[0])

        outputTrain_index = utils.indices_array_generic(nDrug, nADR)

        # An array of output (binary) in order
        outputTrain_value = np.empty(nDrug * nADR)
        for i in range(nDrug):
            for j in range(nADR):
                outputTrain_value[i * nADR + j] = outputTrain[i][j]

        # outputTrain = np.array(outputTrain)
        # outputTrain = torch.from_numpy(outputTrain)
        # outputTrain.cuda(const.CUDA_DEVICE)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(outputTrain_index)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):

            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):  # idx:第几个batch

                # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                sub_x = outputTrain_index[selected_idx]  # batch中drug的特征：batch_size * 1222
                sub_y = outputTrain_value[selected_idx]  # batch中drug的ADR：batch_size * 2707

                optimizer.zero_grad()
                pred, u_emb, v_emb = self.forward(sub_x)

                # pred=pred.cuda(const.CUDA_DEVICE)
                # pred = pred.to(const.CPU)

                # target = torch.tensor(sub_y).reshape(-1, len(pred))      # reshape the test ADR
                target = torch.unsqueeze(torch.Tensor(sub_y), 1)
                target = target.cuda(const.CUDA_DEVICE)
                # xent_loss = self.xent_func(pred.T.float(), target.float()) # require float

                xent_loss = self.xent_func(pred, target)

                loss = xent_loss  ##

                loss.backward()
                optimizer.step()
                xent_loss_cpu = xent_loss.cpu()
                epoch_loss += xent_loss_cpu.detach().numpy()

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("Early stop epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
            
        
            drug_idx = np.arange(len(outputTrain))
            adr_idx = np.arange(len(outputTrain[0]))

            drug_idx = torch.from_numpy(drug_idx).cuda(const.CUDA_DEVICE)
            adr_idx = torch.from_numpy(adr_idx).cuda(const.CUDA_DEVICE)

            # drug_idx = torch.from_numpy(drug_idx)
            # adr_idx = torch.from_numpy(adr_idx)

            chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity

            ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

            nTrain, nTest = inputTrain.shape[0], inputTest.shape[0]  # 1222, 136
            # outSize = outputTrain.shape[1] # 2707
            simMatrix = np.ndarray((nTest, nTrain), dtype=float)  # 136 * 1222
            for i in range(nTest):
                for j in range(nTrain):
                    simMatrix[i][j] = utils.getTanimoto(inputTest[i], inputTrain[j])
            # 计算测试集和训练集样本的两两相似度
            # print("simMatrix:", simMatrix)

            args = np.argsort(simMatrix, axis=1)[:, ::-1]
            # 为测试集中每个样本与训练集中的每个样本相似度进行排序，返回索引

            args = args[:, :const.KNN]  # KNN=60 每个测试数据只保留前六十个相似度最高的样本数据的索引

            u_emb_weighted = []
            for i in range(nTest):  # 求136个测试数据的u_emb，并根据相似度进行加权，相似度越高权重越大
                newF = np.zeros(const.N_FEATURE_NCF, dtype=float)
                matches = args[i]  # 单个测试数据前60相似度的索引
                simScores = simMatrix[i, matches]  # 单个数据的前60相似度值
                ic = -1
                sum = 1e-10
                for j in matches:  # 将simScores(1*60)和chemFeatures中对应的60个药物样本数据（60*k）逐项相乘, 最终得出136*k的测试矩阵
                    ic += 1
                    newF += simScores[ic] * chemFeatures[j].cpu().detach().numpy()
                    sum += simScores[ic]  # 求simScores的总和
                newF /= sum
                u_emb_weighted.append(newF)

            u_emb_weighted = np.array(u_emb_weighted)
            u_emb_weighted = torch.from_numpy(u_emb_weighted)
            # u_emb_weighted = u_emb_weighted.cuda(const.CUDA_DEVICE)

            # concat_data, U_emb, V_emb= self.concat(u_emb_weighted, ADRFeatures)  # (136*2707) * 2k, (136*2707) * k, (136*2707) * k
            concat_data, dot = self.getConcatAndDot(u_emb_weighted, ADRFeatures)
            gmf = self.GMF(dot)
            dnn = self.DNN(concat_data)
            out = gmf + dnn
            # out = dot
            out = torch.sigmoid(out)

            testOut_vec = outputTest.reshape(-1)
            predictedValues = out.cpu().detach().numpy()
            pred_vec = predictedValues.reshape(-1)
            auc = roc_auc_score(testOut_vec, pred_vec)
            aupr = average_precision_score(testOut_vec.reshape(-1), predictedValues.reshape(-1))
            print("epoch:", epoch, "test auc:", auc, "test aupr: ", aupr)

            last_loss = epoch_loss

            if verbose:
                print("epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def predict(self, inputTrain, outputTrain, inputTest):
        drug_idx = np.arange(len(outputTrain))
        adr_idx = np.arange(len(outputTrain[0]))

        drug_idx = torch.from_numpy(drug_idx).cuda(const.CUDA_DEVICE)
        adr_idx = torch.from_numpy(adr_idx).cuda(const.CUDA_DEVICE)

        # drug_idx = torch.from_numpy(drug_idx)
        # adr_idx = torch.from_numpy(adr_idx)

        chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity

        ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

        nTrain, nTest = inputTrain.shape[0], inputTest.shape[0]  # 1222, 136
        # outSize = outputTrain.shape[1] # 2707
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)  # 136 * 1222
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimoto(inputTest[i], inputTrain[j])
        # 计算测试集和训练集样本的两两相似度
        # print("simMatrix:", simMatrix)

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        # 为测试集中每个样本与训练集中的每个样本相似度进行排序，返回索引

        args = args[:, :const.KNN]  # KNN=60 每个测试数据只保留前六十个相似度最高的样本数据的索引

        u_emb_weighted = []
        for i in range(nTest):  # 求136个测试数据的u_emb，并根据相似度进行加权，相似度越高权重越大
            newF = np.zeros(const.N_FEATURE_NCF, dtype=float)
            matches = args[i]  # 单个测试数据前60相似度的索引
            simScores = simMatrix[i, matches]  # 单个数据的前60相似度值
            ic = -1
            sum = 1e-10
            for j in matches:  # 将simScores(1*60)和chemFeatures中对应的60个药物样本数据（60*k）逐项相乘, 最终得出136*k的测试矩阵
                ic += 1
                newF += simScores[ic] * chemFeatures[j].cpu().detach().numpy()
                sum += simScores[ic]  # 求simScores的总和
            newF /= sum
            u_emb_weighted.append(newF)

        u_emb_weighted = np.array(u_emb_weighted)
        u_emb_weighted = torch.from_numpy(u_emb_weighted)
        # u_emb_weighted = u_emb_weighted.cuda(const.CUDA_DEVICE)

        # concat_data, U_emb, V_emb= self.concat(u_emb_weighted, ADRFeatures)  # (136*2707) * 2k, (136*2707) * k, (136*2707) * k
        concat_data, dot = self.getConcatAndDot(u_emb_weighted, ADRFeatures)
        gmf = self.GMF(dot)
        dnn = self.DNN(concat_data)
        out = gmf + dnn
        # out = dot
        out = torch.sigmoid(out)

        return out.cpu().detach().numpy()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1, 1)
        pred = self.sigmoid(pred)
        return np.concatenate([1 - pred, pred], axis=1)


class WideAndDeep(Model):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.name = 'WideAndDeep'
        self.xent_func = torch.nn.BCELoss()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=const.LEARNING_RATE,
                      lamb=const.LAMB,
                      tol_0=1e-4, tol_1=1e-4,):
        # print(lamb)
        # print(lr_1)
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])
        batch_size = num_trainDrug

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.WideAndDeepModel(self.field_dims, self.embed_dim, self.mlp_dims,
                                                        self.dropout).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugWideAndDeepModel(self.field_dims, self.embed_dim, self.mlp_dims,
                                                        self.dropout, num_ADR).cuda(const.CUDA_DEVICE)
            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)
            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

            for epoch in range(1000):
                # train

                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)
                epoch_loss = 0

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)

                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss

                if (last_loss - epoch_loss) / last_loss < tol_1:
                    if early_stop > 5:
                        print("Early stop epoch in iter:{}, xent:{}".format(epoch, epoch_loss))
                        break
                    early_stop += 1

                # if epoch % 10 == 0 and epoch != 0:
                #     predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                #     testOut_vec = outputTest.reshape(-1)
                #     pred_vec = predicted_matrix.reshape(-1)
                #     auc = roc_auc_score(testOut_vec, pred_vec)
                #     print("epoch: ", epoch, "test auc:", auc)

                last_loss = epoch_loss

                # print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))
                # test

            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return const.WideAndDeep_LAMB


class DeepAndCross(Model):

    def __init__(self, field_dims, embed_dim, num_layers, mlp_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_dims = mlp_dims
        self.dropout = dropout

        self.name = 'DeepAndCross'
        self.xent_func = torch.nn.BCELoss()

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=0.0005,
                      lamb=1.5e-6,
                      tol_0=1e-4, tol_1=1e-8,):
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])
        batch_size = num_trainDrug//2

        drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.DeepAndCrossModel(self.field_dims, self.embed_dim, self.num_layers,
                                                         self.mlp_dims, self.dropout).cuda(const.CUDA_DEVICE)

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugDeepAndCrossModel(self.field_dims, self.embed_dim, self.num_layers,
                                                         self.mlp_dims, self.dropout, num_ADR).cuda(const.CUDA_DEVICE)
            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)
            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            for epoch in range(300):
                # train

                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)
                epoch_loss = 0

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)

                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss

                if epoch % 10 == 0 and epoch != 0:
                    predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                    testOut_vec = outputTest.reshape(-1)
                    pred_vec = predicted_matrix.reshape(-1)
                    auc = roc_auc_score(testOut_vec, pred_vec)
                    print("epoch: ", epoch, "test auc:", auc)

                if (last_loss - epoch_loss) / last_loss < tol_1:
                    if early_stop > 5:
                        print("Early stop epoch in iter:{}, xent:{}".format(epoch, epoch_loss))
                        break
                    early_stop += 1

                last_loss = epoch_loss
                print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))

                # test

            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return const.DeepAndCross_LAMB


class DNN(Model):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout

        self.name = 'DNN'
        self.xent_func = torch.nn.BCELoss()

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=0.001,
                      lamb=1.5e-6,
                      tol_0=1e-4, tol_1=1e-8,
                      batch_size=611):
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])

        drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.DNNModel(self.field_dims, self.embed_dim,
                                                self.mlp_dims, self.dropout).cuda(const.CUDA_DEVICE)

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugDNNModel(self.field_dims, self.embed_dim,
                                            self.mlp_dims, self.dropout).cuda(const.CUDA_DEVICE)
            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)
            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            for epoch in range(500):
                # train
                epoch_loss = 0
                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)

                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    epoch_loss += loss
                    optimizer.step()

                # if (last_loss - epoch_loss) / last_loss < tol_1:
                #     if early_stop > 5:
                #         print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))
                #         break
                #     early_stop += 1

                last_loss = epoch_loss
                print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))

                if epoch % 10 == 0 and epoch != 0:
                    predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                    testOut_vec = outputTest.reshape(-1)
                    pred_vec = predicted_matrix.reshape(-1)
                    auc = roc_auc_score(testOut_vec, pred_vec)
                    print("epoch: ", epoch, "test auc:", auc)

                # test

            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return " "


class DeepCrossing(Model):

    def __init__(self, field_dims, embed_dim, mru_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mru_dims = mru_dims
        self.dropout = dropout

        self.name = 'DeepCrossing'
        self.xent_func = torch.nn.BCELoss()

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=0.0005,   # 0.0001
                      lamb=1.5e-6,
                      tol_0=1e-4, tol_1=1e-8,):
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])
        batch_size = num_trainDrug // 2

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.DeepCrossingModel(self.field_dims, self.embed_dim,
                                                         self.mru_dims, self.dropout).cuda(const.CUDA_DEVICE)

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugDeepCrossingModel(self.field_dims, self.embed_dim,
                                                     self.mru_dims, self.dropout, num_ADR).cuda(const.CUDA_DEVICE)
            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)
            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

            for epoch in range(400):  # 500 no early stop
                # train

                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)
                epoch_loss = 0

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)
                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss

                if (last_loss - epoch_loss) / last_loss < tol_1:
                    if early_stop > 5:
                        print("Early stop epoch:{}, xent:{}".format(epoch, epoch_loss))
                        break
                    early_stop += 1

                if epoch % 10 == 0 and epoch != 0:
                    predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                    testOut_vec = outputTest.reshape(-1)
                    pred_vec = predicted_matrix.reshape(-1)
                    auc = roc_auc_score(testOut_vec, pred_vec)
                    print("epoch: ", epoch, "test auc:", auc)

                last_loss = epoch_loss

                print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))

                # test

            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return const.DeepCrossing_LAMB


class PNN(Model):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, method='inner'):
        super().__init__()
        self.name = 'PNN'
        self.xent_func = torch.nn.BCELoss()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.method = method

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=0.005,
                      lamb=1.5e-6,
                      tol_0=1e-4, tol_1=1e-8,):
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])
        batch_size = num_trainDrug // 2

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.PNNModel(self.field_dims, self.embed_dim, self.mlp_dims, self.dropout,
                                                self.method).cuda(const.CUDA_DEVICE)

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugPNNModel(self.field_dims, self.embed_dim, self.mlp_dims, self.dropout,
                                                num_ADR, self.method).cuda(const.CUDA_DEVICE)
            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()

            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)
            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            for epoch in range(800):
                # train

                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)
                epoch_loss = 0

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)

                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss

                if (last_loss - epoch_loss) / last_loss < tol_1:
                    if early_stop > 5:
                        print("Early stop epoch:{}, xent:{}".format(epoch, epoch_loss))
                        break
                    early_stop += 1

                if epoch % 10 == 0 and epoch != 0:
                    predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                    testOut_vec = outputTest.reshape(-1)
                    pred_vec = predicted_matrix.reshape(-1)
                    auc = roc_auc_score(testOut_vec, pred_vec)
                    print("epoch: ", epoch, "test auc:", auc)

                last_loss = epoch_loss

                print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))

                # test
            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return const.PNN_LAMB


class FNN(Model):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout

        self.name = 'FNN'
        self.xent_func = torch.nn.BCELoss()

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=0.001,    # AEOLUS 0.005
                      lamb=1.5e-6,
                      tol_0=1e-4, tol_1=1e-8,):
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])
        batch_size = num_trainDrug

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.FNNModel(self.field_dims, self.embed_dim,
                                                self.mlp_dims, self.dropout).cuda(const.CUDA_DEVICE)

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugFNNModel(self.field_dims, self.embed_dim,
                                                self.mlp_dims, self.dropout, num_ADR).cuda(const.CUDA_DEVICE)
            # self.model = nn.DataParallel(self.model, device_ids=const.GPUS, output_device=const.CUDA_DEVICE)

            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()
            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)

            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            for epoch in range(500):

                # train

                # optimizer.zero_grad()
                # drug_features = torch.from_numpy(inputTrain).cuda(const.CUDA_DEVICE).long()
                #
                # out = self.model(drug_features)
                # loss = self.xent_func(out, outputTrain)
                # loss.backward()
                # optimizer.step()
                #
                # print("epoch: ", epoch, "loss:", loss)

                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)
                epoch_loss = 0

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)

                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss

                if (last_loss - epoch_loss) / last_loss < tol_1:
                    if early_stop > 5:
                        print("Early stop epoch in iter:{}, xent:{}".format(epoch, epoch_loss))
                        break
                    early_stop += 1

                if epoch % 10 == 0 and epoch != 0:
                    predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                    testOut_vec = outputTest.reshape(-1)
                    pred_vec = predicted_matrix.reshape(-1)
                    auc = roc_auc_score(testOut_vec, pred_vec)
                    print("epoch: ", epoch, "test auc:", auc)

                last_loss = epoch_loss

                print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))

                # test
            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return const.FNN_LAMB


class NFM(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.name = 'NFM'
        self.xent_func = torch.nn.BCELoss()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropouts = dropouts

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=0.001, # 0.001
                      lamb=1.5e-6,
                      tol_0=1e-4, tol_1=1e-8,):
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])
        batch_size = num_trainDrug//2

        drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.NFMModel(self.field_dims, self.embed_dim, self.mlp_dims,
                                                self.dropouts).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugNFMModel(self.field_dims, self.embed_dim, self.mlp_dims,
                                            self.dropouts, num_ADR).cuda(const.CUDA_DEVICE)

            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)
            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            for epoch in range(150):
                # train

                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)
                epoch_loss = 0

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)

                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss

                if (last_loss - epoch_loss) / last_loss < tol_1:
                    if early_stop > 5:
                        print("Early stop epoch:{}, xent:{}".format(epoch, epoch_loss))
                        break
                    early_stop += 1

                if epoch % 10 == 0 and epoch != 0:
                    predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                    testOut_vec = outputTest.reshape(-1)
                    pred_vec = predicted_matrix.reshape(-1)
                    auc = roc_auc_score(testOut_vec, pred_vec)
                    print("epoch: ", epoch, "test auc:", auc)

                last_loss = epoch_loss

                print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))

                # test

            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return const.NFM_LAMB


class DeepFM(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.fm = layer.FactorizationMachine(reduce_sum=False)
        self.embeddingByFields = layer.FeaturesEmbeddingByFields(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = layer.MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        self.name = 'DeepFM'
        self.xent_func = torch.nn.BCELoss()
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=0.005,
                      lamb=1.5e-6,
                      tol_0=1e-4, tol_1=1e-8,):
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])
        batch_size = num_trainDrug // 2

        drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.DeepFMModel(self.field_dims, self.embed_dim, self.mlp_dims,
                                                   self.dropout).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugDeepFMModel(self.field_dims, self.embed_dim, self.mlp_dims,
                                               self.dropout, num_ADR).cuda(const.CUDA_DEVICE)

            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)
            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            for epoch in range(300):
                # train

                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)
                epoch_loss = 0

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)

                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss

                if epoch % 10 == 0 and epoch != 0:
                    predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                    testOut_vec = outputTest.reshape(-1)
                    pred_vec = predicted_matrix.reshape(-1)
                    auc = roc_auc_score(testOut_vec, pred_vec)
                    print("epoch: ", epoch, "test auc:", auc)

                if (last_loss - epoch_loss) / last_loss < tol_1:
                    if early_stop > 5:
                        print("Early stop epoch:{}, xent:{}".format(epoch, epoch_loss))
                        break
                    early_stop += 1

                last_loss = epoch_loss

                print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))

                # test
            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return const.DeepFM_LAMB


class AFM(torch.nn.Module):
    """
    A pytorch implementation of Attentional Factorization Machine.
    Reference:
        J Xiao, et al. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, 2017.
    """

    def __init__(self, field_dims, embed_dim, attn_size, dropouts):
        super().__init__()

        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.attn_size = attn_size
        self.dropouts = dropouts

        self.xent_func = torch.nn.BCELoss()
        self.name = 'AFM'

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=100,
                      lr_0=0.01, lr_1=0.001,
                      lamb=1.5e-6,
                      tol_0=1e-4, tol_1=1e-8,):
        num_trainDrug = len(inputTrain)
        num_testDrug = len(inputTest)
        num_ADR = len(outputTrain[0])
        batch_size = num_trainDrug // 2

        if const.JOINT == 0:

            predicted_matrix = np.zeros((num_testDrug, num_ADR))

            for i in range(num_ADR):
                # printing ADR index
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                outputTrain_col = outputTrain[:, i]  # 共现矩阵的一列，即一个ADR与所有drug的关系
                outputTrain_col = torch.from_numpy(outputTrain_col).cuda(const.CUDA_DEVICE).float()
                self.model = deepModel.AFMModel(self.field_dims, self.embed_dim, self.attn_size,
                                                self.dropouts).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

                # train
                Trainer = MyTrainer(self.model, inputTrain, outputTrain_col, loss_func=self.xent_func,
                                    num_epoch=num_epoch, lr=lr_0, lamb=lamb, tol=tol_0)
                Trainer.train()

                # test
                drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
                predicted_col = self.model(drug_features_test)
                predicted_col = predicted_col.cpu().detach().numpy()
                predicted_matrix[:, i] = predicted_col

        else:
            self.model = deepModel.DrugAFMModel(self.field_dims, self.embed_dim, self.attn_size,
                                            self.dropouts, num_ADR).cuda(const.CUDA_DEVICE)  # 为每个ADR建立一个model

            outputTrain = torch.from_numpy(outputTrain).cuda(const.CUDA_DEVICE).float()
            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_1, weight_decay=lamb)
            early_stop = 0
            last_loss = 1e9
            total_batch = num_trainDrug // batch_size

            for epoch in range(1000):
                # train

                all_idx = np.arange(num_trainDrug)
                np.random.shuffle(all_idx)
                epoch_loss = 0

                for idx in range(total_batch):  # idx:第几个batch

                    # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                    # mini-batch training
                    selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                    sub_x = inputTrain[selected_idx]  # batch_size个drug在共现矩阵中的行向量 batch_size * 2707
                    sub_y = outputTrain[selected_idx]
                    optimizer.zero_grad()

                    # 训练结果
                    drug_features = torch.from_numpy(sub_x).cuda(const.CUDA_DEVICE).long()
                    out = self.model(drug_features)

                    loss = self.xent_func(out, sub_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss

                if epoch % 10 == 0 and epoch != 0:
                    predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()
                    testOut_vec = outputTest.reshape(-1)
                    pred_vec = predicted_matrix.reshape(-1)
                    auc = roc_auc_score(testOut_vec, pred_vec)
                    print("epoch: ", epoch, "test auc:", auc)

                if (last_loss - epoch_loss) / last_loss < tol_1:
                    if early_stop > 5:
                        print("Early stop epoch:{}, xent:{}".format(epoch, epoch_loss))
                        break
                    early_stop += 1

                # print("loss decrease:{}".format((last_loss - epoch_loss) / last_loss))

                last_loss = epoch_loss

                print("epoch in iter:{}, xent:{}".format(epoch, epoch_loss))


                # test
            drug_features_test = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE).long()
            predicted_matrix = self.model(drug_features_test).cpu().detach().numpy()

        return predicted_matrix

    def getInfo(self):
        return const.AFM_LAMB


class DIN(nn.Module):
    def __init__(self, num_features, cat_features, seq_features,
                 cat_nums, embedding_size, attention_groups,
                 mlp_hidden_layers, mlp_activation='prelu', mlp_dropout=0.0,
                 d_out=1
                 ):
        super().__init__()
        self.num_features = num_features
        self.cat_features = cat_features
        self.seq_features = seq_features
        self.cat_nums = cat_nums
        self.embedding_size = embedding_size

        self.attention_groups = attention_groups

        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_activation = mlp_activation
        self.mlp_dropout = mlp_dropout

        self.d_out = d_out

        # embedding
        self.embeddings = OrderedDict()
        for feature in self.cat_features + self.seq_features:
            self.embeddings[feature] = nn.Embedding(
                self.cat_nums[feature], self.embedding_size, padding_idx=0)
            self.add_module(f"embedding:{feature}", self.embeddings[feature])

        self.sequence_poolings = OrderedDict()
        self.attention_poolings = OrderedDict()
        total_embedding_sizes = 0
        for feature in self.cat_features:
            total_embedding_sizes += self.embedding_size
        for feature in self.seq_features:
            total_embedding_sizes += self.embedding_size

        # sequence_pooling
        for feature in self.seq_features:
            if not self.is_attention_feature(feature):
                self.sequence_poolings[feature] = MaxPooling(1)
                self.add_module(f"pooling:{feature}", self.sequence_poolings[feature])

        # attention_pooling
        for attention_group in self.attention_groups:
            self.attention_poolings[attention_group.name] = (
                self.create_attention_fn(attention_group))
            self.add_module(f"attention_pooling:{attention_group.name}",
                            self.attention_poolings[attention_group.name])

        total_input_size = total_embedding_sizes + len(self.num_features)

        self.mlp = MLP(
            total_input_size,
            mlp_hidden_layers,
            dropout=mlp_dropout, batchnorm=True, activation=mlp_activation)

        self.final_layer = nn.Linear(mlp_hidden_layers[-1], self.d_out)
        self.apply(init_weights)

    def forward(self, x):

        final_layer_inputs = list()

        number_inputs = list()
        for feature in self.num_features:
            number_inputs.append(x[feature].view(-1, 1))

        embeddings = OrderedDict()
        for feature in self.cat_features:
            embeddings[feature] = self.embeddings[feature](x[feature])

        for feature in self.seq_features:
            if not self.is_attention_feature(feature):
                embeddings[feature] = self.sequence_poolings[feature](
                    self.embeddings[feature](x[feature]))

        for attention_group in self.attention_groups:
            query = torch.cat(
                [embeddings[pair['ad']]
                 for pair in attention_group.pairs],
                dim=-1)
            keys = torch.cat(
                [self.embeddings[pair['pos_hist']](
                    x[pair['pos_hist']]) for pair in attention_group.pairs],
                dim=-1)
            # hist_length = torch.sum(hist>0,axis=1)
            keys_length = torch.min(torch.cat(
                [torch.sum(x[pair['pos_hist']] > 0, axis=1).view(-1, 1)
                 for pair in attention_group.pairs],
                dim=-1), dim=-1)[0]

            embeddings[attention_group.name] = self.attention_poolings[
                attention_group.name](query, keys, keys_length)

        emb_concat = torch.cat(number_inputs + [
            emb for emb in embeddings.values()], dim=-1)

        final_layer_inputs = self.mlp(emb_concat)
        output = self.final_layer(final_layer_inputs)
        if self.d_out == 1:
            output = output.squeeze()

        return output

    def create_attention_fn(self, attention_group):
        return Attention(
            attention_group.pairs_count * self.embedding_size,
            hidden_layers=attention_group.hidden_layers,
            dropout=attention_group.att_dropout,
            activation=attention_group.activation)

    def is_attention_feature(self, feature):
        for group in self.attention_groups:
            if group.is_attention_feature(feature):
                return True
        return False


class CCAModel(Model):
    def __init__(self):
        self.isFitAndPredict = False
        from sklearn.cross_decomposition import CCA
        self.model = CCA(n_components=const.CCA)
        self.name = "CCA"

    def fit(self, inputTrain, outputTrain):
        self.model.fit(inputTrain, outputTrain)
        print(self.model.x_weights_.shape)
        print(self.model.y_weights_.shape)
        print(np.sum(np.multiply(self.model.x_weights_, self.model.x_weights_), axis=0))

        def calCanonicalCoefficent():
            px = np.matmul(inputTrain, self.model.x_weights_)
            py = np.matmul(outputTrain, self.model.y_weights_)
            spxy = np.multiply(px, py)
            spxy = np.sum(spxy, axis=0)
            s1 = np.multiply(px, px)
            s1 = np.sum(s1, axis=0)
            s1 = np.sqrt(s1)
            s2 = np.multiply(py, py)
            s2 = np.sum(s2, axis=0)
            s2 = np.sqrt(s2)
            s = np.multiply(s1, s2)
            corr = np.divide(spxy, s)
            return np.diag(corr)

        self.corrmx = calCanonicalCoefficent()

        def eval():
            px = np.matmul(inputTrain, self.model.x_loadings_)
            py = np.matmul(outputTrain, self.model.y_loadings_)
            x = px - py
            x = np.multiply(x, x)
            print(x.shape)
            s = np.sum(x)
            print(s)

        # eval()
        y = self.predict(inputTrain)
        self.repred = y
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), y.reshape(-1)))

    def getInfo(self):
        return self.model

    def predict(self, input):
        # y = np.matmul(input,self.model.x_loadings_)
        # b = pinv(self.model.y_loadings_)
        # y = np.matmul(y,b)

        # y = np.matmul(input,self.model.x_weights_)
        # y = np.matmul(y,self.corrmx)
        # y = np.matmul(y,self.model.y_weights_.transpose())
        # print y.shape
        v = self.model.predict(input)
        # print v.shape
        return v


class RSCCAModel(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "SCCA"

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, ifold):
        from numpy.linalg import pinv
        pathWX = "%s/WeightChem_%s" % (const.RSCCA_DATA_DIR, ifold)
        pathWY = "%s/WeightSE_%s" % (const.RSCCA_DATA_DIR, ifold)

        wx = np.loadtxt(pathWX)
        wy = np.loadtxt(pathWY)

        def cal(input):
            xwx = np.matmul(input, wx)
            xwxwyt = np.matmul(xwx, wy.transpose())
            yyt = np.matmul(wy, wy.transpose())
            invyyt = pinv(yyt)
            pred = np.matmul(xwxwyt, invyyt)
            return pred

        pred = cal(inputTest)
        repred = cal(inputTrain)
        self.repred = repred
        return pred


class CNNModel(Model):
    def __init__(self):
        self.isFitAndPredict = False
        self.name = "CNN"

    def fitAndPredict(self, iFold):
        import random

        from dataProcessor.DataFactory import GenAllData
        from models.cnnCore import CNNCore

        dataLoader = GenAllData()
        datas = dataLoader.loadFold(iFold)
        self.model = CNNCore(dataLoader.N_FEATURE, 5, dataLoader.N_ADRS)
        # self.loss = MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        N_TRAIN_DRUG = len(datas[const.EC_TRAIN_INP_DATA_INDEX])
        # orderTrain = np.arange(0,N_TRAIN_DRUG)
        self.currentTranIdx = 0

        self.TRAIN_ORDER = np.arange(0, N_TRAIN_DRUG)

        random.seed(1)
        random.shuffle(self.TRAIN_ORDER)

        def getSubList(ls, ids):
            subList = []
            for i in ids:
                subList.append(ls[i])
            return subList

        def getNextMiniBatchTrain():

            batchSize = 1  # Dont change this value.
            start = self.currentTranIdx
            end = self.currentTranIdx + batchSize
            if end <= N_TRAIN_DRUG:
                features = getSubList(datas[const.EC_TRAIN_INP_DATA_INDEX], self.TRAIN_ORDER[start:end])
                features = dataLoader.paddingECEPFeatureToNumpyArray(features)
                self.currentTranIdx = end
                outputs = datas[3][self.TRAIN_ORDER[start:end]]
            else:

                end = N_TRAIN_DRUG
                parts1Features = getSubList(datas[const.EC_TRAIN_INP_DATA_INDEX], self.TRAIN_ORDER[start:end])
                parts1Out = datas[const.EC_TRAIN_OUT_DATA_INDEX][self.TRAIN_ORDER[start:end]]

                end = batchSize - (end - start)
                start = 0

                random.shuffle(self.TRAIN_ORDER)
                parts2Features = getSubList(datas[const.EC_TRAIN_INP_DATA_INDEX], self.TRAIN_ORDER[start:end])
                parts2Out = datas[const.EC_TRAIN_OUT_DATA_INDEX][self.TRAIN_ORDER[start:end]]

                features = []
                for f in parts1Features:
                    features.append(f)
                for f in parts2Features:
                    features.append(f)
                features = dataLoader.paddingECEPFeatureToNumpyArray(features)
                outputs = np.vstack([parts1Out, parts2Out])

            self.currentTranIdx = end

            return features, outputs

        for iter in range(const.CNN_MAX_ITER):
            input, output = getNextMiniBatchTrain()
            inputx = torch.unsqueeze(torch.from_numpy(input).float(), dim=1)
            outputx = torch.from_numpy(output).float()

            optimizer.zero_grad()
            out, z = self.model.forward(inputx)
            # err = self.loss(out, outputx)
            err = self.model.getLoss(out, outputx, z, N_TRAIN_DRUG)

            err.backward()
            optimizer.step()

            if iter % 1000 == 0 and iter > 0:
                print("Train shape", inputx.shape)

                # print(err)
                torch.no_grad()
                # Testing...
                outs = []
                print("Test shape", len(datas[const.EC_TEST_INP_DATA_INDEX]))

                for featureTest in datas[const.EC_TEST_INP_DATA_INDEX]:
                    featureTest = dataLoader.paddingECEPFeatureToNumpyArray([featureTest])
                    featureTest = torch.unsqueeze(torch.from_numpy(featureTest).float(), dim=1)
                    out, _ = self.model.forward(featureTest)
                    out = out.detach().numpy()[0]
                    outs.append(out)
                outs = np.vstack(outs)

                print("Eval: ", iter, roc_auc_score(datas[const.EC_TEST_OUT_DATA_INDEX].reshape(-1), outs.reshape(-1)),
                      average_precision_score(datas[const.EC_TEST_OUT_DATA_INDEX].reshape(-1), outs.reshape(-1)))

                torch.enable_grad()
        # Repredict training data
        outTrains = []
        for featureTrain in datas[const.EC_TRAIN_INP_DATA_INDEX]:
            featureTrain = dataLoader.paddingECEPFeatureToNumpyArray([featureTrain])
            featureTrain = torch.unsqueeze(torch.from_numpy(featureTrain).float(), dim=1)
            out, _ = self.model.forward(featureTrain)
            out = out.detach().numpy()[0]
            outTrains.append(out)
        outTrains = np.vstack(outTrains)
        self.repred = outTrains
        return outs


class NeuNModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "NeuN"

    def fitAndPredict(self, input, output, inputTest, outputTest=None):
        from torch.nn.modules.loss import MSELoss
        nInput, dimInput = input.shape
        nOutput, dimOutput = output.shape
        modules = []
        # modules.append(nn.Linear(dimInput, const.NeuN_H1))
        # modules.append(nn.ReLU())
        # modules.append(nn.Linear(const.NeuN_H1, dimOutput))

        modules.append(nn.Linear(dimInput, const.NeuN_H1))
        modules.append(nn.Sigmoid())
        modules.append(nn.Linear(const.NeuN_H1, const.NeuN_H2))
        modules.append(nn.Sigmoid())
        modules.append(nn.Linear(const.NeuN_H2, dimOutput))

        modules.append(nn.Sigmoid())

        # modules.append(nn.Softmax(dim=-1))

        self.model = nn.Sequential(*modules)
        self.loss = MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        outs = []
        inputx = torch.from_numpy(input).float()
        outputx = torch.from_numpy(output).float()
        inputTestx = torch.from_numpy(inputTest).float()
        for i in range(const.NeuIter):
            optimizer.zero_grad()
            out = self.model.forward(inputx)
            err = self.loss(out, outputx)
            err.backward()
            print("epoch: ", i, "loss: ", err)
            optimizer.step()
            if i % 10 == 0:
                out2 = self.model.forward(inputTestx)
                out2 = out2.detach().numpy()
                outs.append(out2)
                out = out.detach().numpy()
                print("In Train: ", roc_auc_score(output.reshape(-1), out.reshape(-1)))
                if outputTest is not None:
                    print("Eval: ", roc_auc_score(outputTest.reshape(-1), out2.reshape(-1)))

        outx = self.model.forward(inputx)
        outx = outx.detach().numpy()
        # print "In Train: ",roc_auc_score(output.reshape(-1),outx.reshape(-1))

        self.repred = outx

        return outs

    def getInfo(self):
        return self.name


class LNSMModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "LNSM"
        self.inputTrain = ""
        self.Y = ""

    def fit(self, input, output):
        self.inputTrain = input
        self.Y = lnsm.learnLNSM(input, output)
        # self.repred = output

    def predict(self, inputTest):
        preds = []
        for v in inputTest:
            w = lnsm.getRowLNSM(v, self.inputTrain, -1)
            pred = np.matmul(w, self.Y)
            preds.append(pred)
        preds = np.vstack(preds)
        return preds

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        self.fit(intpuTrain, outputTrain)
        self.repred = self.predict(intpuTrain)
        return self.predict(inputTest)


class MultiSVM(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "SVM"

    def svmModel(self, i):

        # print (os.getpid(), i)
        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        model2 = svm.SVC(C=const.SVM_C, gamma='auto', kernel='rbf', probability=True)
        output = self.sharedOutputTrain.array[:, i]
        nTest = self.sharedInputTest.array.shape[0]
        ar = checkOneClass(output, nTest)
        ar2 = checkOneClass(output, self.sharedInputTrain.array.shape[0])

        if type(ar) == int:

            model2.fit(self.sharedInputTrain.array, output)
            output = model2.predict_proba(self.sharedInputTest.array)[:, 1]
            rep = model2.predict_proba(self.sharedInputTrain.array)[:, 1]
        else:
            output = ar
            rep = ar2

        return output, rep, i

    def __call__(self, i):
        return self.svmModel(i)

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        from sklearn import svm

        print(inputTrain.shape, outputTrain.shape, inputTest.shape)

        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        outputs = []
        reps = []
        nTest = inputTest.shape[0]
        print("SVM for %s classes" % nClass)
        model = svm.SVC(C=const.SVM_C, gamma='auto', kernel='rbf', probability=True)
        self.model = model

        from multiprocessing import Pool

        from models.sharedMem import SharedNDArray

        self.sharedInputTrain = SharedNDArray.copy(inputTrain)
        self.sharedInputTest = SharedNDArray.copy(inputTest)
        self.sharedOutputTrain = SharedNDArray.copy(outputTrain)

        if const.SVM_PARALLEL:
            print("In parallel mode")
            start = time.time()

            iters = np.arange(0, nClass)
            pool = Pool(const.N_PARALLEL)
            adrOutputs = pool.map_async(self, iters)
            pool.close()
            pool.join()

            outputs = []
            reps = []
            while not adrOutputs.ready():
                print("num left: {}".format(adrOutputs._number_left))
                adrOutputs.wait(1)

            print(adrOutputs)
            dout = dict()
            for output in adrOutputs.get():
                dout[output[2]] = output[0], output[1]

            for ii in range(len(dout)):
                out1, out2 = dout[ii]
                outputs.append(out1)
                reps.append(out2)

            end = time.time()
            print("Elapsed: ", end - start)

        else:
            print("In sequential mode")
            start = time.time()

            for i in range(nClass):
                if i % 10 == 0:
                    print("\r%s" % i, end="")
                output = outputTrain[:, i]
                ar = checkOneClass(output, nTest)
                ar2 = checkOneClass(output, inputTrain.shape[0])

                # print clf
                if type(ar) == int:
                    model.fit(inputTrain, output)
                    output = model.predict_proba(inputTest)[:, 1]
                    rep = model.predict_proba(inputTrain)[:, 1]
                else:
                    output = ar
                    rep = ar2
                outputs.append(output)
                reps.append(rep)

            end = time.time()
            print("Elapsed: ", end - start)

        outputs = np.vstack(outputs).transpose()
        reps = np.vstack(reps).transpose()
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps

        print(outputs.shape)
        print("\nDone")
        return outputs

    def getInfo(self):
        return self.name


class KGSIM(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "KGSIM"

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        print(len(inputTrain), outputTrain.shape, len(inputTest))

        nTrain, nTest = len(inputTrain), len(inputTest)
        outSize = outputTrain.shape[1]
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.get3WJaccardOnSets(inputTest[i], inputTrain[j])

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        args = args[:, :const.KGSIM]
        # print args

        outputs = []
        for i in range(nTest):
            out = np.zeros(outSize, dtype=float)
            matches = args[i]
            simScores = simMatrix[i, matches]
            ic = -1
            sum = 1e-10
            for j in matches:
                ic += 1
                out += simScores[ic] * outputTrain[j]
                sum += simScores[ic]
            out /= sum
            outputs.append(out)
        outputs = np.vstack(outputs)

        return outputs

    def getInfo(self):
        return "KGSIM %s" % const.KNN


class RFModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "RF"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.ensemble import RandomForestClassifier
        print(intpuTrain.shape, outputTrain.shape, inputTest.shape)

        def fixListArray(listar):
            out = []
            for ar in listar:
                if min(ar.shape) == 1:
                    zr = np.zeros(len(ar))
                    ar = np.concatenate([ar, zr[:, np.newaxis]], axis=1)
                out.append(ar)
            return out

        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        predicts = []
        nTest = inputTest.shape[0]
        print("RF for %s classes" % nClass)
        cc = 0
        reps = []
        model = RandomForestClassifier(n_estimators=const.RF)
        self.model = model
        model.fit(intpuTrain, outputTrain)
        print(intpuTrain.shape, outputTrain.shape)
        o = model.predict_proba(inputTest)
        r = model.predict_proba(intpuTrain)

        o = fixListArray(o)
        r = fixListArray(r)

        o = np.asarray(o)
        r = np.asarray(r)

        print(o.shape, r.shape)
        outputs = o[:, :, 1].transpose()
        reps = r[:, :, 1].transpose()
        print("Debug")
        print(outputs.shape, reps.shape)
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps

        # for i in range(nClass):
        #     if i % 10 == 0:
        #         print("\r%s" % i, end="")
        #     output = outputTrain[:, i]
        #     ar = checkOneClass(output, nTest)
        #     ar2 = checkOneClass(output, intpuTrain.shape[0])
        #
        #     # print model
        #     if type(ar) == int:
        #
        #         model.fit(intpuTrain, output)
        #         pred = model.predict_proba(inputTest)[:, 1]
        #         rep = model.predict_proba(intpuTrain)[:, 1]
        #
        #     else:
        #         pred = ar
        #         rep = ar2
        #         cc += 1
        #     predicts.append(pred)
        #     reps.append(rep)
        #
        # outputs = np.vstack(predicts).transpose()
        # reps = np.vstack(reps).transpose()
        # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        # self.repred = reps
        # print("\nDone. Null cls: %s" % cc)
        # print(outputs.shape)

        return outputs

    def getInfo(self):
        return self.name


class GBModel(Model):

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "GB"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.ensemble import GradientBoostingClassifier
        print(intpuTrain.shape, outputTrain.shape, inputTest.shape)

        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        predicts = []
        nTest = inputTest.shape[0]
        print("Gradient Boosting Classifier for %s classes" % nClass)
        cc = 0
        reps = []
        clf = GradientBoostingClassifier(n_estimators=const.RF)
        self.model = clf
        for i in range(nClass):
            if i % 10 == 0:
                print("\r%s" % i, end="")
            output = outputTrain[:, i]
            ar = checkOneClass(output, nTest)
            ar2 = checkOneClass(output, intpuTrain.shape[0])

            # print clf
            if type(ar) == int:

                clf.fit(intpuTrain, output)
                pred = clf.predict_proba(inputTest)[:, 1]
                rep = clf.predict_proba(intpuTrain)[:, 1]

            else:
                pred = ar
                rep = ar2
                cc += 1
            predicts.append(pred)
            reps.append(rep)

        outputs = np.vstack(predicts).transpose()
        reps = np.vstack(reps).transpose()
        print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps
        print("\nDone. Null cls: %s" % cc)
        print(outputs.shape)

        return outputs

    def getInfo(self):
        return self.model


class KNN(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "KNN"

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        from sklearn.neighbors import KNeighborsClassifier
        print(inputTrain.shape, outputTrain.shape, inputTest.shape)

        nTrain, nTest = inputTrain.shape[0], inputTest.shape[0]

        # self.model = KNeighborsClassifier(const.KNN,metric=utils.getTanimotoScore)
        # self.model.fit(inputTrain, outputTrain)
        # outputs = np.asarray(self.model.predict_proba(inputTest))[:, :, 1].transpose()
        # print("Pred shape:", outputs.shape)

        outSize = outputTrain.shape[1]
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getSimByType(inputTest[i], inputTrain[j], const.KNN_SIM)

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        args = args[:, :const.KNN]
        # print args

        outputs = []
        for i in range(nTest):
            out = np.zeros(outSize, dtype=float)
            matches = args[i]
            simScores = simMatrix[i, matches]
            ic = -1
            sum = 1e-10
            for j in matches:
                ic += 1
                out += simScores[ic] * outputTrain[j]
                sum += simScores[ic]
            out /= sum
            outputs.append(out)
        outputs = np.vstack(outputs)

        return outputs

    def getInfo(self):
        return "KNN %s" % const.KNN
