import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

import const
import models.layer as layer
import utils


class Model:
    def __init__(self):
        self.isFitAndPredict = False
        self.name = "General"
        self.repred = ""
        self.model = ""

    def fit(self, intputTrain, outputTrain):
        pass

    def predict(self, input):
        pass

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        pass

    def getInfo(self):
        pass


class DrugNCFwoshare(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims_wide, num_deep_layers, num_users, num_items, dropout):
        # a question: use the pre-defined DNN or adjustable mlp
        super(DrugNCFwoshare, self).__init__()
        self.name = 'DrugNCFwoshare'
        self.num_items = num_items
        self.num_users = num_users
        self.lr = layer.FeaturesLinear(field_dims, output_dim=num_items)
        self.embedding_k = embed_dim
        self.W = torch.nn.Embedding(self.num_users + 1, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items + 1, self.embedding_k)
        self.H1 = torch.nn.Embedding(self.num_items + 1, self.embedding_k)

        if len(mlp_dims_wide) == 0:
            self.mlp_wide = nn.Sequential(
                layer.FeaturesLinear(field_dims, output_dim=self.embedding_k),
                nn.Sigmoid()
            )
        else:
            self.mlp_wide = nn.Sequential(
                layer.MultiLayerPerceptron(sum(field_dims), mlp_dims_wide, dropout=dropout,
                                           output_dim=self.embedding_k),
                nn.Sigmoid()
            )

        self.xent_func = torch.nn.BCELoss()
        self.field_dims = field_dims

        self.relu = torch.nn.ReLU()  #
        self.sigmoid = torch.nn.Sigmoid()

        self.linear_g = nn.Linear(num_items, num_items)

        self.linear_g2 = nn.Linear(1, 1)

    def getInfo(self):
        Info = "DrugNCFwoshare: k = " + str(self.embedding_k) + " ,learning_rate = " + str(
            const.LEARNING_RATE) + " ,lambda = " + str(
            const.LAMB) + " ,KNN = " + str(const.KNN) + " ,tol = " + str(const.TOL) + " ,cuda = " + const.CUDA_DEVICE
        return Info

    def Embedding(self, x, adr_idx):
        # input: a batch of index list : [index of drug, index of ADR]

        U_emb = self.W(x)  # embedding: from batch_size to batch_size * k
        V_emb = self.H(adr_idx)  # embedding: from batch_size to batch_size * k

        V1_emb = self.H1(adr_idx)

        return U_emb, V_emb, V1_emb

    def forward(self, x, drug_features_x, adr_idx):
        # embed, concat and DNN
        # batch_size = len(x)

        nADR = len(adr_idx)

        U_emb, V_emb, V1_emb = self.Embedding(x, adr_idx)

        wide_part = self.mlp_wide(drug_features_x)
        wide_part = torch.mm(wide_part, V_emb.T)

        wide_part = self.linear_g(wide_part)

        wide_lr = self.lr(drug_features_x)
        wide_part = wide_lr + wide_part

        mf_part = torch.mm(U_emb, V1_emb.T)

        mf_part = self.linear_g2(mf_part.reshape(-1, 1)).reshape(-1, nADR)

        out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part
        out = self.sigmoid(out.reshape(-1))

        return out

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=500, lr=const.LEARNING_RATE,
                      lamb=const.LAMB, tol=1e-4, verbose=1):  # lambda 1e-4 1e-3 , batch_size=4096
        # print(lamb)
        if const.CURRENT_DATA == 'Liu':
            tol = 1e-2
        else:
            tol = 1e-3

        nDrug = len(outputTrain)
        nADR = len(outputTrain[0])

        nDrug_test = len(outputTest)
        nADR_test = len(outputTest[0])

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        last_auc = 0

        batch_size = nDrug

        num_sample = nDrug
        total_batch = num_sample // batch_size

        inputTrain_cuda = torch.from_numpy(inputTrain).cuda(const.CUDA_DEVICE).long()

        early_stop = 0
        early_stop_overfit = 0
        loss_collected = []
        auroc_collected = []
        auprc_collected = []

        # we collect the auprc and the auroc everytime 

        drug_idx = np.arange(len(outputTrain))
        adr_idx = np.arange(len(outputTrain[0]))

        drug_idx = torch.from_numpy(drug_idx).cuda(const.CUDA_DEVICE)
        adr_idx = torch.from_numpy(adr_idx).cuda(const.CUDA_DEVICE)

        nTrain, nTest = inputTrain.shape[0], inputTest.shape[0]  # 1222, 136

        simMatrix = np.ndarray((nTest, nTrain), dtype=float)  # 136 * 1222
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimoto(inputTest[i], inputTrain[j])
        # 计算测试集和训练集样本的两两相似度
        # print("simMatrix:", simMatrix)

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        # 为测试集中每个样本与训练集中的每个样本相似度进行排序，返回索引

        args = args[:, :const.KNN]  # KNN=60 每个测试数据只保留前六十个相似度最高的样本数据的索引

        inputTest = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE)
        outputTrain_cuda = torch.tensor(outputTrain).to(torch.float32).cuda(const.CUDA_DEVICE)
        outputTest_cuda = torch.tensor(outputTest).cuda(const.CUDA_DEVICE)

        for epoch in range(num_epoch):

            all_idx = np.arange(nDrug)
            np.random.shuffle(all_idx)

            epoch_loss = 0
            epoch_auc = 0
            for idx in range(total_batch):  # idx:第几个batch

                # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index

                sub_x = drug_idx[selected_idx]

                drug_features = inputTrain_cuda[sub_x]
                sub_y = outputTrain_cuda[sub_x]

                # batch中drug的ADR, size: batch_size, value: bool

                optimizer.zero_grad()
                pred = self.forward(sub_x, drug_features, adr_idx=adr_idx)

                target = torch.unsqueeze(sub_y, 1)
                target = target.cuda(const.CUDA_DEVICE).reshape(-1)

                xent_loss = self.xent_func(pred, target)

                loss = xent_loss  ##

                loss.backward()
                optimizer.step()
                xent_loss_cpu = xent_loss.cpu()
                epoch_loss += xent_loss_cpu.detach().numpy()

            loss_collected.append(epoch_loss)

            if const.SAVE_TRAJECTORY:
                chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity
                ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

                ADRFeatures_1 = self.H1(adr_idx)

                u_emb_weighted = []
                for i in range(nTest):  # 求136个测试数据的u_emb，并根据相似度进行加权，相似度越高权重越大
                    newF = np.zeros(self.embedding_k, dtype=float)
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
                u_emb_weighted = torch.from_numpy(u_emb_weighted).to(torch.float32).cuda(const.CUDA_DEVICE)

                wide_part = torch.mm(self.mlp_wide(inputTest), ADRFeatures.T).cuda(const.CUDA_DEVICE)
                wide_part = self.linear_g(wide_part)

                wide_part += self.lr(inputTest)
                mf_part = torch.mm(u_emb_weighted, ADRFeatures_1.T).cuda(const.CUDA_DEVICE)

                mf_part = self.linear_g2(mf_part.reshape(-1, 1)).reshape(-1, nADR)

                out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part

                out = self.sigmoid(out.reshape(-1))

                testOut_vec = outputTest.reshape(-1)
                predictedValues = out.cpu().detach().numpy()
                pred_vec = predictedValues.reshape(-1)
                aucs = roc_auc_score(testOut_vec, pred_vec)
                auprs = average_precision_score(testOut_vec.reshape(-1), predictedValues.reshape(-1))
                # auroc_collected.append(aucs)
                # auprc_collected.append(auprs)

                print("epoch:", epoch, "test auc:", aucs, "test aupr: ", auprs)

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if epoch >= 10 and relative_loss_div < tol:
                if early_stop > 5:
                    print("[DrugNCF][Train] Early stop in epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            print("[DrugNCF][In Train] epoch:{}, xent:{}".format(epoch, epoch_loss))

            # if epoch != 0:

            #     chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity
            #     ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

            #     ADRFeatures_1 = self.H1(adr_idx)

            #     u_emb_weighted = []
            #     for i in range(nTest):  # 求136个测试数据的u_emb，并根据相似度进行加权，相似度越高权重越大
            #         newF = np.zeros(self.embedding_k, dtype=float)
            #         matches = args[i]  # 单个测试数据前60相似度的索引
            #         simScores = simMatrix[i, matches]  # 单个数据的前60相似度值
            #         ic = -1
            #         sum = 1e-10
            #         for j in matches:  # 将simScores(1*60)和chemFeatures中对应的60个药物样本数据（60*k）逐项相乘, 最终得出136*k的测试矩阵
            #             ic += 1
            #             newF += simScores[ic] * chemFeatures[j].cpu().detach().numpy()
            #             sum += simScores[ic]  # 求simScores的总和
            #         newF /= sum
            #         u_emb_weighted.append(newF)

            #     u_emb_weighted = np.array(u_emb_weighted)
            #     u_emb_weighted = torch.from_numpy(u_emb_weighted).to(torch.float32).cuda(const.CUDA_DEVICE)

            #     wide_part = torch.mm(self.mlp_wide(inputTest), ADRFeatures.T).cuda(const.CUDA_DEVICE)
            #     wide_part = self.linear_g(wide_part)

            #     wide_part += self.lr(inputTest)
            #     mf_part = torch.mm(u_emb_weighted, ADRFeatures_1.T).cuda(const.CUDA_DEVICE)

            #     mf_part = self.linear_g2(mf_part.reshape(-1, 1)).reshape(-1, nADR)

            #     out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part
            #     out = self.sigmoid(out.reshape(-1))

            #     testOut_vec = outputTest.reshape(-1)
            #     pred_vec = out.cpu().detach().numpy().reshape(-1)
            #     auc = roc_auc_score(testOut_vec, pred_vec)
            #     print("epoch: ", epoch, "test auc:", auc)
            #     print("epoch: ", epoch, "relative_loss_tol:", relative_loss_div)

            #     if auc < last_auc:
            #         print("overfit")
            #         early_stop_overfit += 1
            #     if early_stop_overfit > 10:
            #         print("[DrugNCF][Train] epoch:{}, xent:{}".format(epoch, epoch_loss))
            #         break
            #     last_auc = auc

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

        if not const.SAVE_TRAJECTORY:
            chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity
            ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

            ADRFeatures_1 = self.H1(adr_idx)

            u_emb_weighted = []
            for i in range(nTest):  # 求136个测试数据的u_emb，并根据相似度进行加权，相似度越高权重越大
                newF = np.zeros(self.embedding_k, dtype=float)
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
            u_emb_weighted = torch.from_numpy(u_emb_weighted).to(torch.float32).cuda(const.CUDA_DEVICE)

            wide_part = torch.mm(self.mlp_wide(inputTest), ADRFeatures.T).cuda(const.CUDA_DEVICE)
            wide_part = self.linear_g(wide_part)

            wide_part += self.lr(inputTest)
            mf_part = torch.mm(u_emb_weighted, ADRFeatures_1.T).cuda(const.CUDA_DEVICE)

            mf_part = self.linear_g2(mf_part.reshape(-1, 1)).reshape(-1, nADR)

            out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part
            out = self.sigmoid(out.reshape(-1))

        return out.cpu().detach().numpy(), loss_collected, auroc_collected, auprc_collected


class DrugNCF(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims_wide, mlp_dims_cf, num_users, num_items, dropout):
        # a question: use the pre-defined DNN or adjustable mlp
        super(DrugNCF, self).__init__()
        self.name = 'DrugNCF'
        self.num_items = num_items
        self.num_users = num_users
        self.num_layers_cf = len(mlp_dims_cf)
        self.embedding_k = embed_dim

        self.cf_embedding = nn.Linear(sum(field_dims), 2 * self.embedding_k)
        self.W = torch.nn.Embedding(self.num_users + 1, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items + 1, self.embedding_k)

        self.lr = layer.FeaturesLinear(field_dims, output_dim=num_items)
        if len(mlp_dims_cf):
            self.mlp_cf = nn.Sequential(
                layer.MultiLayerPerceptron(self.embedding_k * 2, mlp_dims_cf, dropout=dropout,
                                           output_dim=num_items),
                nn.Sigmoid()
            )

        if len(mlp_dims_wide) == 0:  # the number of the layers -2 in wide&deep
            self.mlp_wide = nn.Sequential(
                layer.FeaturesLinear(field_dims, output_dim=self.embedding_k),
                nn.Sigmoid()
            )
        else:
            self.mlp_wide = nn.Sequential(
                layer.MultiLayerPerceptron(sum(field_dims), mlp_dims_wide, dropout=dropout,
                                           output_dim=self.embedding_k),
                nn.Sigmoid()
            )

        self.xent_func = torch.nn.BCELoss()
        self.field_dims = field_dims

        self.relu = torch.nn.ReLU()  #
        self.sigmoid = torch.nn.Sigmoid()

        self.linear_out = nn.Linear(self.embedding_k, num_items)

        self.linear_g = nn.Linear(num_items, num_items)

        self.linear_mf = nn.Linear(1, 1)

    def getInfo(self):
        Info = "DrugNCF: k = " + str(self.embedding_k) + ", num_wide_layer=" + str(const.N_WIDE_LAYERS) \
               + ", num_cf_layer=" + str(const.N_CF_LAYERS) \
               + ", learning_rate = " + str(
            const.LEARNING_RATE) + ", lambda = " + str(
            const.LAMB) + ", KNN = " + str(const.KNN) + ", tol = " + str(const.TOL)
        return Info

    def MF(self, x_idx, adr_idx):
        U_emb = self.W(x_idx)  # [batch_size, k]
        V_emb = self.H(adr_idx)  # [number of ADR, k]
        out = torch.mm(U_emb, V_emb.T)
        out = self.linear_mf(out.reshape(-1, 1)).reshape(-1, len(adr_idx))
        return out, U_emb, V_emb

    def NCF(self, x):
        # input x: [batch_size, dim of features]
        # if num_layers = 0 : MF, others: NCF
            x = x.float()
            x_emb = self.cf_embedding(x)  # from [batch_size, dim of features] to [batch_size, 2k]
            u_emb = x_emb[:, :self.embedding_k]   # [batch_size, k]
            i_emb = x_emb[:, self.embedding_k:]   # [batch_size, k]
            out = self.mlp_cf(x_emb)  # [batch_size, k]
            return out, u_emb, i_emb

    def forward(self, drug_features_x, x_idx, adr_idx):
        # embed, concat and DNN
        # batch_size = len(x)

        nADR = len(adr_idx)

        wide_part = self.mlp_wide(drug_features_x)  # [batch_size, k]
        wide_lr = self.lr(drug_features_x)  # [batch_size, number of ADR]

        if self.num_layers_cf == 0:  # MF
            mf_part, U_emb, V_emb = self.MF(x_idx, adr_idx)
            wide_part = torch.mm(wide_part, V_emb.T)
            # wide_part = self.linear_g(wide_part)
            wide_part = wide_part + wide_lr
            out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part
            out = self.sigmoid(out.reshape(-1))
            return out

        else:
            ncf_part, u_emb, i_emb = self.NCF(drug_features_x)
            wide_part = torch.mul(wide_part, i_emb)
            wide_part = torch.sigmoid(wide_part)
            wide_part = self.linear_out(wide_part)
            wide_part = wide_part + wide_lr
            out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * ncf_part
            out = self.sigmoid(out.reshape(-1))

        return out

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=500, lr=const.LEARNING_RATE,
                      lamb=const.LAMB, tol=1e-4, verbose=1):  # lambda 1e-4 1e-3 , batch_size=4096
        print(lamb)
        if const.CURRENT_DATA == 'Liu':
            tol = 1e-2
        else:
            tol = 1e-3

        nDrug = len(outputTrain)
        nADR = len(outputTrain[0])

        nDrug_test = len(outputTest)
        nADR_test = len(outputTest[0])

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9
        last_auc = 0

        batch_size = nDrug

        num_sample = nDrug
        total_batch = num_sample // batch_size

        inputTrain_cuda = torch.from_numpy(inputTrain).cuda(const.CUDA_DEVICE).long()

        early_stop = 0
        early_stop_overfit = 0
        loss_collected = []
        auroc_collected = []
        auprc_collected = []

        # we collect the auprc and the auroc everytime 

        drug_idx = np.arange(len(outputTrain))
        adr_idx = np.arange(len(outputTrain[0]))

        drug_idx = torch.from_numpy(drug_idx).cuda(const.CUDA_DEVICE)
        adr_idx = torch.from_numpy(adr_idx).cuda(const.CUDA_DEVICE)

        nTrain, nTest = inputTrain.shape[0], inputTest.shape[0]  # 1222, 136

        simMatrix = np.ndarray((nTest, nTrain), dtype=float)  # 136 * 1222
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimoto(inputTest[i], inputTrain[j])
        # 计算测试集和训练集样本的两两相似度
        # print("simMatrix:", simMatrix)

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        # 为测试集中每个样本与训练集中的每个样本相似度进行排序，返回索引

        args = args[:, :const.KNN]  # KNN=60 每个测试数据只保留前六十个相似度最高的样本数据的索引

        inputTest = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE)
        outputTrain_cuda = torch.tensor(outputTrain).to(torch.float32).cuda(const.CUDA_DEVICE)
        outputTest_cuda = torch.tensor(outputTest).cuda(const.CUDA_DEVICE)

        for epoch in range(num_epoch):

            all_idx = np.arange(nDrug)
            np.random.shuffle(all_idx)

            epoch_loss = 0
            epoch_auc = 0
            for idx in range(total_batch):  # idx:第几个batch

                # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index

                sub_drug_index = drug_idx[selected_idx].long()  # sub_drug_index

                sub_drug_features = inputTrain_cuda[sub_drug_index]
                sub_y = outputTrain_cuda[sub_drug_index]

                # batch中drug的ADR, size: batch_size, value: bool

                optimizer.zero_grad()
                pred = self.forward(sub_drug_features, sub_drug_index, adr_idx)

                target = torch.unsqueeze(sub_y, 1)
                target = target.cuda(const.CUDA_DEVICE).reshape(-1)

                xent_loss = self.xent_func(pred, target)

                loss = xent_loss  ##

                loss.backward()
                optimizer.step()
                xent_loss_cpu = xent_loss.cpu()
                epoch_loss += xent_loss_cpu.detach().numpy()

            loss_collected.append(epoch_loss)

            if const.SAVE_TRAJECTORY:
                if self.num_layers_cf == 0:

                    chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity
                    ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

                    u_emb_weighted = []
                    for i in range(nTest):  # 求136个测试数据的u_emb，并根据相似度进行加权，相似度越高权重越大
                        newF = np.zeros(self.embedding_k, dtype=float)
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
                    u_emb_weighted = torch.from_numpy(u_emb_weighted).to(torch.float32).cuda(const.CUDA_DEVICE)

                    wide_part = torch.mm(self.mlp_wide(inputTest), ADRFeatures.T).cuda(const.CUDA_DEVICE)
                    # wide_part = self.linear_g(wide_part)
                    wide_part += self.lr(inputTest)

                    mf_part = torch.mm(u_emb_weighted, ADRFeatures.T)
                    mf_part = self.linear_mf(mf_part.reshape(-1, 1)).reshape(-1, nADR)

                    out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part

                    out = self.sigmoid(out.reshape(-1))

                else:  # ncf
                    ncf_part, u_emb, i_emb = self.NCF(inputTest)
                    wide_part = self.mlp_wide(inputTest)
                    wide_part = torch.mul(wide_part, i_emb)
                    wide_part = torch.sigmoid(wide_part)
                    wide_part = self.linear_out(wide_part)
                    wide_part += self.lr(inputTest)
                    # wide_part = self.linear_g(wide_part)
                    out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * ncf_part
                    out = self.sigmoid(out.reshape(-1))

                testOut_vec = outputTest.reshape(-1)
                predictedValues = out.cpu().detach().numpy()
                pred_vec = predictedValues.reshape(-1)
                aucs = roc_auc_score(testOut_vec, pred_vec)
                auprs = average_precision_score(testOut_vec.reshape(-1), predictedValues.reshape(-1))
                # auroc_collected.append(aucs)
                # auprc_collected.append(auprs)

                print("epoch:", epoch, "test auc:", aucs, "test aupr: ", auprs)

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if epoch >= 10 and relative_loss_div < tol:
                if early_stop > 5:
                    print("[DrugNCF][Train] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            print("[DrugNCF][In Train] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

        if not const.SAVE_TRAJECTORY:
            if self.num_layers_cf == 0:

                chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity
                ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

                u_emb_weighted = []
                for i in range(nTest):  # 求136个测试数据的u_emb，并根据相似度进行加权，相似度越高权重越大
                    newF = np.zeros(self.embedding_k, dtype=float)
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
                u_emb_weighted = torch.from_numpy(u_emb_weighted).to(torch.float32).cuda(const.CUDA_DEVICE)

                wide_part = torch.mm(self.mlp_wide(inputTest), ADRFeatures.T).cuda(const.CUDA_DEVICE)
                # wide_part = self.linear_g(wide_part)
                wide_part += self.lr(inputTest)

                mf_part = torch.mm(u_emb_weighted, ADRFeatures.T)
                mf_part = self.linear_mf(mf_part.reshape(-1, 1)).reshape(-1, nADR)

                out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part

                out = self.sigmoid(out.reshape(-1))

            else:  # ncf
                ncf_part, u_emb, i_emb = self.NCF(inputTest)
                wide_part = self.mlp_wide(inputTest)
                wide_part = torch.mul(wide_part, i_emb)
                wide_part = torch.sigmoid(wide_part)
                wide_part = self.linear_out(wide_part)
                wide_part += self.lr(inputTest)
                out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * ncf_part
                out = self.sigmoid(out.reshape(-1))

        return out.cpu().detach().numpy(), loss_collected, auroc_collected, auprc_collected
