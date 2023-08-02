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
        self.embedding_k = embed_dim
        self.W = torch.nn.Embedding(self.num_users + 1, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items + 1, self.embedding_k)
        self.H1 = torch.nn.Embedding(self.num_items + 1, self.embedding_k)
        if num_deep_layers == 0:
            self.mlp_deep = nn.Linear(self.embedding_k * 2, 1, bias = False)
        elif num_deep_layers == 1:
            self.mlp_deep = nn.Sequential(
                nn.Linear(self.embedding_k *2, self.embedding_k), 
                nn.Sigmoid(), 
                nn.Linear(self.embedding_k, 1, bias = False)
            )
        else:
            layers = [nn.Linear(self.embedding_k *2, self.embedding_k), nn.Sigmoid() ]
            for ik in range(num_deep_layers - 1):
                layers.extend([nn.Linear(self.embedding_k, self.embedding_k), nn.Sigmoid()])
            layers.append(nn.Linear(self.embedding_k, 1, bias = False))
            self.mlp_deep = nn.Sequential(*layers)

        if len(mlp_dims_wide) == 0:
            self.mlp_wide = layer.FeaturesLinear(field_dims, output_dim=self.embedding_k)
        else:
            self.mlp_wide = layer.MultiLayerPerceptron(sum(field_dims), mlp_dims_wide, dropout=dropout, output_dim=self.embedding_k)

        self.xent_func = torch.nn.BCELoss()
        self.field_dims = field_dims

        self.relu = torch.nn.ReLU()  #
        self.sigmoid = torch.nn.Sigmoid()

        self.linear_g = nn.Linear( 1, 1)

    def getInfo(self):
        Info = "DrugNCFwoshare: k = " + str(self.embedding_k) + " ,learning_rate = " + str(const.LEARNING_RATE) + " ,lambda = " + str(
            const.LAMB) + " ,KNN = " + str(const.KNN) + " ,tol = " + str(const.TOL) + " ,cuda = " + const.CUDA_DEVICE
        return Info

    def getConcat(self, drug_latent, ADR_latent):
        # drug_latent: number of drugs * k
        # ADR_latent: k * number of ADRs
        # return: (number of drugs * number of ADRs) * 2k

        drug_latent = drug_latent.cuda(const.CUDA_DEVICE)
        ADR_latent = ADR_latent.cuda(const.CUDA_DEVICE)  # 调用Cuda加速效果明显

        num_drug = len(drug_latent)  # How many drugs data input
        num_ADR = len(ADR_latent)  # How many ADRs

        concat_data = torch.Tensor(num_drug * num_ADR, self.embedding_k * 2).cuda(
            const.CUDA_DEVICE)  # (2707*batch_size)  * 2k

        i = 0
        for u in drug_latent:
            uu = u.reshape(1, self.embedding_k)
            for v in ADR_latent:
                vv = v.reshape(1, self.embedding_k)
                concat_data[i] = torch.cat([uu, vv], axis=1)
                # dot[i] = torch.mm(uu.double(), vv.T.double())
                i += 1

        return concat_data

    def Embedding(self, x):
        # input: a batch of index list : [index of drug, index of ADR]

        user_idx = torch.LongTensor(x[:, 0]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of drugs from x
        item_idx = torch.LongTensor(x[:, 1]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of ADRs from x

        U_emb = self.W(user_idx)  # embedding: from batch_size to batch_size * k
        V_emb = self.H(item_idx)  # embedding: from batch_size to batch_size * k

        # another embedding
        V1_emb = self.H1(item_idx)

        return U_emb, V_emb, V1_emb
        

    def forward(self, x, drug_features_x):
        # embed, concat and DNN
        # batch_size = len(x)

        U_emb, V_emb, V1_emb = self.Embedding(x)

        z_emb = torch.cat([U_emb, V_emb], axis=1)
        
        dnn = self.mlp_deep(z_emb)
        # DrugFeature = DrugFeature.cuda(const.CUDA_DEVICE)

        wide_part = self.mlp_wide(drug_features_x)
        wide_part = torch.mul(wide_part, V1_emb)
        # print( wide_part.shape )
        wide_part_t = torch.sum(wide_part, dim = 1).reshape(-1, 1)

        wide_part_t = self.linear_g(wide_part_t)

        # wide_part_t = torch.Tensor( [ wide_part[i].dot(V_emb[i]) for i in range(0, batch_size) ]  ).reshape(-1, 1).cuda(const.CUDA_DEVICE)

        # wide_part = torch.mm(self.mlp_wide(DrugFeature), V_emb.T)
        # wide_part_t = torch.Tensor( [ wide_part[i, i] for i in range(0, batch_size) ] ).reshape(-1, 1).cuda(const.CUDA_DEVICE)

        out = wide_part_t + dnn
        out = self.sigmoid(out.reshape(-1))

        return out
    
    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=1000, lr=const.LEARNING_RATE, lamb=const.LAMB, tol=1e-4, batch_size=65536, verbose=1):  # lambda 1e-4 1e-3 , batch_size=4096


        nDrug = len(outputTrain)
        nADR = len(outputTrain[0])

        nDrug_test = len(outputTest)
        nADR_test = len(outputTest[0])

        outputTrain_index = utils.indices_array_generic(nDrug, nADR)
        outputTest_index = utils.indices_array_generic(nDrug_test, nADR_test)

        # An array of output (binary) in order
        outputTrain_value = np.empty(nDrug * nADR)
        for i in range(nDrug):
            for j in range(nADR):
                outputTrain_value[i * nADR + j] = outputTrain[i][j]

        inputTest_value = inputTest[outputTest_index[:, 0]]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(outputTrain_index)
        total_batch = num_sample // batch_size

        inputTrain_cuda = torch.from_numpy(inputTrain).cuda(const.CUDA_DEVICE).long()
        # print(inputTrain_cuda.shape)

        early_stop = 0
        loss_collected = []
        auroc_collected = []
        auprc_collected = []

        # we collect the auprc and the auroc everytime 

        drug_idx = np.arange(len(outputTrain))
        adr_idx = np.arange(len(outputTrain[0]))

        drug_idx = torch.from_numpy(drug_idx).cuda(const.CUDA_DEVICE)
        adr_idx = torch.from_numpy(adr_idx).cuda(const.CUDA_DEVICE)

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

        inputTest = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE)

        for epoch in range(num_epoch):

            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):  # idx:第几个batch

                # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                sub_x = outputTrain_index[selected_idx]  # batch中drug与对应ADR的索引数据对，size：batch_size * 2
                drug_index = sub_x[:, 0]  # 取出数据对中的drug索引

                drug_features = inputTrain_cuda[drug_index]
                sub_y = outputTrain_value[selected_idx]
                
                  # batch中drug的ADR, size: batch_size, value: bool

                optimizer.zero_grad()
                pred = self.forward(sub_x, drug_features)

                # pred=pred.cuda(const.CUDA_DEVICE)
                # pred = pred.to(const.CPU)

                # target = torch.tensor(sub_y).reshape(-1, len(pred))      # reshape the test ADR
                target = torch.unsqueeze(torch.Tensor(sub_y), 1)
                target = target.cuda(const.CUDA_DEVICE).reshape(-1)
                # xent_loss = self.xent_func(pred.T.float(), target.float()) # require float

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
                u_emb_weighted = torch.from_numpy(u_emb_weighted)

                concat_data = self.getConcat(u_emb_weighted, ADRFeatures)

                dnn = self.mlp_deep(concat_data)

                wide_part_t = torch.mm(self.mlp_wide(inputTest), ADRFeatures_1.T).reshape(-1, 1).cuda(const.CUDA_DEVICE)
                wide_part_t = self.linear_g(wide_part_t)

                out = wide_part_t + dnn
                out = self.sigmoid(out.reshape(-1))

                testOut_vec = outputTest.reshape(-1)
                predictedValues = out.cpu().detach().numpy()
                pred_vec = predictedValues.reshape(-1)
                aucs = roc_auc_score(testOut_vec, pred_vec)
                auprs = average_precision_score(testOut_vec.reshape(-1), predictedValues.reshape(-1))
                auroc_collected.append(aucs)
                auprc_collected.append(auprs)

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[DrugNCF][Train] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            print("[DrugNCF][Train] epoch:{}, xent:{}".format(epoch, epoch_loss))


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
            u_emb_weighted = torch.from_numpy(u_emb_weighted)

            concat_data = self.getConcat(u_emb_weighted, ADRFeatures)

            dnn = self.mlp_deep(concat_data)

            wide_part_t = torch.mm(self.mlp_wide(inputTest), ADRFeatures_1.T).reshape(-1, 1).cuda(const.CUDA_DEVICE)
            wide_part_t = self.linear_g(wide_part_t)

            out = wide_part_t + dnn
            out = self.sigmoid(out.reshape(-1))

        return out.cpu().detach().numpy(), loss_collected, auroc_collected, auprc_collected


class DrugNCF(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, mlp_dims_wide, num_deep_layers, num_users, num_items, dropout):
        # a question: use the pre-defined DNN or adjustable mlp
        super(DrugNCF, self).__init__()
        self.name = 'DrugNCF'
        self.num_items = num_items
        self.num_users = num_users
        self.lr = layer.FeaturesLinear(field_dims)
        self.embedding_k = embed_dim
        self.W = torch.nn.Embedding(self.num_users + 1, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items + 1, self.embedding_k)
        if num_deep_layers == 0:
            self.mlp_deep = nn.Linear(self.embedding_k * 2, 1, bias = False)
        elif num_deep_layers == 1:
            self.mlp_deep = nn.Sequential(
                nn.Linear(self.embedding_k *2, self.embedding_k), 
                nn.Sigmoid(), 
                nn.Linear(self.embedding_k, 1, bias = False)
            )
        else:
            layers = [nn.Linear(self.embedding_k *2, self.embedding_k), nn.Sigmoid() ]
            for ik in range(num_deep_layers - 1):
                layers.extend([nn.Linear(self.embedding_k, self.embedding_k), nn.Sigmoid()])
            layers.append(nn.Linear(self.embedding_k, 1, bias = False))
            self.mlp_deep = nn.Sequential(*layers)

        if len(mlp_dims_wide) == 0:
            self.mlp_wide = layer.FeaturesLinear(field_dims, output_dim=self.embedding_k)
        else:
            self.mlp_wide = layer.MultiLayerPerceptron(sum(field_dims), mlp_dims_wide, dropout=dropout, output_dim=self.embedding_k)

        self.xent_func = torch.nn.BCELoss()
        self.field_dims = field_dims

        self.relu = torch.nn.ReLU()  #
        self.sigmoid = torch.nn.Sigmoid()

        self.linear_g = nn.Linear( 1, 1)

    def getInfo(self):
        Info = "DrugNCF: k = " + str(self.embedding_k) + " ,learning_rate = " + str(const.LEARNING_RATE) + " ,lambda = " + str(
            const.LAMB) + " ,KNN = " + str(const.KNN) + " ,tol = " + str(const.TOL) + " ,cuda = " + const.CUDA_DEVICE
        return Info

    # def getConcat(self, drug_latent, ADR_latent):
    #     # drug_latent: number of drugs * k
    #     # ADR_latent: k * number of ADRs
    #     # return: (number of drugs * number of ADRs) * 2k

    #     drug_latent = drug_latent.cuda(const.CUDA_DEVICE)
    #     ADR_latent = ADR_latent.cuda(const.CUDA_DEVICE)  # 调用Cuda加速效果明显

    #     num_drug = len(drug_latent)  # How many drugs data input
    #     num_ADR = len(ADR_latent)  # How many ADRs

    #     concat_data = torch.Tensor(num_drug * num_ADR, self.embedding_k * 2).cuda(
    #         const.CUDA_DEVICE)  # (2707*batch_size)  * 2k

    #     i = 0
    #     for u in drug_latent:
    #         uu = u.reshape(1, self.embedding_k)
    #         for v in ADR_latent:
    #             vv = v.reshape(1, self.embedding_k)
    #             concat_data[i] = torch.cat([uu, vv], axis=1)
    #             # dot[i] = torch.mm(uu.double(), vv.T.double())
    #             i += 1

    #     return concat_data

    def Embedding(self, x):
        # input: a batch of index list : [index of drug, index of ADR]

        user_idx = torch.LongTensor(x[:, 0]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of drugs from x
        item_idx = torch.LongTensor(x[:, 1]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of ADRs from x

        U_emb = self.W(user_idx)  # embedding: from batch_size to batch_size * k
        V_emb = self.H(item_idx)  # embedding: from batch_size to batch_size * k

        return U_emb, V_emb

    # def Embedding(self, x):
    #     # input: a batch of index list : [index of drug, index of ADR]

    #     user_idx = torch.LongTensor(x[:, 0]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of drugs from x
    #     item_idx = torch.LongTensor(x[:, 1]).cuda(const.CUDA_DEVICE)  # extract the a batch of indexes of ADRs from x

    #     U_emb = self.W(user_idx)  # embedding: from batch_size to batch_size * k
    #     V_emb = self.H(item_idx)  # embedding: from batch_size to batch_size * k

    #     return U_emb, V_emb
        

    def forward(self, x, drug_features_x):
        # embed, concat and DNN
        # batch_size = len(x)

        U_emb, V_emb = self.Embedding(x)

        # z_emb = torch.cat([U_emb, V_emb], axis=1)

        # dnn = self.mlp_deep(z_emb)
        # DrugFeature = DrugFeature.cuda(const.CUDA_DEVICE)

        wide_part = self.mlp_wide(drug_features_x)
        wide_part = torch.mul(wide_part, V_emb)
        # print( wide_part.shape )
        wide_part = torch.sum(wide_part, dim=1).reshape(-1, 1)
        wide_lr = self.lr(drug_features_x)
        wide_part = wide_lr + wide_part

        # wide_part_t = self.linear_g(wide_part_t)

        # wide_part_t = torch.Tensor( [ wide_part[i].dot(V_emb[i]) for i in range(0, batch_size) ]  ).reshape(-1, 1).cuda(const.CUDA_DEVICE)

        # wide_part = torch.mm(self.mlp_wide(DrugFeature), V_emb.T)
        # wide_part_t = torch.Tensor( [ wide_part[i, i] for i in range(0, batch_size) ] ).reshape(-1, 1).cuda(const.CUDA_DEVICE)


        mf_part = torch.mul(V_emb, U_emb)
        mf_part = torch.sum(mf_part, dim=1).reshape(-1, 1)

        # mf_part = self.linear_g(mf_part)

        out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL)*mf_part
        out = self.sigmoid(out.reshape(-1))

        return out

    def fitAndPredict(self, inputTrain, outputTrain, inputTest, outputTest, num_epoch=500, lr=const.LEARNING_RATE, lamb=const.LAMB, tol=const.TOL, verbose=1):  # lambda 1e-4 1e-3 , batch_size=4096
        print(lamb)

        nDrug = len(outputTrain)
        nADR = len(outputTrain[0])

        nDrug_test = len(outputTest)
        nADR_test = len(outputTest[0])

        outputTrain_index = utils.indices_array_generic(nDrug, nADR)
        outputTest_index = utils.indices_array_generic(nDrug_test, nADR_test)

        # An array of output (binary) in order
        outputTrain_value = np.empty(nDrug * nADR)
        for i in range(nDrug):
            for j in range(nADR):
                outputTrain_value[i * nADR + j] = outputTrain[i][j]

        inputTest_value = inputTest[outputTest_index[:, 0]]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-6)
        last_loss = 1e9
        last_auc = 0

        batch_size = nDrug

        num_sample = len(outputTrain_index)
        total_batch = num_sample // batch_size

        inputTrain_cuda = torch.from_numpy(inputTrain).cuda(const.CUDA_DEVICE).long()
        # print(inputTrain_cuda.shape)

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

        inputTest = torch.from_numpy(inputTest).cuda(const.CUDA_DEVICE)

        for epoch in range(num_epoch):

            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            epoch_auc = 0
            for idx in range(total_batch):  # idx:第几个batch

                # print(time.strftime('%Y-%m-%d %H:%M:%S'))

                # mini-batch training
                selected_idx = all_idx[batch_size * idx:(idx + 1) * batch_size]  # 一共有batch_size个index
                sub_x = outputTrain_index[selected_idx]  # batch中drug与对应ADR的索引数据对，size：batch_size * 2
                drug_index = sub_x[:, 0]  # 取出数据对中的drug索引

                drug_features = inputTrain_cuda[drug_index]
                sub_y = outputTrain_value[selected_idx]
                
                  # batch中drug的ADR, size: batch_size, value: bool

                optimizer.zero_grad()
                pred = self.forward(sub_x, drug_features)

                # pred=pred.cuda(const.CUDA_DEVICE)
                # pred = pred.to(const.CPU)

                # target = torch.tensor(sub_y).reshape(-1, len(pred))      # reshape the test ADR
                target = torch.unsqueeze(torch.Tensor(sub_y), 1)
                target = target.cuda(const.CUDA_DEVICE).reshape(-1)
                # xent_loss = self.xent_func(pred.T.float(), target.float()) # require float

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

                # concat_data = self.getConcat(u_emb_weighted, ADRFeatures)

                # dnn = self.mlp_deep(concat_data)

                wide_part = torch.mm(self.mlp_wide(inputTest), ADRFeatures.T).reshape(-1, 1).cuda(const.CUDA_DEVICE)
                # wide_part_t = self.linear_g(wide_part_t)

                # out = wide_part_t + dnn

                mf_part = torch.mm(u_emb_weighted, ADRFeatures.T ).reshape(-1, 1).cuda(const.CUDA_DEVICE)

                out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part

                out = self.sigmoid(out.reshape(-1))

                testOut_vec = outputTest.reshape(-1)
                predictedValues = out.cpu().detach().numpy()
                pred_vec = predictedValues.reshape(-1)
                aucs = roc_auc_score(testOut_vec, pred_vec)
                auprs = average_precision_score(testOut_vec.reshape(-1), predictedValues.reshape(-1))
                auroc_collected.append(aucs)
                auprc_collected.append(auprs)

            relative_loss_div = (last_loss - epoch_loss) / (last_loss + 1e-10)
            if relative_loss_div < tol:
                if early_stop > 5:
                    print("[DrugNCF][Train] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1

            last_loss = epoch_loss

            print("[DrugNCF][In Train] epoch:{}, xent:{}".format(epoch, epoch_loss))

            # if epoch != 0:
            #     chemFeatures = self.W(drug_idx)  # Embeddings of train drugs, to calculate similarity
            #     ADRFeatures = self.H(adr_idx)  # Embeddings of train ADRs, to concat with embeddings of test drugs

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

            #     # concat_data = self.getConcat(u_emb_weighted, ADRFeatures)       # 111

            #     # dnn = self.mlp_deep(concat_data)    # 111

            #     wide_part = torch.mm(self.mlp_wide(inputTest), ADRFeatures.T).reshape(-1, 1).cuda(const.CUDA_DEVICE)
            #     # wide_part_t = self.linear_g(wide_part_t)  # 111

            #     # out = wide_part_t + dnn      # 111

            #     mf_part = torch.mm(u_emb_weighted, ADRFeatures.T ).reshape(-1, 1).cuda(const.CUDA_DEVICE)

            #     out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part

            #     out = self.sigmoid(out.reshape(-1))

            #     testOut_vec = outputTest.reshape(-1)
            #     predictedValues = out.cpu().detach().numpy()
            #     pred_vec = predictedValues.reshape(-1)
            #     auc = roc_auc_score(testOut_vec, pred_vec)
            #     aupr = average_precision_score(testOut_vec.reshape(-1), predictedValues.reshape(-1))

            #     if early_stop_overfit > 5:
            #         print("[DrugNCF][Train] epoch:{}, xent:{}".format(epoch, epoch_loss))
            #         break

            #     if auc < last_auc:
            #         print("overfitting")
            #         early_stop_overfit += 1

            #     last_auc = auc
            #     print("epoch:", epoch, "test auc:", auc, "test aupr: ", aupr)


            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

        if not const.SAVE_TRAJECTORY:
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
            # concat_data = self.getConcat(u_emb_weighted, ADRFeatures)   # 111

            # dnn = self.mlp_deep(concat_data) # 111

            wide_part = torch.mm(self.mlp_wide(inputTest), ADRFeatures.T).reshape(-1, 1).cuda(const.CUDA_DEVICE)
            # wide_part_t = self.linear_g(wide_part_t) # 111

            # out = wide_part_t + dnn # 111
            mf_part = torch.mm(u_emb_weighted, ADRFeatures.T ).reshape(-1, 1).cuda(const.CUDA_DEVICE)

            out = const.LAMBDA_GLOBAL * wide_part + (1 - const.LAMBDA_GLOBAL) * mf_part
            out = self.sigmoid(out.reshape(-1))

        return out.cpu().detach().numpy(), loss_collected, auroc_collected, auprc_collected

