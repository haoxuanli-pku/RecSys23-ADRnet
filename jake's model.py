## New code, NCF for MF prediction
import numpy as np
import utils
import const
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import time

from sklearn import svm
from models import lnsm

class NCF(nn.Module):
    def __init__(self, num_drugs, num_adrs, embedding_k=const.N_FEATURE):
        super(NCF, self).__init__()
        # Number of train drugs
        self.num_drugs = num_drugs
        # Number of train ADRs
        self.num_adrs = num_adrs
        # Latent variable k
        self.embedding_k = embedding_k
        # W -> 1222*k
        self.W = torch.nn.Embedding(self.num_drugs, self.embedding_k)
        # H -> 2707*k
        self.H = torch.nn.Embedding(self.num_adrs, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*2, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()
        # Random initial value
        self.parameter = nn.Linear(1,1).parameters()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        # concat
        z_emb = torch.cat([U_emb, V_emb], axis=1)

        h1 = self.linear_1(z_emb)
        h1 = self.relu(h1)

        out = self.linear_2(h1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4, batch_size=1024, verbose=0):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                optimizer.zero_grad()
                pred, u_emb, v_emb = self.forward(sub_x, True)

                pred = self.sigmoid(pred)

                xent_loss = self.xent_func(pred, torch.unsqueeze(torch.Tensor(sub_y),1))

                loss = xent_loss
                loss.backward()
                optimizer.step()
                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[NCF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[Warning] Reach preset epochs, it seems does not converge.")

    def partial_fit(self, x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4):
        self.fit(x, y, num_epoch=1000, lr=0.05, lamb=0, tol=1e-4)

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy().flatten()

    def repred(self):
        # (1222*2707, 2k) Tensor
        prediction = torch.Tensor(self.num_drugs * self.num_adrs, self.embedding_k * 2)
        i = 0
        for u in self.W.weight:
            uReshape = u.reshape(1, self.embedding_k)
            for v in self.H.weight:
                vReshape = v.reshape(1, self.embedding_k)
                z_emb = torch.cat([uReshape, vReshape], axis=1)
                prediction[i] = z_emb
                i += 1
        h1 = self.linear_1(prediction)
        h1 = self.relu(h1)
        out = self.linear_2(h1)
        out = out.detach().numpy()
        return out

    # def predict_new_drug(self, testFeature, adrFeature):
    #     testFeature = np.asarray(testFeature)
    #     nTest = testFeature.shape[0]
    #     nADR = adrFeature.shape[1]
    #     out = np.empty([nTest, nADR])
    #     for i in range(nTest):
    #         for j in range(nADR):
    #             z_emb = torch.cat([torch.from_numpy(testFeature[i,:]), torch.from_numpy(adrFeature[:,j])])
    #             h1 = self.linear_1(z_emb)
    #             h1 = self.relu(h1)
    #             prediction = self.linear_2(h1)
    #             out[i][j] = prediction
    #     return out

    def predict_new_drug(self, testInput, trainInput):
        trainEmbeddingArray = (self.W.weight).detach().to('cpu').numpy()
        nTest = testInput.shape[0]
        # Similarity calculation, refer to MFModel
        simMatrix = np.ndarray((nTest, self.num_drugs), dtype=float)
        for i in range(nTest):
            for j in range(self.num_drugs):
                simMatrix[i][j] = utils.getTanimotoScore(testInput[i], trainInput[j])

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        args = args[:, :const.KNN]
        testFeatures = []
        for i in range(nTest):
            newF = np.zeros(const.N_FEATURE, dtype=float)
            matches = args[i]
            simScores = simMatrix[i, matches]
            ic = -1
            sum = 1e-10
            for j in matches:
                ic += 1
                newF += simScores[ic] * trainEmbeddingArray[j]
                sum += simScores[ic]
            newF /= sum
            testFeatures.append(newF)
        testVecs = np.array(testFeatures)
        testVecs = torch.from_numpy(testVecs)
        # Similarity calculation done, testVecs -> (136, k) tensor representing new drug embedding

        print('Test ves shape ', testVecs.shape)
        # Predict a 136*2707 output matrix, refer to repred method
        prediction = torch.Tensor(nTest * self.num_adrs, self.embedding_k * 2)
        i = 0
        for u in self.H.weight:
            uReshape = u.reshape(1, self.embedding_k)
            for v in testVecs:
                vReshape = v.reshape(1, self.embedding_k)
                z_emb = torch.cat([uReshape, vReshape], axis=1)
                prediction[i] = z_emb
                i += 1
        h1 = self.linear_1(prediction)
        h1 = self.relu(h1)
        out = self.linear_2(h1)
        out = out.detach().numpy()
        print('Success')
        return out



class MFModel(Model):
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "MF"

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.decomposition import NMF


        self.model = NMF(const.N_FEATURE)
        chemFeatures = self.model.fit_transform(outputTrain)
        adrFeatures = self.model.components_

        nTrain, nTest = intpuTrain.shape[0], inputTest.shape[0]
        # print('Careful, here comes the train and test shape info ',inputTest.shape[1], intpuTrain.shape[1])
        # print('And output shape: ', outputTrain.shape[0], outputTrain.shape[1])
        
        outSize = outputTrain.shape[1]
        # simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        # for i in range(nTest):
        #     for j in range(nTrain):
        #         simMatrix[i][j] = utils.getTanimotoScore(inputTest[i], intpuTrain[j])

        # args = np.argsort(simMatrix, axis=1)[:, ::-1]
        # args = args[:, :const.KNN]
        # # print args
        # # print('Let us see the args: ', len(simMatrix))
        # testFeatures = []
        # for i in range(nTest):
        #     newF = np.zeros(const.N_FEATURE, dtype=float)
        #     matches = args[i]
        #     simScores = simMatrix[i, matches]
        #     ic = -1
        #     sum = 1e-10
        #     for j in matches:
        #         ic += 1
        #         newF += simScores[ic] * chemFeatures[j]
        #         sum += simScores[ic]
        #     newF /= sum
        #     testFeatures.append(newF)
        # testVecs = np.vstack(testFeatures)
        # self.repred = np.matmul(chemFeatures, adrFeatures)
        # # This is with optimization, KNN optimization, find the most KNN similar training drug to calc avg chemFeatures
        # out = np.matmul(testVecs, adrFeatures)

        # Use NCF to predict
        # # ntrain = 1222, outsize = 2707
        ncf = NCF(nTrain, outSize, const.N_FEATURE)
        # An array of all indices in 1222*2707 matrix
        ncfTrainInput = utils.indices_array_generic(nTrain, outSize)
        # An array of output (binary) in order
        ncfTrainOutput = np.empty(nTrain*outSize)
        for i in range(nTrain):
            for j in range(outSize):
                ncfTrainOutput[i*outSize+j] = outputTrain[i][j]
        # Train with the input&output data
        ncf.fit(ncfTrainInput, ncfTrainOutput)
        # Predict new drugs (# = 136)
        out = ncf.predict_new_drug(inputTest, intpuTrain)
        # Repred, see how our embeddings predict train output
        self.repred = ncf.repred()
        # NCF done

        return out

    def getInfo(self):
        return "MF %s" % const.N_FEATURE
