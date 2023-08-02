# from DataFactory import DataLoader, DataLoader2
from copy import deepcopy

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import const


class PredictorWrapper():

    def __init__(self, model=None):
        # self.dataLoader = DataLoader()
        # self.loader2 = DataLoader2()
        pass

    def __getMeanSE(self, ar):
        mean = np.mean(ar)
        se = np.std(ar) / np.sqrt(len(ar))
        return mean, se

    def evalAModel(self, model):
        # print model.getInfo()
        import time

        from dataProcessor import DataFactory
        from dataProcessor.DataFactory import GenAllData
        from logger.logger2 import MyLogger
        logger = MyLogger("results/logs_%s.dat" % model.name)
        logger.infoAll("K-Fold data folder: %s" % const.CURRENT_KFOLD)
        logger.infoAll("Feature mode: %s" % const.FEATURE_MODE)
        logger.infoAll("Model: %s" % model.name)
        logger.infoAll("Format: AUC STDERR AUPR STDERR")

        logger.infoAll(model.getInfo())

        dataLoader = GenAllData()

        arAuc = []
        arAupr = []
        trainAucs = []
        trainAuprs = []
        runningTimes = []

        loss_all = []
        auroc_all = []
        auprc_all = []

        modelsaved = deepcopy(model)
        lamb = const.getLambda(model.name, const.CURRENT_DATA)

        for i in range(10):
        # for i in [1]:
            start = time.time()
            datas = dataLoader.loadFold(i)
            trainInp, trainKGInp, trainOut, testInp, testKGInp, testOut = datas[1], datas[2], datas[3], datas[5], datas[
                6], datas[7]

            if const.FEATURE_MODE == const.BIO2RDF_FEATURE: #将原input映射为高维0-1矩阵
                trainInp = DataFactory.convertBioRDFSet2Array(trainKGInp)
                testInp = DataFactory.convertBioRDFSet2Array(testKGInp)
            elif const.FEATURE_MODE == const.COMBINE_FEATURE:
                trainInp2 = DataFactory.convertBioRDFSet2Array(trainKGInp)
                testInp2 = DataFactory.convertBioRDFSet2Array(testKGInp)

                trainInp = np.concatenate([trainInp, trainInp2], axis=1)
                testInp = np.concatenate([testInp, testInp2], axis=1)

            print(trainInp.shape, trainOut.shape, testInp.shape, testOut.shape)

            # lamb = const.getLambda(model.name, const.CURRENT_DATA)

            if model.name == "CNN":
                predictedValues = model.fitAndPredict(i)
            elif model.name == "SCCA":
                if const.FEATURE_MODE != 2:
                    logger.infoAll(
                        "Error: Input data for SCCA is only currently generated with FEATURE_MODE = 2. Please run R script.")
                    exit(-1)
                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, i)
            elif model.name == "KGSIM":
                if const.FEATURE_MODE != const.BIO2RDF_FEATURE:
                    logger.infoAll("Fatal error: KGSIM only runs with const.FEATURE_MODE == const.BIO2RDF_FEATURE. "
                                   "Current mode is const.CHEM_FEATURE.")
                    exit(-1)
                predictedValues = model.fitAndPredict(trainKGInp, trainOut, testKGInp)

            elif model.name == "FM":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, lamb=const.FM_LAMB)

            elif model.name == "FFM":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, lamb=const.FFM_LAMB)

            elif model.name == "LR":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, lamb=const.LR_LAMB)

            elif model.name == "LRFC":

                model = model.cuda(const.CUDA_DEVICE)

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, lamb=const.LR_LAMB)

            elif model.name == "LSPLM":

                model = model.cuda(const.CUDA_DEVICE)

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, lamb=const.LSPLM_LAMB)

            elif model.name == "POLY2":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, lamb=const.POLY_LAMB)

            elif model.name == "CF":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp)

            elif model.name == "GBDT":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp)

            elif model.name == "GBDTLR":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp)

            elif model.name == "AutoRec":

                model = model.cuda(const.CUDA_DEVICE)

                predictedValues = model.fitAndPredict(trainOut, testOut).cpu().detach().numpy()

            elif model.name == "WideAndDeep":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=lamb)

            elif model.name == "DeepAndCross":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=lamb)

            elif model.name == "DNN":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut)

            elif model.name == "DeepCrossing":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=lamb)

            elif model.name == "PNN":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=lamb)

            elif model.name == "FNN":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=lamb)

            elif model.name == "NFM":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=lamb)

            elif model.name == "DeepFM":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=lamb)

            elif model.name == "AFM":

                predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=lamb)

            elif model.name == "NCF" or model.name == "MFModel":

                model = model.cuda(const.CUDA_DEVICE)

                model.fit(trainInp, trainOut, testInp, testOut, lamb=lamb)
                predictedValues = model.predict(trainInp, trainOut, testInp)
            
            elif model.name == 'DrugNCF' or model.name == 'DrugNCFwoshare':

                model = model.cuda(const.CUDA_DEVICE)
                predictedValues, loss_collected, auroc_collected, auprc_collected = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb=const.LAMB)

                loss_all.append(loss_collected)
                auroc_all.append(auroc_collected)
                auprc_all.append(auprc_collected)

                # if const.FEATURE_MODE == 0 or const.FEATURE_MODE == 1:
                #     predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb = const.LAMB, batch_size = 8192)
                # elif const.FEATURE_MODE == 2:
                #     predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut, lamb = const.LAMB, batch_size = 8192)
                # else:
                #     print("const FEARURE MODE should be contained in 0, 1, 2; please check again!")
                #     exit(-1)
            
            else:
                if model.isFitAndPredict:
                    if model.name == "NeuN":
                        predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, testOut)
                    elif model.name == 'SCCA':
                        predictedValues = model.fitAndPredict(trainInp, trainOut, testInp, i)

                    else:
                        predictedValues = model.fitAndPredict(trainInp, trainOut, testInp)   # MF
                else:
                    model.fit(trainInp, trainOut)
                    predictedValues = model.predict(testInp)

                if model.name == "NeuN":
                    predictedValues = predictedValues[-1]

            end = time.time()
            elapsed = end - start
            runningTimes.append(elapsed)


            testOut_vec = testOut.reshape(-1)
            pred_vec = predictedValues.reshape(-1)
            aucs = roc_auc_score(testOut_vec, pred_vec)
            auprs = average_precision_score(testOut_vec.reshape(-1), predictedValues.reshape(-1))
            print(aucs, auprs)
            print(elapsed)
            print(time.strftime('%Y-%m-%d %H:%M:%S'))
            logger.infoAll("Fold : %s" % i)
            logger.infoAll("Test : %s %s" % (aucs,auprs))
            # logger.infoAll("Train: %s %s %s %s" % (meanTrainAUC, seTranAUC, meanTranAUPR, seTrainAUPR))
            logger.infoAll("Running time: %s" % elapsed)
            # if model.name == "KNN":
            #     model.repred = model.fitAndPredict(trainInp, trainOut, trainInp)
            # if model.name == "NCF":
            #     model.repred = model.predict(trainInp, trainOut, trainInp)
            # trainAUC = roc_auc_score(trainOut.reshape(-1), model.repred.reshape(-1))
            # trainAUPR = average_precision_score(trainOut.reshape(-1), model.repred.reshape(-1))
            # trainAucs.append(trainAUC)
            # trainAuprs.append(trainAUPR)
            # print(trainAUC, trainAUPR)
            # print(time.strftime('%Y-%m-%d %H:%M:%S'))
            #if (model.name == "SCCA"):
            #    exit(-1)
            arAuc.append(aucs)
            arAupr.append(auprs)

            model = deepcopy(modelsaved)
        
        # loss_all = [[1, 2, 3, 4], [2, 3, 6], [1, 2, 5]]
        max_len = np.max( np.array( [ len(xs) for xs in loss_all ] ) )
        for ik in range(len(loss_all)):
            loss_all[ik].extend([np.nan for _ in range(max_len - len(loss_all[ik]) ) ])
            if const.SAVE_TRAJECTORY:
                auroc_all[ik].extend([np.nan for _ in range(max_len - len(auroc_all[ik]) ) ])
                auprc_all[ik].extend([np.nan for _ in range(max_len - len(auprc_all[ik]) ) ])
        
        loss_all = np.array(loss_all).T
        if const.SAVE_TRAJECTORY:
            auroc_all = np.array(auroc_all).T
            auprc_all = np.array(auprc_all).T

        # np.savetxt( "trajectory/model_" + model.name + "_data_" + const.CURRENT_DATA + "_embed_" + str(const.N_FEATURE) + "_" + const.CHANGE_LAYER + "_layer_"  + str(const.N_LAYERS) + "_featuremode_" + str(const.FEATURE_MODE) + ".txt", loss_all, delimiter = ',')
        np.savetxt( "trajectory/model_" + model.name + "_data_" + const.CURRENT_DATA + "_embed_" + str(const.N_FEATURE) + "_cf layer_"  + str(const.N_CF_LAYERS) + "_widelayer_" + str(const.N_WIDE_LAYERS) + "_featuremode_" + str(const.FEATURE_MODE) + ".txt", loss_all, delimiter = ',')

        if const.SAVE_TRAJECTORY:
            np.savetxt( "auprc/model_" + model.name + "_data_" + const.CURRENT_DATA + "_embed_" + str(const.N_FEATURE) + "_cf layer_"  + str(const.N_CF_LAYERS) + "_widelayer_" + str(const.N_WIDE_LAYERS) + "_featuremode_" + str(const.FEATURE_MODE) + ".txt", auprc_all, delimiter = ',')
            np.savetxt( "auroc/model_" + model.name + "_data_" + const.CURRENT_DATA + "_embed_" + str(const.N_FEATURE) + "_cf layer_"  + str(const.N_CF_LAYERS) + "_widelayer_" + str(const.N_WIDE_LAYERS) + "_featuremode_" + str(const.FEATURE_MODE) + ".txt", auroc_all, delimiter = ',')

        meanAuc, seAuc = self.__getMeanSE(arAuc)
        meanAupr, seAupr = self.__getMeanSE(arAupr)
        #
        # meanTrainAUC, seTranAUC = self.__getMeanSE(trainAucs)
        # meanTranAUPR, seTrainAUPR = self.__getMeanSE(trainAuprs)
        #
        meanTime, stdTime = self.__getMeanSE(runningTimes)

        # logger.infoAll(model.name)
        # logger.infoAll(model.getInfo())
        # logger.infoAll((trainInp.shape, testOut.shape))
        logger.infoAll("Test : %s %s %s %s" % (meanAuc, seAuc, meanAupr, seAupr))
        logger.infoAll("Test : %.2f±%.2f, %.2f±%.2f" % (meanAuc*100, seAuc*100, meanAupr*100, seAupr*100))
        # with open("final_results/model_" + model.name + "_data_" + const.CURRENT_DATA + "_embed_" + str(const.N_FEATURE) + "_" + const.CHANGE_LAYER + "_layer_"  + str(const.N_LAYERS) + "_featuremode_" + str(const.FEATURE_MODE) + ".txt", 'w') as f:
        #     f.write("Test : %.2f±%.2f, %.2f±%.2f" % (meanAuc*100, seAuc*100, meanAupr*100, seAupr*100))
        with open("final_results/model_" + model.name + "_data_" + const.CURRENT_DATA + "_embed_" + str(const.N_FEATURE) + "_deeplayer_"  + str(const.N_CF_LAYERS) + "_widelayer_" + str(const.N_WIDE_LAYERS) + "_featuremode_" + str(const.FEATURE_MODE) + ".txt", 'w') as f:
            f.write("Test : %.2f±%.2f, %.2f±%.2f" % (meanAuc*100, seAuc*100, meanAupr*100, seAupr*100))

        # # logger.infoAll("Train: %s %s %s %s" % (meanTrainAUC, seTranAUC, meanTranAUPR, seTrainAUPR))
        logger.infoAll("Avg running time: %s %s" % (meanTime, stdTime))
        return meanAuc, seAuc, meanAupr, seAupr
