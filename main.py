from optparse import OptionParser

import const
from models.DrugNCF import DrugNCF, DrugNCFwoshare
from models.models import (AFM, CF, DNN, FFM, FM, FNN, GBDT, GBDTLR, KGSIM,
                           KNN, LR, LRFC, LSPLM, MF, NCF, NFM, PNN, POLY2,
                           AutoRec, CCAModel, DeepAndCross, DeepCrossing,
                           DeepFM, GBModel, LogisticModel, MFModel, MultiSVM,
                           NeuNModel, RandomModel, RFModel, RSCCAModel,
                           WideAndDeep)
from predictorWrapper import PredictorWrapper


def runSVM():
    wrapper = PredictorWrapper()
    PLIST = [i for i in range(1, 2)]
    for p in PLIST:
        const.SVM_C = p
        model = MultiSVM()
        print(wrapper.evalAModel(model))


def runRF():
    wrapper = PredictorWrapper()
    PLIST = [10 * i for i in range(1, 10)]

    for p in PLIST:
        const.RF = p
        model = RFModel()
        print(wrapper.evalAModel(model))


def runGB():
    wrapper = PredictorWrapper()
    PLIST = [60 * i for i in range(1, 2)]

    for p in PLIST:
        const.RF = p
        model = GBModel()
        print(wrapper.evalAModel(model))


def runKNN():
    wrapper = PredictorWrapper()
    KLIST = [10 * i for i in range(1, 10)]
    for k in KLIST:
        const.KNN = k
        model = KNN()
        print(wrapper.evalAModel(model))


def runKGSIM():
    wrapper = PredictorWrapper()
    KLIST = [20 * i for i in range(1, 2)]
    for k in KLIST:
        const.KGSIM = k
        model = KGSIM()
        print(wrapper.evalAModel(model))


def runCCA():
    wrapper = PredictorWrapper()
    NCLIST = [10 * i for i in range(1, 10)]
    for c in NCLIST:
        const.CCA = c
        model = CCAModel()
        print(wrapper.evalAModel(model))


def runSCCA():
    wrapper = PredictorWrapper()
    NCLIST = [10 * i for i in range(1, 2)]
    for c in NCLIST:
        const.CCA = c
        model = RSCCAModel()
        print(wrapper.evalAModel(model))


def runRandom():
    wrapper = PredictorWrapper()
    model = RandomModel()
    print(wrapper.evalAModel(model))


def runGBDT():
    wrapper = PredictorWrapper()
    model = GBDT()
    print(wrapper.evalAModel(model))


def runGBDTLR():
    wrapper = PredictorWrapper()
    model = GBDTLR()
    print(wrapper.evalAModel(model))


def runCF():
    wrapper = PredictorWrapper()
    model = CF()
    print(wrapper.evalAModel(model))


def runMF():
    wrapper = PredictorWrapper()
    # KLIST = [10 * i for i in range(1, 10)]
    # BLIST = [-4, -2, 0, 2, 4]
    # for k in KLIST:
    #     const.N_FEATURE = k
    #     model = MFModel()
    #     print(wrapper.evalAModel(model))
    model = MF()
    print(wrapper.evalAModel(model))


def runMFModel():
    wrapper = PredictorWrapper()
    model = MFModel(1222, 2707)
    print(wrapper.evalAModel(model))


def runFM():
    wrapper = PredictorWrapper()

    model = FM(field_dims)
    print(wrapper.evalAModel(model))


def runFFM():
    wrapper = PredictorWrapper()

    model = FFM(field_dims, 16)
    print(wrapper.evalAModel(model))

def runLR():
    wrapper = PredictorWrapper()

    model = LR(field_dims)
    print(wrapper.evalAModel(model))


# a Fully connected layer to implement multinomial classification
def runLRFC():
    wrapper = PredictorWrapper()

    # model = LR(field_dims)
    model = LRFC(field_dims, num_ADR)
    print(wrapper.evalAModel(model))


def runLSPLM():
    wrapper = PredictorWrapper()
    model = LSPLM(field_dims, const.LS_PLM_M)
    print(wrapper.evalAModel(model))


def runPOLY2():
    wrapper = PredictorWrapper()
    model = POLY2(field_dims)
    print(wrapper.evalAModel(model))


def runAutoRec():
    wrapper = PredictorWrapper()
    model = AutoRec(2707, 100)
    print(wrapper.evalAModel(model))


def runNCF():
    wrapper = PredictorWrapper()

    const.N_FEATURE_NCF = 80
    const.KNN = 65
    const.LAMB = 1.5e-6

    const.LEARNING_RATE = 0.05
    const.TOL = 1e-4

    model = NCF(1222, 2707, embedding_k=const.N_FEATURE_NCF) 
    print(wrapper.evalAModel(model))


def runWideAndDeep():

    if const.JOINT == 1:
        wrapper = PredictorWrapper()
        model = WideAndDeep(field_dims, embed_dim=512, mlp_dims=(256, 128), dropout=0.2)
        print(wrapper.evalAModel(model))

    else:
        wrapper = PredictorWrapper()
        model = WideAndDeep(field_dims, embed_dim=40, mlp_dims=(8, 8), dropout=0.2)
        print(wrapper.evalAModel(model))

    # lamblist = [1e-7,1e-8]
    # for lamb in lamblist:
    #     const.WideAndDeep_LAMB = lamb
    #     model = WideAndDeep(field_dims, embed_dim=8, mlp_dims=(8, 8), dropout=0.2)
    #     print(wrapper.evalAModel(model))


def runDeepAndCross():
    if const.JOINT == 1:
        wrapper = PredictorWrapper()
        model = DeepAndCross(field_dims, embed_dim=128, num_layers=2, mlp_dims=(128, 256), dropout=0.2)
        print(wrapper.evalAModel(model))

    else:
        wrapper = PredictorWrapper()
        model = DeepAndCross(field_dims, embed_dim=40, num_layers=2, mlp_dims=(8, 8), dropout=0.2)
        print(wrapper.evalAModel(model))


def runDNN():
    wrapper = PredictorWrapper()
    model = DNN(field_dims, embed_dim=512, mlp_dims=(512, num_ADR), dropout=0.2)
    print(wrapper.evalAModel(model))


def runDeepCrossing():
    if const.JOINT == 1:
        wrapper = PredictorWrapper()
        model = DeepCrossing(field_dims, embed_dim=256, mru_dims=(128, 128), dropout=0.2)
        print(wrapper.evalAModel(model))

    else:
        wrapper = PredictorWrapper()
        model = DeepCrossing(field_dims, embed_dim=8, mru_dims=(56, 56), dropout=0.2)
        print(wrapper.evalAModel(model))


def runPNN():
    if const.JOINT == 1:
        wrapper = PredictorWrapper()
        model = PNN(field_dims, embed_dim=512, mlp_dims=(256, 128), dropout=0.2, method='inner')
        print(wrapper.evalAModel(model))

    else:
        wrapper = PredictorWrapper()
        model = PNN(field_dims, embed_dim=8, mlp_dims=(8, 8), dropout=0.2, method='inner')
        print(wrapper.evalAModel(model))


def runFNN():
    if const.JOINT == 1:
        wrapper = PredictorWrapper()
        model = FNN(field_dims, embed_dim=512, mlp_dims=(256, 128), dropout=0.2)
        print(wrapper.evalAModel(model))
    else:
        wrapper = PredictorWrapper()
        model = FNN(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
        print(wrapper.evalAModel(model))


def runNFM():
    if const.JOINT == 1:
        wrapper = PredictorWrapper()
        model = NFM(field_dims, embed_dim=512, mlp_dims=(256, 128), dropouts=(0.2, 0.2))
        print(wrapper.evalAModel(model))
    else:
        wrapper = PredictorWrapper()
        model = NFM(field_dims, embed_dim=16, mlp_dims=(16, 16), dropouts=(0.2, 0.2))
        print(wrapper.evalAModel(model))


def runDeepFM():
    if const.JOINT == 1:
        wrapper = PredictorWrapper()
        model = DeepFM(field_dims, embed_dim=128, mlp_dims=(1024, 512), dropout=0.2)
        print(wrapper.evalAModel(model))
    else:
        wrapper = PredictorWrapper()
        model = DeepFM(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
        print(wrapper.evalAModel(model))


def runAFM():
    if const.JOINT == 1:
        wrapper = PredictorWrapper()
        model = AFM(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
        print(wrapper.evalAModel(model))
    else:
        wrapper = PredictorWrapper()
        model = AFM(field_dims, embed_dim=4, attn_size=4, dropouts=(0.2, 0.2))
        print(wrapper.evalAModel(model))


def runNeu():
    wrapper = PredictorWrapper()
    PLIST = [10 * i for i in range(1, 2)]
    for p in PLIST:
        # const.NeuN_H1 = p
        model = NeuNModel()
        print(wrapper.evalAModel(model))


def runLLR():
    wrapper = PredictorWrapper()
    PLIST = [10]
    for p in PLIST:
        const.SVM_C = p
        model = LogisticModel()
        print(wrapper.evalAModel(model))


def runCNN():
    from models.models import CNNModel
    wrapper = PredictorWrapper()
    model = CNNModel()
    print(wrapper.evalAModel(model))


def runLNSM():
    from models.models import LNSMModel
    wrapper = PredictorWrapper()
    PLIST = [i * 10 for i in range(1, 10)]
    for p in PLIST:
        const.KNN = p
        model = LNSMModel()
        print(wrapper.evalAModel(model))


def runDrugNCF():
    wrapper = PredictorWrapper()
    const.SAVE_TRAJECTORY = 1

    const.KNN = 65

    # const.LAMB = 1.5e-6 # NCF: default, lambda -> larger
    # const.LEARNING_RATE = 0.005 # 0.001 (adjust the const.LAMB to 1e-5, 1e-4, ...)
    const.TOL = 1e-4

    if const.CURRENT_DATA == 'Liu':
        const.LAMB = 1e-5
        const.LEARNING_RATE = 0.001
    else:
        const.LAMB = 3e-5  # 5e-4
        const.LEARNING_RATE = 0.0075  # 0.0075

    print('Data: ' + options.data)
    # if const.CHANGE_LAYER == 'deep':
    #     mlp_dims_wide = tuple()
    #     num_deep_layers = const.N_LAYERS
    # elif const.CHANGE_LAYER == 'wide':
    #     mlp_dims_wide = tuple([ const.N_FEATURE for _ in range(const.N_LAYERS) ])
    #     num_deep_layers = 4
    # else:
    #     print('You should only select from deep and wide')
    #     exit(-1)

    # Liu: k = 256, mlp_dims_wide=(512, ) lamb = 3e-6, LAMBDA = 0.5 AUC: 92

    num_wide_layers = const.N_WIDE_LAYERS
    num_cf_layers = const.N_CF_LAYERS
    embed_size = const.N_FEATURE
    # mlp_dims_wide = tuple(  [ embed_size*(2**layer) for layer in range(1, num_deep_layers + 1) ])

    # default

    # num_wide_layers_list = [4]
    # num_cf_layers_list = [0, 1, 2, 3, 4]
    #
    # for num_wide_layers in num_wide_layers_list:
    #     const.N_WIDE_LAYERS = num_wide_layers
    #     for num_cf_layers in num_cf_layers_list:
    #         const.N_CF_LAYERS = num_cf_layers
    #         if const.CURRENT_DATA == 'Liu':
    #             model = DrugNCF(field_dims, embed_dim=embed_size, mlp_dims_wide=[512 for i in range(num_wide_layers)],
    #                             mlp_dims_cf=[embed_size for i in range(num_cf_layers)], num_users=num_DRUG, num_items=num_ADR,
    #                             dropout=0.2)
    #         else:
    #             model = DrugNCF(field_dims, embed_dim=embed_size, mlp_dims_wide=[128 for i in range(num_wide_layers)],
    #                             mlp_dims_cf=[embed_size for i in range(num_cf_layers)], num_users=num_DRUG, num_items=num_ADR,
    #                             dropout=0.2)
    #
    #         print(wrapper.evalAModel(model))

    if const.CURRENT_DATA == 'Liu':
        model = DrugNCF(field_dims, embed_dim=embed_size, mlp_dims_wide=[512 for i in range(num_wide_layers)],
                        mlp_dims_cf=[embed_size for i in range(num_cf_layers)], num_users=num_DRUG, num_items=num_ADR,
                        dropout=0.2)
    else:
        model = DrugNCF(field_dims, embed_dim=embed_size, mlp_dims_wide=[128 for i in range(num_wide_layers)],
                        mlp_dims_cf=[embed_size for i in range(num_cf_layers)], num_users=num_DRUG, num_items=num_ADR,
                        dropout=0.2)

    print(wrapper.evalAModel(model))

    '''
        SOTA: AEOLUS: both dims 0, lamb=5e-4, lr=0.0075  AUC=90.86
                      ncf dim1, wide1, lamb=3e-5, lr=0.0075 AUC=91.19
                      
              Liu: 1e-5
                
    '''


def runDrugNCFwoshare():
    wrapper = PredictorWrapper()

    const.KNN = 65
    const.LAMB = 1.5e-6 # NCF: default, lambda -> larger
    const.LEARNING_RATE = 0.05 # 0.001 (adjust the const.LAMB to 1e-5, 1e-4, ...)
    const.TOL = 1e-4

    print('Data: ' + options.data)
    # if const.CHANGE_LAYER == 'deep':
    #     mlp_dims_wide = tuple()
    #     num_deep_layers = const.N_LAYERS
    # elif const.CHANGE_LAYER == 'wide':
    #     mlp_dims_wide = tuple([ const.N_FEATURE for _ in range(const.N_LAYERS) ])
    #     num_deep_layers = 1
    # else:
    #     print('You should only select from deep and wide')
    #     exit(-1)

    num_deep_layers = const.N_DEEP_LAYERS
    mlp_dims_wide = tuple([ const.N_FEATURE for _ in range(const.N_WIDE_LAYERS) ])
    
    model = DrugNCFwoshare(field_dims, embed_dim = const.N_FEATURE, mlp_dims_wide = mlp_dims_wide, num_deep_layers=num_deep_layers, num_users = num_DRUG, num_items = num_ADR, dropout = 0.2)
    print(wrapper.evalAModel(model))


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("-m", "--model", dest="modelName", type='string', default="LNSM",
                      help="MODELNAME:\n"
                      # "KNN: k-nearest neighbor,\n"
                           "LNSM: linear neighbor similarity,\n"
                           "CCA: canonical correlation analysis,\n"
                           "RF: random forest,\n"
                           "SVM: support vector machines,\n"
                      # "GB: gradient boosting,\n"
                           "LR: logistic regression,\n"
                           "MF: matrix factorization,\n"
                           "NCF: Network Collaborative Filtering,\n"
                           "FM: Factorization Machine,\n"
                           "MLN: multilayer feedforward neural network,\n"
                           "DCN: neural fingerprint model [default: %default]")
    parser.add_option("-d", "--data", dest="data", type='string', default="AEOLUS", help="data: Liu, AEOLUS [default: "
                                                                                         "%default]")
    parser.add_option("-g", "--gen", dest="gen", action='store_true', default=False,
                      help="generate combined training data for R script and exit.")
    parser.add_option("-i", "--init", dest="init", action='store_true', default=False)
    parser.add_option("-f", "--feature", dest="feature", type='int', default=2,
                      help='feature: 0 PubChem, 1 ChemBio, 2 Combination'
                           '[default: %default]. '
                           'DCN is assigned to 2DChem  '
                           'descriptors. ')
    parser.add_option("-j", "--joint", dest="joint", type='int', default=0,
                      help='joint: 0 predict by each ADR, 1 jointly predict all ADRs')
    # parser.add_option("-c", "--cuda", dest="cuda", type='int', default=1,
    #                   help='cuda: 0 train at cuda:0, 1 train at cuda:1')
    parser.add_option("-k", "--embedding_size", dest="embedding_size", type='int', default=64,
                    help='embedding_size')
    parser.add_option("-c", "--cf_layers", dest="cf_num_layers", type='int', default=1,
                    help='number of layers in MLP(deep)')
    parser.add_option("-w", "--wide_layers", dest="wide_num_layers", type='int', default=0,
                    help='number of layers in MLP(wide)')
    parser.add_option("-p", "--lambda_global", dest="lambda_global", type='float', default=1.0,
                    help='the weight of the wide and deep part')
    parser.add_option("-t", "--save_trajectory", dest="save_trajectory", action='store_true', default=False,
                    help='whether to save the trajectory of the AUROC and AUPRC')
        
    # parser.add_option("-x", "--change_layer", dest="change_layer", type='string', default='deep',
    #                 help='vary the layers in deep NCF part while not the wide MF part ')

    (options, args) = parser.parse_args()

    init = options.init

    if init == True:
        from dataProcessor import DataFactory

        DataFactory.genKFoldECFPLiu()  # 生成K折Liu数据
        DataFactory.genKFoldECFPAEOLUS()  # 生成K折AEOLUS数据
        print("Generating %s-Fold data completed.\n" % const.KFOLD)
        exit(-1)
    if options.gen:
        from dataProcessor import DataFactory

        DataFactory.genCombineData()
        print("Generating combined data completed.\n")
        exit(-1)

    if options.data == "Liu":
        const.CURRENT_KFOLD = const.KFOLD_FOLDER_EC_Liu
        const.CURRENT_DATA = "Liu"
        num_DRUG = 828
        num_ADR = 1385
    elif options.data == "AEOLUS":
        const.CURRENT_KFOLD = const.KFOLD_FOLDER_EC_AEOLUS
        const.CURRENT_DATA = "AEOLUS"
        num_DRUG = 1358
        num_ADR = 2707
    else:
        print("Fatal error: Unknown data. Only"
              " Liu and AEOLUS datasets are supported.")

    modelName = options.modelName
    const.FEATURE_MODE = options.feature
    const.JOINT = options.joint
    const.N_FEATURE = options.embedding_size
    const.N_CF_LAYERS = options.cf_num_layers
    const.N_WIDE_LAYERS = options.wide_num_layers
    const.SAVE_TRAJECTORY = options.save_trajectory
    const.LAMBDA_GLOBAL = options.lambda_global

    # const.CHANGE_LAYER = options.change_layer

    # if options.cuda == 0:
    #     const.CUDA_DEVICE = "cuda:0"
    # else:
    #     const.CUDA_DEVICE = "cuda:1"

    if const.FEATURE_MODE == const.BIO2RDF_FEATURE:  # 设置field_dims
        field_dims = [6712]
    elif const.FEATURE_MODE == const.CHEM_FEATURE:
        field_dims = [881]
    elif const.FEATURE_MODE == const.COMBINE_FEATURE:
        field_dims = [881, 6712]

    # if modelName == "KNN":
    #    runKNN()
    if modelName == "LNSM":
        runLNSM()
    # elif modelName == "KGSIM":
    #    runKGSIM()
    elif modelName == "CCA":
        runCCA()
    elif modelName == "RF":
        runRF()
    elif modelName == "SVM":
        runSVM()
    elif modelName == "RD":
        runRandom()
    elif modelName == "MLN":
        runNeu()
    elif modelName == "GB":
        runGB()
    # elif modelName == "SCCA":
    #     runSCCA()
    elif modelName == "CF":
        runCF()
    elif modelName == "MF":
        runMF()
    elif modelName == "MFModel":
        runMFModel()
    elif modelName == "NCF":
        runNCF()
    elif modelName == "POLY2":
        runPOLY2()
    elif modelName == "FM":
        runFM()
    elif modelName == "FFM":
        runFFM()
    elif modelName == "LR":
        runLR()
    elif modelName == "LRFC":
        runLRFC()
    elif modelName == "LSPLM":
        runLSPLM()
    elif modelName == "LLR":
        runLLR()
    elif modelName == "GBDT":
        runGBDT()
    elif modelName == "GBDTLR":
        runGBDTLR()
    elif modelName == "AutoRec":
        runAutoRec()
    elif modelName == "WideAndDeep":
        runWideAndDeep()
    elif modelName == "DeepAndCross":
        runDeepAndCross()
    elif modelName == "DNN":
        runDNN()
    elif modelName == "DeepCrossing":
        runDeepCrossing()
    elif modelName == "PNN":
        runPNN()
    elif modelName == "FNN":
        runFNN()
    elif modelName == "NFM":
        runNFM()
    elif modelName == "DeepFM":
        runDeepFM()
    elif modelName == "AFM":
        runAFM()
    elif modelName == "CNN":
        runCNN()
    elif modelName == 'DrugNCF':
        runDrugNCF()
    elif modelName == 'DrugNCFwoshare':
        runDrugNCFwoshare()
    else:
        print("Method named %s is unimplemented." % modelName)
