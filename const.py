import os

import torch

CPU = torch.device('cpu')

CUDA_DEVICE ='cuda:0'
GPUS = ['cuda:0', 'cuda:1']

LAMBDA_GLOBAL = 1

N_FEATURE = 64
N_CF_LAYERS = 0
N_WIDE_LAYERS = 0
SAVE_TRAJECTORY = False
# N_LAYERS = 1
# CHANGE_LAYER = 'deep'

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_FOLDER = "%s/data" % CUR_DIR
RSCCA_DATA_DIR = "%s/data/RSCCA" % CUR_DIR


# DATA_PATH = "%s/random_group_cv_data.indication"%DATA_FOLDER
# KFOLD_FOLDER = "%s/kfolds"%DATA_FOLDER

LIU_DATA_ECFP_FOLDER = "%s/Liu_Data" % DATA_FOLDER
AEOLUS_ROOT_DATA = "%s/AEOLUS_Data" % DATA_FOLDER

CURRENT_DATA = "AEOLUS"

LAYER_1_SIZE = 64
LAYER_2_SIZE = 32

LEARNING_RATE = 1e-4
CUT = 10
LAMB = 0

KFOLD_FOLDER_EC_Liu = "%s/kfolds_ec_liu" % DATA_FOLDER
KFOLD_FOLDER_EC_AEOLUS = "%s/kfolds_ec_aeolus" % DATA_FOLDER

LIU_ADR_PATH = "%s/ECFPLiuData.dat" % LIU_DATA_ECFP_FOLDER
LIU_ECFP_PATH = "%s/ECFPFeature.dat" % LIU_DATA_ECFP_FOLDER
LIU_BIO2RDF_PATH = "%s/LiuBioRDFFeature.dat" % LIU_DATA_ECFP_FOLDER
LIU_INFO = "%s/ECFP.info" % LIU_DATA_ECFP_FOLDER


AEOLUS_ADR_PATH = "%s/AEOLUS_FinalDrugADR.tsv" % AEOLUS_ROOT_DATA
AEOLUS_CHEM_PATH = "%s/drugCidInfo.dat_Fix" % AEOLUS_ROOT_DATA
AEOLUS_BIO2RDF_PATH = "%s/drugBio2RDF.dat" % AEOLUS_ROOT_DATA
AEOLUS_ECFP_PATH = "%s/AEOUS_Feature.dat" % AEOLUS_ROOT_DATA
AEOLUS_INFO = "%s/AEOLUS_ECFP.info" % AEOLUS_ROOT_DATA


BIO2RDF_FOLDER = "%s/Bio2RDF" % DATA_FOLDER
BIO2RDF_INFO = "%s/Bio2RDFInfo.txt" % BIO2RDF_FOLDER
BIO2RDF_DRUG_TRIPLE_PATH = "%s/Bio2RDFDrugTriple.txt" % BIO2RDF_FOLDER
BIO2RDF_FEATURE_PATH = "%s/Bio2RDFDrugFeature.txt" % BIO2RDF_FOLDER
#DATA_ROOT_2 = "/home/anhnd/DTI Project/Codes/BioDataLoader/out/data"


EC_TRAIN_INP_DATA_INDEX = 0
EC_TRAIN_OUT_DATA_INDEX = 3

EC_TEST_INP_DATA_INDEX = 4
EC_TEST_OUT_DATA_INDEX = 7

KFOLD = 10
TRAIN_PREFIX = "train_"
TEST_PREFIX = "test_"

TRAIN_PREFIX_EC = "train_ec_"
TEST_PREFIX_EC = "test_ec_"

CF_KNN = 10
MF_KNN = 80
KNN = 90
KNN_SIM = 2
KGSIM = 60

RF = 10
CCA = 50
LS_PLM_M = 7

SVM_PARALLEL = True
N_PARALLEL = 18

#NeuN_H1 = 250
#NeuN_H2 = 250

NeuN_H1 = 1000
NeuN_H2 = 800

NeuIter = 150
LEARNING_RATE = 0.005
SVM_C = 1
N_FEATURE_MF = 60
N_FEATURE_NCF = 80
BATCH_SIZE = 8192
TOL = 1e-2
ALPHA = 0.1


CH_NUM_1 = 100
CH_NUM_2 = 80
CH_NUM_3 = 80
CH_NUM_4 = 60
FINGER_PRINT_SIZE = 50
CNN_MAX_ITER = 150000
CNN_LB_1 = 0.0001
CNN_LB_2 = 0.01

CURRENT_KFOLD = KFOLD_FOLDER_EC_AEOLUS  # KFOLD_FOLDER_EC_Liu


CHEM_FEATURE = 0
BIO2RDF_FEATURE = 1
COMBINE_FEATURE = 2

JOINT = 0

NUM_BIO2RDF_FEATURE = 6712

FEATURE_MODE = BIO2RDF_FEATURE

EPOCH = 500

POLY_LAMB = 1e-6
FM_LAMB = 1e-6
FFM_LAMB = 1e-6
LR_LAMB = 1e-5
LSPLM_LAMB = 1e-6



NCF_LAMB = 1e-6
WideAndDeep_LAMB = 1e-5    # AEOLUS  6e-5  Liu 1e-5
DeepAndCross_LAMB = 1e-2   # AEOLUS 1e-3  Liu 1e-2
DeepCrossing_LAMB = 1e-4   # both 1e-4
FNN_LAMB = 5e-5   # both 5e-5
PNN_LAMB = 1e-5   # AEOLUS 1e-4  Liu 1e-5
NFM_LAMB = 2e-5   # AEOLUS 1.2e-4  Liu 2e-5
DeepFM_LAMB = 5e-6   # both 5e-6
AFM_LAMB = 4e-5  # both 4e-5


def getLambda(name, data):
    lamb = 0
    if data == "AEOLUS":
        if name == "NCF":
            lamb = 1e-6
        elif name == "WideAndDeep" or name == "DrugNCF":
            lamb = 6e-5
        elif name == "DeepAndCross":
            lamb = 1e-3
        elif name == "DeepCrossing":
            lamb = 1e-4
        elif name == "FNN":
            lamb = 5e-5
        elif name == "PNN":
            lamb = 1e-4
        elif name == "NFM":
            lamb = 1.2e-4
        elif name == "DeepFM":
            lamb = 5e-6
        elif name == "AFM":
            lamb = 4e-5

    if data == "Liu":
        if name == "NCF":
            lamb = 1e-6
        elif name == "WideAndDeep" or name == "DrugNCF":
            # lamb = 1e-5
            lamb = 1e-5
        elif name == "DeepAndCross":
            lamb = 1e-2
        elif name == "DeepCrossing":
            lamb = 1e-4
        elif name == "FNN":
            lamb = 5e-5
        elif name == "PNN":
            lamb = 1e-5
        elif name == "NFM":
            lamb = 2e-5
        elif name == "DeepFM":
            lamb = 5e-6
        elif name == "AFM":
            lamb = 4e-5

    return lamb
