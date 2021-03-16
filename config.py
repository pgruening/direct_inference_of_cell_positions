from os.path import join

PERC_SPLIT = .9
SPLIT_SEED = 0
AUG_TYPE = 'minimal'

EXP_FOLDER = './experiments'
LEARNING_RATE = 0.01
OPTIMIZER = 'Adam'
BATCH_SIZE = 4
CROP_SIZE = 512
LR_STEPS = 2
NUM_WORKERS = 0  # use 0 when you want to debug
EPOCHS = 100
MOMENTUM = .9
SAVE_INTERVALL = 10
SEED = 0
INPUT_DIM = 3
DEVICE = None
DEF_FOLDER = 'default'
MODEL_NAME = 'model.pt'
MODEL_TYPE = 'smp_resnet18'
NUM_CLASSES = 2
PERC_SPLIT = .9
SPLIT_SEED = 0
AUG_TYPE = 'minimal'
ES_METRIC = 'val_dice'
WEIGHT_DECAY = 0.0001

# evaluation
MATCH_THRES = .3
OPENING_KERNEL = 2

# dataset related
DATASET_LENGTH = 2000

# dataset path for simulated training data
BASE_FOLDER = '/data'
#BASE_FOLDER = '/data_ssd0/gruening/killa_seg/clean_simulation_data/data'
SIM_IMAGES = join(BASE_FOLDER, 'train', 'holograms')
SIM_LABELS = join(BASE_FOLDER, 'train', 'cell_labels')

SIM_EVAL_BASE = join(BASE_FOLDER, 'eval')

# original data
LABELS = None
LABEL_THRES = 90
