SEED = 1024
IN_CHANNEL = 3
OUT_CHANNEL = 3
NGF = 64
NDF = 64
DISCRIMINATOR_LAYER = 3
INPUT_SIZE = 256

# train
EPOCHS = 200
BATCH_SIZE = 4
TRAIN_NUM_WORKERS = 8
TEST_NUM_WORKERS = 8

G_LR = 0.0002
G_BETAS = (0.5, 0.99)
D_LR = 0.0002
D_BETAS = (0.5, 0.99)
LR_DECAY_EPOCHS = 8

PRINT_FREQUENT = 0.01
EVAL_FREQUENT = 0.2

REAL_LABEL = 1
FAKE_LABEL = 0

L1_LOSS_LAMUDA = 100
USING_DROPOUT_DURING_EVAL = False

# data
# DATASET_ROOT = r"/home/MyDisk3/YZJ/dataset/edges2shoes"
# DATASET = "edge2shoes"
DATASET_ROOT = r"/home/MyDisk3/YZJ/dataset/mogaoku/connect"
DATASET = "Mogaoku"

# output
CONSTANT_FEATURE_DIS_LEN = 4
OUTPUT_ROOT = "./output"
OUTPUT_MODEL_KEY = "model"
OUTPUT_LOG_KEY = "log"
OUTPUT_IMAGE_KEY = "images"
