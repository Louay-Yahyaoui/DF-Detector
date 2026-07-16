from pathlib import Path
import torch

# Default paths.
PROCESSED_DATA_DIR = Path("data")
PLOT_DIR = Path(".out")
MODEL_CKPT = None

# We chose these 4 deepfake creation methods to train on.
DEEPFAKE_METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap"]

# Default values.
N_DATA = 1000 # Number of examples per method.
N_TEST = 1000 # Number of test videos per class.
IMG_DIMS = (224, 224)
SCALE = 1.3
TOTAL_FRAMES = 32
DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training hyperparams
BATCH_SIZE = 32
VAL_SPLIT = 0.3
EPOCHS = 20
LR = 3e-4
MIN_LR = 1e-5