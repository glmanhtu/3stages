import multiprocessing
import os

import torch

from utils import resource_utils

N_CORES = multiprocessing.cpu_count()

UNBC_DATASET_PATH = resource_utils.get_resource(path='unbc/dataset')
assert os.path.isdir(UNBC_DATASET_PATH)
UNBC_BASE_GPA_LANDMARKS_PATH = resource_utils.get_resource(path='unbc/unbc_gpa_base_landmarks.npy')

GPU_ID = 0 if 'PAIN_GPU_ID' not in os.environ else os.environ['PAIN_GPU_ID']
DEVICE_ID = "cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu"

# Detect if we have a GPU available
device = torch.device(DEVICE_ID)

AU_CENTRAL_POINTS = {
    1: ((21, 21), (22, 22)),
    2: ((18, 36), (25, 45)),
    4: ((21, 22), None),
    5: ((18, 36), (25, 45)),
    6: ((3, 3, 17, 17, 51), (26, 26, 13, 13, 51)),
    7: ((37, 37), (44, 44)),
    9: ((31, 31), (35, 35)),
    10: ((49, 49), (53, 53)),
    12: ((48, 48), (54, 54)),
    14: ((3, 3, 4, 48, 48), (12, 13, 13, 54, 54)),
    15: ((48, 48), (54, 54)),
    17: ((8, 57), (57, 57)),
    20: ((6, 48), (10, 54)),
    25: ((51, 51), (57, 57)),
    26: ((51, 51), (57, 57)),
    43: ((36, 37, 38, 39, 40, 41), (42, 43, 44, 45, 46, 47))
}
