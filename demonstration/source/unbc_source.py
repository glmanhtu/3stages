import cv2
from torchvision import transforms

from databases.unbc.unbc_mcmaster_cnn import UNBCCNNDataset
from utils.constants import UNBC_DATASET_PATH


class UNBCSource:
    def __init__(self, src=UNBC_DATASET_PATH, img_max_width=600):
        # initialize the variable used to indicate if the thread should be stopped
        self.src = src
        self.img_width = img_max_width

    def read(self):
        # initialize the video camera stream
        dataset = UNBCCNNDataset(dataset_path=self.src, init_transform=transforms.Compose([]), apply_balancing=False)
        for sample in dataset:
            frame = sample['image']
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame
        return
