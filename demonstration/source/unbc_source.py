import multiprocessing
from multiprocessing.context import Process

import cv2
from torchvision import transforms

from databases.unbc.unbc_mcmaster_cnn import UNBCCNNDataset
from demonstration.utils import image_utils


class UNBCSource:
    def __init__(self, src=0, img_max_width=600, max_size=100):
        # initialize the variable used to indicate if the thread should be stopped
        self.queue = multiprocessing.Queue(maxsize=max_size)
        self.src = src
        self.img_width = img_max_width

    def start(self):
        # start reading frames from the video stream
        reader_p = Process(target=self.update, args=(self.queue, self.src, self.img_width))
        reader_p.daemon = True
        reader_p.start()
        return self

    def update(self, queue, src, im_width):
        # initialize the video camera stream
        dataset = UNBCCNNDataset(excluded_subjects=[], init_transform=transforms.Compose([]), apply_balancing=False)
        for sample in dataset:
            frame = sample['image']
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = image_utils.resize_image_if_lager(frame, im_width)
            queue.put(frame)
