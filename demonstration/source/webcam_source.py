import multiprocessing
from multiprocessing.context import Process

import cv2

from demonstration.utils import image_utils


class WebcamVideoStream:
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
        stream = cv2.VideoCapture(src)
        while True:
            grabbed, frame = stream.read()
            if not grabbed:
                return
            frame = image_utils.resize_image_if_lager(frame, im_width)
            queue.put(frame)
