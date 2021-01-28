import cv2

from demonstration.utils import image_utils


class WebcamVideoStream:
    def __init__(self, src=0, img_max_width=600):
        # initialize the variable used to indicate if the thread should be stopped
        self.src = src
        self.im_width = img_max_width

    def read(self):
        # initialize the video camera stream
        stream = cv2.VideoCapture(self.src)
        while True:
            grabbed, frame = stream.read()
            if not grabbed:
                return
            frame = image_utils.resize_image_if_lager(frame, self.im_width)
            yield frame
