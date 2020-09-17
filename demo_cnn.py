import queue

import cv2
import torch
from matplotlib.colors import LinearSegmentedColormap

from demonstration.analysing.pain_level_analysing import FacialExtractor, PainAnalysingEstimator
# from demonstration.source.unbc_source import UNBCSource
from demonstration.source.webcam_source import WebcamVideoStream
from demonstration.utils.fps import FPS

torch.backends.cudnn.benchmark = True

cmap = LinearSegmentedColormap.from_list('', ["green", "yellow", "red"])
stream_source = WebcamVideoStream(src=0, img_max_width=600)
# stream_source = UNBCSource(img_max_width=600)
fps = FPS()
analysing_queue = queue.Queue()
visualising_queue = queue.Queue()

fps.start()
stream_source.start()
FacialExtractor(stream_source.queue, analysing_queue).start()
PainAnalysingEstimator(analysing_queue, visualising_queue).start()

while True:
    image, bboxes, pain_levels = visualising_queue.get(block=True)
    for idx, bbox in enumerate(bboxes):
        if bbox is None:
            continue
        x, y, w, h = tuple(bbox)
        pain_level = pain_levels[idx][0].item()
        if pain_level < 0:
            pain_level = 0
        color = cmap(pain_level / 16)
        color = tuple(255 * x for x in color[:3])
        color = color[2], color[1], color[0]
        cv2.rectangle(image, (x, y), (w, h), color, 2)
        cv2.putText(image, str(round(pain_level, 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    fps.update()
    cv2.putText(image, 'FPS: %d' % fps.fps(), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.imshow('frame', image)
    cv2.waitKey(1)