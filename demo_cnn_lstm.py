import argparse
import os
import queue
import sys

import cv2
import torch
from matplotlib.colors import LinearSegmentedColormap

from demonstration.analysing.pain_level_analysing import FacialExtractor, PainAnalysingEstimator, LSTMPainEstimator
from demonstration.source.unbc_source import UNBCSource
from demonstration.source.webcam_source import WebcamVideoStream
from demonstration.utils.fps import FPS

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pain level intensity estimation using Stage 1+2')
    parser.add_argument('--webcam', dest='webcam', action='store_true', help='Using webcam as input')
    parser.add_argument('--skip-frames', dest='skip', action='store', help='Number of skipping frames', type=int, default=2)
    parser.add_argument('--video-file', dest='video_path', action='store', help='Path to video file', required=False)

    args = parser.parse_args()
    cmap = LinearSegmentedColormap.from_list('', ["green", "yellow", "red"])

    if args.webcam:
        stream_source = WebcamVideoStream(src=0, img_max_width=600)
    elif args.video_path is not None:
        assert os.path.isfile(args.video_path)
        stream_source = WebcamVideoStream(src=args.video_path, img_max_width=600)
    else:
        stream_source = UNBCSource(img_max_width=600)

    fps = FPS()

    fps.start()
    facial_extractor = FacialExtractor()
    pain_estimator = PainAnalysingEstimator(classify=False)
    lstm_estimator = LSTMPainEstimator()

    prev = ()
    for idx, image in enumerate(stream_source.read()):
        if idx % (args.skip + 1) != 0 and prev is not None:
            pain_levels = prev
        else:
            image, faces, bboxes = facial_extractor.extract_face(image)
            features = pain_estimator.estimate(faces)
            pain_levels = lstm_estimator.estimate(bboxes, features)
            prev = pain_levels

        if len(bboxes) > 0 and pain_levels is not None:
            bbox = bboxes[0]
            x, y, w, h = tuple(bbox)
            pain_level = pain_levels
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


