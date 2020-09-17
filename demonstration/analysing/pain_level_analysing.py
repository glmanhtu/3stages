import threading

import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

from network import rnn
from network.mtcnn import MTCNN
from utils import resource_utils, cnn_utils, rnn_utils
from utils.constants import device


class FacialExtractor(threading.Thread):
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.__input_queue = input_queue
        self.__analysing_queue = output_queue
        self.__mtcnn = MTCNN(margin=20, post_process=True, min_keep_prob=0.8, select_largest=False)

    def run(self) -> None:
        while True:
            original_image = self.__input_queue.get()
            faces, bounding_boxes, _ = self.__mtcnn(original_image, is_brg=True, face_detect_im_width=120)
            self.__analysing_queue.put((original_image, faces, bounding_boxes))


class PainAnalysingEstimator(threading.Thread):
    def __init__(self, input_queue, output_queue, classify=True):
        super().__init__()
        self.__input_queue = input_queue
        self.__output_queue = output_queue
        self.__model = InceptionResnetV1(pretrained='vggface2', classify=classify, device=device, num_classes=1)
        checkpoint = resource_utils.get_checkpoint_file_path(name='108-th108_cnn.ckpt')
        load_status, _, _ = cnn_utils.load_pretrained_model(checkpoint, self.__model)
        if not load_status:
            raise Exception('Unable to load model ' + checkpoint)

    def run(self) -> None:
        while True:
            original_image, faces, bounding_boxes = self.__input_queue.get()
            pain_levels_detected = []
            if len(faces) > 0:
                faces = torch.stack(faces)
                faces = faces.to(device, dtype=torch.float)
                with torch.set_grad_enabled(False):
                    pain_levels_detected = self.__model(faces)
            self.__output_queue.put((original_image, bounding_boxes, pain_levels_detected))


class LSTMPainEstimator(threading.Thread):
    def __init__(self, input_queue, output_queue, sequence_length=16):
        super().__init__()
        self.__input_queue = input_queue
        self.__output_queue = output_queue
        self.__sequence_length = sequence_length
        n_layers, n_neurons, dropout = 2, 744, 0.5
        checkpoint = resource_utils.get_checkpoint_file_path(name='095-tv095_lstm.ckpt')
        self.__model = rnn.LSTM(1792, n_neurons, num_layers=n_layers, num_classes=1, dropout=dropout).to(device)
        load_status, _ = rnn_utils.load_pretrained_model(checkpoint, self.__model)
        if not load_status:
            raise Exception('Unable to load model ' + checkpoint)

        self.__sequence_embedded = []

    def run(self) -> None:
        while True:
            pain_level = 0
            image, bboxes, embedded = self.__input_queue.get(block=True)
            if len(bboxes) > 0:
                embedded = embedded[0]          # For now, we support only one face in the video
                if len(self.__sequence_embedded) < self.__sequence_length:
                    self.__sequence_embedded.append(embedded)
                else:
                    del self.__sequence_embedded[0]
                    self.__sequence_embedded.append(embedded)
                if len(self.__sequence_embedded) == self.__sequence_length:
                    sequence_embedded = torch.stack(self.__sequence_embedded)
                    with torch.set_grad_enabled(False):
                        pain_level = self.__model(sequence_embedded[None]).item()
            self.__output_queue.put((image, bboxes, pain_level))
