import torch

from network import rnn
from network.inception_resnet import get_pretrained_facenet
from network.mtcnn import MTCNN
from utils import resource_utils, cnn_utils, rnn_utils
from utils.constants import device


class FacialExtractor:
    def __init__(self):
        super().__init__()
        self.__mtcnn = MTCNN(margin=2, post_process=True,
                             min_keep_prob=0.8, select_largest=False, device=device).eval()

    def extract_face(self, original_image):
        faces, bounding_boxes, _ = self.__mtcnn(original_image, is_brg=True, face_detect_im_width=120)
        return original_image, faces, bounding_boxes


class PainAnalysingEstimator:
    def __init__(self, classify=True):
        super().__init__()
        self.__model = get_pretrained_facenet(classify=classify, pretrained=None).eval()
        checkpoint = resource_utils.get_checkpoint_file_path(name='107-hs107_cnn.ckpt')
        load_status, _, _ = cnn_utils.load_pretrained_model(checkpoint, self.__model)
        if not load_status:
            raise Exception('Unable to load model ' + checkpoint)

    def estimate(self, faces):
        if len(faces) > 0:
            faces = torch.stack(faces)
            faces = faces.to(device, dtype=torch.float)
            pain_levels_detected = self.__model(faces)
            return pain_levels_detected
        return None


class LSTMPainEstimator:
    def __init__(self, sequence_length=16):
        super().__init__()
        self.__sequence_length = sequence_length
        n_layers, n_neurons, dropout = 2, 744, 0.5
        checkpoint = resource_utils.get_checkpoint_file_path(name='107-hs107_rnn.ckpt')
        self.__model = rnn.LSTM(1792, n_neurons, num_layers=n_layers, num_classes=1, dropout=dropout).to(device)
        load_status, _, _ = rnn_utils.load_pretrained_model(checkpoint, self.__model)
        if not load_status:
            raise Exception('Unable to load model ' + checkpoint)

        self.__sequence_embedded = []

    def estimate(self, bboxes, embedded):
        if len(bboxes) > 0:
            embedded = embedded[0]          # For now, we support only one face in the video
            if len(self.__sequence_embedded) < self.__sequence_length:
                self.__sequence_embedded.append(embedded)
            else:
                del self.__sequence_embedded[0]
                self.__sequence_embedded.append(embedded)
            if len(self.__sequence_embedded) == self.__sequence_length:
                sequence_embedded = torch.stack(self.__sequence_embedded)
                pain_level = self.__model(sequence_embedded[None]).item()
                return pain_level
        return None

