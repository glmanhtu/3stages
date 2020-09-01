# Adapted from https://github.com/timesler/facenet-pytorch

import cv2
import numpy as np
import torch
from facenet_pytorch.models.mtcnn import PNet, RNet, ONet, fixed_image_standardization
from torch import nn

from demonstration.utils import image_utils
from utils import resource_utils
from network.detect_face import detect_face, extract_face


class MTCNN(nn.Module):
    """MTCNN face detection module.

    This class loads pretrained P-, R-, and O-nets and, given raw input images as PIL images,
    returns images cropped to include the face only. Cropped faces can optionally be saved to file
    also.

    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        min_face_size {int} -- Minimum face size to search for. (default: {20})
        thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
        factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
        post_process {bool} -- Whether or not to post process images tensors before returning. (default: {True})
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
            If False, the face with the highest detection probability is returned. (default: {True})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter. If a save_path is specified, the first face is saved to that
            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(
            self, image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            select_largest=True, keep_all=False, device=None, min_keep_prob=0.6,
    ):
        super().__init__()

        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.min_keep_prob = min_keep_prob

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.device = torch.device('cpu')
        self.face_cascade = cv2.CascadeClassifier(
            resource_utils.get_resource('haarcascade/haarcascade_frontalface_alt2.xml'))
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, image, is_brg=False, face_detect_im_width=None, use_haar=False):
        """Run MTCNN face detection on a PIL image. This method performs both detection and
        extraction of faces, returning tensors representing detected faces rather than the bounding
        boxes. To access bounding boxes, see the MTCNN.detect() method below.

        Arguments:
            img {PIL.Image or list} -- A PIL image or a list of PIL images.

        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved face
                image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether or not to return the detection probability.
                (default: {False})

        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra
                dimension (batch) as the first dimension.

        Example:
        >>> from facenet_pytorch import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """
        if is_brg:
            img = image[..., ::-1]   # from BGR to RGB
        else:
            img = image
        image_to_detect_face = img
        if face_detect_im_width is not None:
            # Resize the image smaller for faster face detection
            image_to_detect_face = image_utils.resize_image_if_lager(img, face_detect_im_width)

        # Detect faces
        with torch.no_grad():
            if not use_haar:
                batch_boxes, batch_probs = self.detect(image_to_detect_face)
            else:
                batch_boxes, batch_probs = self.haarcascade_face_detect(image_to_detect_face)

        if face_detect_im_width is not None:
            # Converting the coordinates of boxes back to original one
            for i, bbox in enumerate(batch_boxes):
                if bbox is None:
                    continue
                for j, box in enumerate(bbox):
                    batch_boxes[i][j] = box * img.shape[1] / face_detect_im_width

        # Process all bounding boxes and probabilities
        faces, probs, boxes = [], [], []
        for box_im, prob_im in zip(batch_boxes, batch_probs):
            if box_im is None:
                continue

            for i, box in enumerate(box_im):
                if prob_im[i] < self.min_keep_prob:
                    continue
                try:
                    face, margin_box = extract_face(img, box, self.image_size, self.margin)
                    if self.post_process:
                        face = fixed_image_standardization(face)
                    faces.append(face)
                    boxes.append(margin_box)
                    probs.append(prob_im[i])
                except Exception:
                    # Ignore exception, for now
                    continue

        return faces, boxes, probs

    def haarcascade_face_detect(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(image, 1.3, 5)
        result = []
        prob = []
        for (x, y, w, h) in faces:
            result.append([x, y, x + w, y + h])
            prob.append(0.99)
        return [np.array(result)], [prob]


    def detect(self, img, landmarks=False):
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.

        Arguments:
            img {PIL.Image or list} -- A PIL image or a list of PIL images.

        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})

        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.

        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """

        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes)
        probs = np.array(probs)
        points = np.array(points)

        if landmarks:
            return boxes, probs, points

        return boxes, probs
