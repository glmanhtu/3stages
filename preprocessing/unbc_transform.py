import cv2
import numpy as np
import torch
from PIL import Image


def get_landmark_most_points(landmarks):
    min_x, min_y, max_x, max_y = 9999, 9999, 0, 0
    for landmark in landmarks:
        if min_x > landmark[0]:
            min_x = landmark[0]
        if max_x < landmark[0]:
            max_x = landmark[0]
        if min_y > landmark[1]:
            min_y = landmark[1]
        if max_y < landmark[1]:
            max_y = landmark[1]
    return min_x, min_y, max_x, max_y


class ToPIL(object):

    def __call__(self, sample):
        return Image.fromarray(sample)


class ImageFeaturesExtraction(object):

    def __init__(self, dl_model) -> None:
        super().__init__()
        self.dl_model = dl_model

    def __call__(self, sample):
        with torch.set_grad_enabled(False):
            # CNN expect 4-dimension input data (including batch size)
            sample = sample[None]
            return self.dl_model(sample)[0]


class Alignment:
    """
    Simple face alignment algorithm, which based on the eyes of the face to rotate both landmarks and image
    """
    def __init__(self, scale=1):

        self.left_eyes = (36, 42)
        self.right_eyes = (42, 48)
        self.nose_center = (30, 31)
        self.scale = scale

    @staticmethod
    def _angle_between_2_pt(p1, p2):
        """
        to calculate the angle rad by two points
        """
        x1, y1 = p1
        x2, y2 = p2
        tan_angle = (y2 - y1) / (x2 - x1)
        return np.degrees(np.arctan(tan_angle))

    def _get_rotation_matrix(self, left_eye_pt, right_eye_pt, center_point, scale):
        """
        to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
        """
        eye_angle = self._angle_between_2_pt(left_eye_pt, right_eye_pt)
        rotate = cv2.getRotationMatrix2D(center_point, eye_angle, scale)
        return rotate

    @staticmethod
    def _transform_landmarks(landmarks, rotate):
        """
        Rotate the landmarks to match with the new aligned image
        """
        ones = np.ones(shape=(len(landmarks), 1))
        points_ones = np.hstack([landmarks, ones])
        return rotate.dot(points_ones.T).T

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]

        left_eye_pts = landmarks[self.left_eyes[0]:self.left_eyes[1], :]
        right_eye_pts = landmarks[self.right_eyes[0]:self.right_eyes[1], :]

        # compute the center of mass (centroid) for each eye
        leye_center = left_eye_pts.mean(axis=0).astype("int")
        reye_center = right_eye_pts.mean(axis=0).astype("int")

        min_x, min_y, max_x, max_y = get_landmark_most_points(landmarks)

        center_point = int(min_x + (max_x - min_x) / 2), int(min_y + (max_y - min_y) / 2)

        trotate = self._get_rotation_matrix(tuple(leye_center), tuple(reye_center), tuple(center_point), self.scale)

        # We increase the warped width and height for preventing losing pixels when doing the rotation
        warped = cv2.warpAffine(image, trotate, (int(w * 1.2), int(h * 1.2)))
        landmarks = self._transform_landmarks(landmarks, trotate)

        sample['landmarks'] = landmarks
        sample['image'] = warped

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        sample['image'] = image
        sample['landmarks'] = landmarks

        return sample
