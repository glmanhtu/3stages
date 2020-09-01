from random import randrange

import cv2
import numpy as np
import torch

from utils.constants import UNBC_BASE_GPA_LANDMARKS_PATH
from preprocessing import gpa

from preprocessing.unbc_transform import get_landmark_most_points


def transform_landmarks(landmarks, rotate):
    """
    Rotate the landmarks to match with the new aligned image
    """
    ones = np.ones(shape=(len(landmarks), 1))
    points_ones = np.hstack([landmarks, ones])
    return rotate.dot(points_ones.T).T


class FixedImageStandardization(object):

    def __call__(self, sample):
        processed_tensor = (sample * 255 - 127.5) / 128.0
        return processed_tensor


class NPFixedImageStandardization(object):

    def __call__(self, sample):
        return (sample - 127.5) / 128.0


class PerPixelMeanSubtraction(object):

    def __init__(self, image_mean_file):
        self.image_mean = torch.load(image_mean_file)

    def __call__(self, sample):
        return sample - self.image_mean


class Resize:
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

        if new_w > image.shape[1]:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)

        landmarks *= [new_w / w, new_h / h]

        sample['image'] = image
        sample['landmarks'] = landmarks

        return sample


class CentralCropV1(object):
    """Crop the image in a sample.
        Make sure the head is in the central of image
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size, percent=0.15, close_top=0.90):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.percent = percent
        self.close_top = close_top

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        min_x, min_y, max_x, max_y = get_landmark_most_points(landmarks)

        # if max_x > w:
        #     different = max_x - w
        #     min_x -= different
        # if max_y > h:
        #     different = max_y - h
        #     min_y -= different

        gap = (max_y - min_y) * self.percent
        distance = int(max_y - min_y + gap * 2)
        if distance > min(w, h):
            distance = min(w, h)

        x = int(min_x - gap)
        if x < 0:
            x = 0
        y = int((min_y - gap) * self.close_top)
        if y < 0:
            y = 0
        # if y + distance < max_y:
        #     y = int(max_y) - distance

        image = image[y: y + distance, x: x + distance].copy()

        landmarks = landmarks - np.array([x, y])

        if new_w > image.shape[1]:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_CUBIC)
        else:
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)

        landmarks *= [new_w / distance, new_h / distance]

        sample['image'] = image
        sample['landmarks'] = landmarks

        return sample


class CentralCropV3:

    def __init__(self, output_size, anchor=None):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        if anchor is None:
            self.anchor = np.array([[127.6475, 227.8161], [79.1608, 87.0376], [176.8392, 87.0376]], np.float32)
        else:
            self.anchor = anchor

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        idx = [8, 36, 45]  # """Anchor points"""
        pts = np.float32(landmarks[idx, :])
        M = cv2.getAffineTransform(pts, self.anchor)
        dst = cv2.warpAffine(image, M, self.output_size)
        sample['image'] = dst
        sample['landmarks'] = transform_landmarks(landmarks, M)
        return sample


class CentralCrop(object):
    """Crop the image in a sample.
        Make sure the head is in the central of image
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size, gap_percent=0.05, showing_top=0.65):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.percent = gap_percent
        self.showing_top = showing_top

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        min_x, min_y, max_x, max_y = get_landmark_most_points(landmarks)

        # if max_x > w:
        #     different = max_x - w
        #     min_x -= different
        # if max_y > h:
        #     different = max_y - h
        #     min_y -= different

        max_face = max(max_y - min_y, max_x - min_x)
        gap = min((max_y - min_y), (max_x - min_x)) * self.percent
        distance = new_w
        if distance > min(w, h):
            distance = min(w, h)

        x = int((min_x + max_x) / 2 - distance / 2)
        if x < 0:
            x = 0
        if x + distance < max_x:
            x = int(max_x - distance)
        if x + distance > w:
            x = w - distance
        y = int((min_y - gap) * self.showing_top)
        if y < 0:
            y = 0
        if y + distance < max_y:
            y = int(max_y - distance)
        if y + distance > h:
            y = h - distance

        image = image[y: y + distance, x: x + distance].copy()

        assert image.shape[0] == image.shape[1]

        landmarks = landmarks - np.array([x, y])

        if new_w != image.shape[1]:
            # scale up
            image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LANCZOS4)
        # else:
        #     # Shrinking
        #     image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_CUBIC)

        landmarks *= np.array([new_w, new_h]) / float(distance)

        sample['image'] = image
        sample['landmarks'] = landmarks

        # image_debug_utils.show_landmarks(sample['image'], landmarks)
        return sample


class AUCentralLocalisation:

    def __init__(self, central_idx):
        self.central_idx = central_idx

    def __call__(self, sample):
        landmarks = sample['landmarks']
        # image_debug_utils.show_landmarks(sample['image'], landmarks)
        for au in self.central_idx:
            key = "coord_au_%d" % au
            sample[key] = []
            centre_points = self.central_idx[au]
            for centre_point in centre_points:
                if centre_point is None:
                    continue
                sample[key].append(tuple(np.array([landmarks[i] for i in centre_point]).mean(axis=0)))
        # image_debug_utils.show_au_centrals(sample)
        return sample


class RandomLandmarkRotation:
    def __init__(self, mean_shape=None, degree=20, prob=0.5):
        if mean_shape is None:
            self.mean_shape = np.load(UNBC_BASE_GPA_LANDMARKS_PATH)
        else:
            self.mean_shape = mean_shape
        self.degree = degree
        self.prob = prob

    def __call__(self, sample):
        should_rotate = np.random.choice(np.arange(0, 2), p=[1 - self.prob, self.prob])
        mean_shape = self.mean_shape
        if should_rotate:
            angle = randrange(-self.degree, self.degree)
            rot_mat = cv2.getRotationMatrix2D(tuple(self.mean_shape.mean(axis=0)), angle, 1.0)
            mean_shape = transform_landmarks(self.mean_shape, rot_mat)
        return GPAAlignment(mean_shape)(sample)


class GPAAlignment:
    """
    Employ Generalised Procrusters analysis to align both the image and the landmarks, based on the reference
    mean shape. See preprocessing/common/gpa.py for how to generate the mean shape
    """

    def __init__(self, mean_shape=None):
        if mean_shape is None:
            self.mean_shape = np.load(UNBC_BASE_GPA_LANDMARKS_PATH)
        else:
            self.mean_shape = mean_shape

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks'
        ]
        _, aligned_landmarks, tform = gpa.procrustes(self.mean_shape, landmarks)

        rotate = np.identity(3)
        rotate[:2, :2] = tform['rotation'].transpose()
        # to_centroid = np.identity(3)
        # to_centroid[:2, -1] = landmarks.mean(axis=0)
        # to_orig = np.identity(3)
        # to_orig[:2, -1] = -landmarks.mean(axis=0)
        translate = np.identity(3)
        translate[:2, -1] = tform['translation']

        scale = np.identity(3)
        scale[0, 0] = tform['scale']
        scale[1, 1] = tform['scale']

        m_translate = np.dot(translate, scale)
        m = np.dot(m_translate, rotate)

        rows, cols, _ = image.shape
        im2 = cv2.warpAffine(image, m[:-1, :], (cols, rows), flags=cv2.INTER_CUBIC)

        sample['image'] = im2
        sample['landmarks'] = aligned_landmarks

        return sample


class HeatMapGenerator:

    def __init__(self, aus, input_shape, output_shape, sigma=2., combine_same_aus=False):
        self.aus = aus
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.sigma = sigma
        self.combine_same_aus = combine_same_aus

    def compute_percent_hiding(self, landmarks, point):
        # For each AU points, it should have one left, and one right point. But when the person turn their head,
        # One should be hided by some %
        pass

    def render_gaussian_heatmap(self, coord, sigma):
        in_shape = self.input_shape
        out_shape = self.output_shape
        x = [i for i in range(out_shape[1])]
        y = [i for i in range(out_shape[0])]
        xx, yy = torch.meshgrid([torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)])
        xx = xx.T.reshape((1, *out_shape, 1))
        yy = yy.T.reshape((1, *out_shape, 1))
        x = torch.floor(coord[:, 0].reshape([1, 1, len(coord)]) / in_shape[1] * out_shape[1] + 0.5)
        y = torch.floor(coord[:, 1].reshape([1, 1, len(coord)]) / in_shape[0] * out_shape[0] + 0.5)
        heatmap = torch.exp(-(((xx - x) / float(sigma)) ** 2.) / 2. - (((yy - y) / float(sigma)) ** 2.) / 2.)
        return heatmap.squeeze().permute((2, 0, 1))

    def __call__(self, sample):
        intensities = []
        coord = []
        aus_idx = []

        for au in self.aus:
            au_coord = sample['coord_' + au]
            tmp_idx = []
            for x in au_coord:
                intensities.append(sample[au])
                tmp_idx.append(len(intensities) - 1)
            aus_idx.append(tmp_idx)
            coord += au_coord
        with torch.set_grad_enabled(False):
            intensities = torch.tensor(intensities, dtype=torch.float32)
            intensity_map = intensities.reshape((len(intensities), 1, 1))
            coord = torch.tensor(coord, dtype=torch.float32)
            gt_heatmap = self.render_gaussian_heatmap(coord, self.sigma)
            gt_heatmap = gt_heatmap * intensity_map
        if self.combine_same_aus:
            heatmaps = []
            for idxs in aus_idx:
                heatmap = torch.zeros(gt_heatmap[idxs[0]].shape)
                for idx in idxs:
                    intensity_max = gt_heatmap[idx].max().item()
                    heatmap += gt_heatmap[idx]
                    heatmap[heatmap > intensity_max] = intensity_max
                heatmaps.append(heatmap)
            gt_heatmap = torch.stack(heatmaps)
        # image_debug_utils.show_heatmaps_aus(sample, gt_heatmap / 5)
        sample['heatmap'] = gt_heatmap
        return sample


