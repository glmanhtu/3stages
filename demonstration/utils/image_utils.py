import cv2


def resize_image_if_lager(image, im_width):
    h, w = image.shape[:2]
    ratio = im_width / w
    if ratio < 1:
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
    return image
