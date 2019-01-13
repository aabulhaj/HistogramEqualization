import numpy as np

import utils


def _histogram_equalize_image(image, hist_orig):
    """Equalizes a given image. The image must have one value per pixel, if you want to equalize an RGB image then
        convert it to YIQ and pass an image with Y values only
    :param image: The image we want to equalize (Normalized).
    :param hist_orig: Histogram for the given image.
    :return: A histogram equalized image (Normalized).
    """
    cum_hist = np.cumsum(hist_orig)
    cum_hist = (cum_hist * 255) / cum_hist[-1]

    image = np.interp(image, np.linspace(0, 1, 256), np.round(cum_hist))

    return utils.normalize_image(image)


def _histogram_equalize_rgb(im_orig):
    """Performs histogram equalization of a given RGB image.
    :param im_orig: The input RGB float64 image with values in [0, 1].
    :return: a list [im_eq, hist_orig, hist_eq] where
        im_eq - is the equalized image. RGB float64 image with values in [0, 1].
        hist_orig - is a 256 bin histogram of the original image (array with shape (256, ) ).
        hist_eq - is a 256 bin histogram of the equalized image (array with shape (256, ) ).
    """
    imYIQ = utils.rgb2yiq(im_orig)

    hist_orig = utils.get_histogram(imYIQ[:, :, 0])

    imYIQ[:, :, 0] = _histogram_equalize_image(imYIQ[:, :, 0], hist_orig)

    hist_eq = utils.get_histogram(imYIQ[:, :, 0])

    im_eq = utils.yiq2rgb(imYIQ)

    return [im_eq, hist_orig, hist_eq]


def _histogram_equalize_grayscale(im_orig):
    """Performs histogram equalization of a given grayscale image.
    :param im_orig: The input grayscale float64 image with values in [0, 1].
    :return: a list [im_eq, hist_orig, hist_eq] where
        im_eq - is the equalized image. grayscale float64 image with values in [0, 1].
        hist_orig - is a 256 bin histogram of the original image (array with shape (256, ) ).
        hist_eq - is a 256 bin histogram of the equalized image (array with shape (256, ) ).
    """
    image = im_orig.copy()

    hist_orig = utils.get_histogram(image)

    im_eq = _histogram_equalize_image(image, hist_orig)

    hist_eq = utils.get_histogram(im_eq)

    return [im_eq, hist_orig, hist_eq]


def histogram_equalize(im_orig):
    """Performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: The input grayscale or RGB float64 image with values in [0, 1].
    :return: a list [im_eq, hist_orig, hist_eq] where
        im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
        hist_orig - is a 256 bin histogram of the original image (array with shape (256, ) ).
        hist_eq - is a 256 bin histogram of the equalized image (array with shape (256, ) ).
    """
    if im_orig.ndim == 3:
        return _histogram_equalize_rgb(im_orig)
    return _histogram_equalize_grayscale(im_orig)
