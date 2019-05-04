import skimage as ski
import numpy as np


class ImageProcessor:
    """ImageProcessor is a class that implements the image processing
    functions from scikit-image. It has no input parameters for the
    construction and does not have any attributes.
    """

    def __init__(self):
        pass

    def adaptiveEqualization(self, img):
        """Applies histogram equalization to input image
        :param img: Image be processed
        :return: hist_eql_img: img after histogram equalization
        """
        hist_eql_img = np.array(np.zeros(img.shape))
        if img.ndim >= 3:
            for channel in range(img.shape[2]):
                ch_hist_eql = ski.exposure.equalize_hist(img[:, :, channel])

                hist_eql_img[:, :, channel] = ski.exposure.rescale_intensity(
                    ch_hist_eql, out_range=(0, 255))
        else:
            hist_eql_img = ski.exposure.equalize_hist(img)
            hist_eql_img = ski.exposure.rescale_intensity(hist_eql_img,
                                                          out_range=(0, 255))

        hist_eql_img = hist_eql_img.astype(np.uint8)

        return hist_eql_img

    def contrastStretch(self, img):
        """Applies contrast stretching to input image
        :param img: Image to be processed
        :return: cont_stretch_img: img after contrast stretching
        """
        cont_stretch_img = ski.exposure.rescale_intensity(img)

        return cont_stretch_img

    def logCompression(self, img):
        """Applies logarithmic compression to input image
        :param img: Image to be processed
        :return: log_comp_img: img after logarithmic compression
        """
        log_comp_img = ski.exposure.adjust_log(img)
        return log_comp_img

    def reverseVideo(self, img):
        """Inverts the colors in an image

        :param img: Image to be processed
        :return: inverted_img: Image with inverted colors
        """
        inverted_img = np.invert(img)
        return inverted_img

    def isGrayscale(self, img):
        """Checks to see if an image is grayscale
        isGrayscale determines if an images is grayscale by assuming a
        grayscale image will have one of the following properties
        1. Only have two dimensions
        2. If it has 3D (indicating RGB pixel color values), R=B=G for all
        pixels.
        :param img: Input image
        :return: is_grayscale: Indicates whether the input image is grayscale
        """

        if img.ndim == 2:
            is_grayscale = True
            return is_grayscale
        img_dimm = img.shape

        for x in range(0, img_dimm[0]):
            for y in range(0, img_dimm[1]):
                if img[x, y, 0] == img[x, y, 1] == img[x, y, 2]:
                    continue
                else:
                    is_grayscale = False
                    return is_grayscale

        # It makes it through the loop without finding a place where pixels
        # are not equal (causing it to return False), then assume that it is
        #  a grayscale image.
        is_grayscale = True
        return is_grayscale

    def histogram(self, img):
        """Generates a list of histograms with intensity values for each
        channel in the image.

        Each item in the list consists of a 2D numpy array, in which the
        first dimension is the histogram itself, and the second dimension
        is the bin values. A histogram item from this list could be plotted
        as plt.plot(histogram_item[1], histogram_item[0])

        :param img: input image
        :return: hist (list): List of histograms for each color channel
        """
        hist = []
        if self.isGrayscale(img):
            [bins, hist_vals] = ski.exposure.histogram(img)
            bins = bins.tolist()
            hist_vals = hist_vals.tolist()
            hist_temp = (bins, hist_vals)

            hist.append(hist_temp)

            return hist
        else:
            dimm = img.shape
            hist = []
            for d in range(0, dimm[2]):
                [bins, hist_vals] = ski.exposure.histogram(img[:, :, d])
                bins = bins.tolist()
                hist_vals = hist_vals.tolist()
                hist_temp = (bins, hist_vals)

                hist.append(hist_temp)
            return hist
