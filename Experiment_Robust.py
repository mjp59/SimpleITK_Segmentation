import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')
import os
from scipy.io import loadmat
import SimpleITK as sitk
from ImageProcessor import ImageProcessor
from matplotlib.pyplot import imread
import logging

path = 'data_ForXT/'
csvname = 'Practice_Images.csv'


def load_image(file, ax=2):
    """

    :param file: name of file that needs to be read in from mat file
    :param ax: direction along image that you take the mean of rows against
    :return: image in numpy array along with height, weight, and pixel spatial spacing
    """
    img = loadmat(path+file)['bimg']
    bax = loadmat(path + file)['bax']  # bax 416
    blat = loadmat(path + file)['blat']  # blat 796
    if len(img.shape) == 3:
        img = np.mean(img, axis=ax).astype(np.uint8)
    h, w = img.shape
    spacing_ax = np.diff(bax).mean()
    spacing_lat = np.diff(blat).mean()
    return img, h, w, spacing_ax, spacing_lat


def load_image_png(file):
    """

    :param file: name of file that needs to be read in from png file
    :return: image in numpy array
    """
    img = imread(path + file)
    pro = ImageProcessor()
    img_con = pro.adaptiveEqualization(img)
    return img_con


def write_itk(img, sax, slat):
    """

    :param img: image in numpy array form
    :param sax: spacing in axial dimension
    :param slat: spacing in lateral dimension
    :return: image with pixel depth of 4 in itk image type. Can be used with simple itk
    """
    pro = ImageProcessor()
    img_con = pro.adaptiveEqualization(img)
    img_1 = sitk.GetImageFromArray(
        np.repeat(img_con[np.newaxis, :, :], 4, axis=0))
    img_1.SetSpacing([slat, sax, 1])
    return img_1


def write_itk_png(img):
    """

    :param img: image in numpy array form
    :return: image with pixel depth of 4 in itk image type. Can be used with simple itk
    """
    pro = ImageProcessor()
    img_con = pro.adaptiveEqualization(img)
    img_1 = sitk.GetImageFromArray(
        np.repeat(img_con[np.newaxis, :, :], 4, axis=0))
    # img_1.SetSpacing([slat, sax, 1])
    return img_1


def find_seed(img, w, h, num_seeds):
    """

    :param img: image in numpy array
    :param w: pixel width of the image
    :param h: pixel height of the image
    :param num_seeds: number of desired seeds in the image
    :return:
    """
    summed_rows = img.mean(axis=1)
    cutoff_top = int(.05 * h)
    cutoff_bot = int(.5 * h)
    index = np.argmax(summed_rows[cutoff_top:cutoff_bot]) + cutoff_top
    cutoff_close = int(.43 * w)
    spacing_mod = 2 * cutoff_close
    spacing = int((w-spacing_mod)/num_seeds)
    seed_list = []
    for i in range(num_seeds):
        x = (i * spacing) + cutoff_close
        seed = (x, int(index), 2)
        seed_list.append(seed)
    return seed_list


def get_contrast(img, w, h, seed):
    """

    :param img: simple itk image object. Pixel value of the image is access through the object
    :param w: pixel width of the image
    :param h: pixel height of the image
    :param seed: Coordinates for one seed in the seed list
    :return: A rough estimate of the contrast between the skin and the bottom fourth of the image
    """
    count = 0
    sum = 0
    count_noise = 0
    sum_noise = 0
    seed_height = seed[1]
    for i in range(int(w / 2) - 3, int(w / 2) + 3):
        for j in range((seed_height - 3), (seed_height + 3)):
            sum = sum + img[i, j, 2]
            count = count + 1
    for k in range(w):
        for m in range(int(.75 * h), h):
            sum_noise = sum_noise + img[k, m, 2]
            count_noise = count_noise + 1
    avg = sum / count
    avg_noise = sum_noise / count_noise
    contrast = (avg-avg_noise) / avg_noise
    return contrast


def confidence_connect(image, seed, mult, num_iter=100, int_negrad=2):
    """

    :param image: Simple itk image object
    :param seed: Seed list with all the seeds from the find_seeds method
    :param mult: Statistic multiplier for mean and stdev to be used in confidence connect method
    :return: itk image object that is the segmentation of the image overlaid on the original image
    """
    seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    seg.CopyInformation(image)
    for i in range(len(seed)):
        index = seed[i]
        seg[index] = 1
    seg = sitk.BinaryDilate(seg, 3)
    seg = sitk.ConfidenceConnected(image, seedList=seed,
                                   numberOfIterations=num_iter,
                                   multiplier= mult,
                                   initialNeighborhoodRadius=int_negrad,
                                   replaceValue=1)
    vectorRadius = (20, 20, 20)
    vectorRad2 = (5, 5, 5)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(seg, vectorRadius, kernel)
    seg_clean1 = sitk.BinaryMorphologicalOpening(seg_clean, vectorRad2, kernel)
    #myshow(sitk.LabelOverlay(image, seg_clean), "Confidence Connected: 20 Iter, mult=1.6, neig.rad=150")
    return seg_clean1


def get_tops(mask, h, w):
    """Gets a cubic regression top edge given a binary mask

    Args:
        otsu: dtype uint8 numpy array with non-skin as 0
            and skin as 255
        h: height of input
        w: width of input

    Returns:
        numpy array of size (w) of pixel positions of top edge

    """
    # Get tops
    tops = np.zeros(w)
    for i in range(w):
        column = mask[2, :, i]
        for j in range(40, h):
            if column[j] == 1:
                tops[i] = j
                break
        else:
            tops[i] = h
    # Refine tops 1/2
    newx = []
    newtops = []
    for i in range(w):
        if 40 < tops[i] < 250:
            newx.append(i)
            newtops.append(tops[i])
    if newx:
        x = newx
        tops = newtops
    else:
        print('Failed to find tops')
        return np.poly1d([0])
    z = np.polyfit(x, tops, 1)  # Linear
    p = np.poly1d(z)
    # Refine tops 2/2
    err = p(x) - tops
    abserr = np.abs(err)
    std = np.std(err)
    newx = []
    newtops = []
    for i in range(len(err)):
        if abserr[i] <= std:  # (Felix) I changed `<=`
            newx.append(x[i])
            newtops.append(tops[i])
    #if not newx:             # (Felix) ... which should prevent this condition
     #   print('Failed to find tops')
        #return np.poly1d([0])
    z = np.polyfit(newx, newtops, 3)  # Cubic
    p = np.poly1d(z)
    return p


def get_array(mask):
    """

    :param mask: simple itk image object that is the segmentation of the skin
    :return: numpy array of the segmentation pixel values
    """
    array = sitk.GetArrayFromImage(mask)
    return array


def get_bottom(array, h, w, p):
    """

    :param array: dtype uint8 numpy array with non-skin as 0
            and skin as 255
    :param h: height of image in pixels
    :param w: width of image in pixels
    :param p: cubic function fit to the top of the image
    :return: cubic function fit to the bottom of the image
    """
    # Get bottom
    bots = np.zeros(w)
    for i in range(w):
        column = array[2, :, i]
        pval = int(p(i))
        if pval < 0:
            pval = 100
        for j in range(pval + 10, h):  # start another 10 pixels in
            if column[j] == 0:
                bots[i] = j
                break
        else:
            bots[i] = h
    # Refine bots 1/2
    newx = []
    newbots = []
    for i in range(w):
        pval = int(p(i))
        if pval < 0:
            pval = 100
        if pval < bots[i] < 250:  # min thickness 25 pixels, max 150 pixels
            newx.append(i)
            newbots.append(bots[i])
    if newx:
        x = newx
        bots = newbots
    else:
        print('Failed to find bots')
        return np.poly1d([0])
    z = np.polyfit(x, bots, 1)
    q = np.poly1d(z)
    # Refine bots 2/2
    mae = np.abs(q(x) - bots)
    std = np.std(mae)
    newx = []
    newbots = []
    for i in range(len(mae)):
        if mae[i] <= std:
            newx.append(x[i])
            newbots.append(bots[i])
    if not newx:
        print('Failed to find bottoms')
        return np.poly1d([0])
    z = np.polyfit(newx, newbots, 3)
    q = np.poly1d(z)
    return q


def get_bottoms(array, h, w):
    """

    :param array: numpy array is the segmentation mask from confidence connect method
    :param h: height of image
    :param w: width of image
    :return: pixel value that bot of skin occurs at
    """
    flipped = np.flipud(array[1, :, :])
    run = write_bot_image(flipped)
    array_run = get_array(run)
    bot_func = get_tops(array_run, h, w)
    bot = (h-1) - bot_func(w/2)
    return bot


def fill_bright(image, mask, top, bot, w, border_correct=10):
    """

    :param image: itk image file containing the orginal image
    :param mask: itk image file containing the segmentation of the image
    :param top: line equation that is fitted to the top of the skin
    :param bot: line equation that is fitted to the bottom of the skin
    :param w: pixel width of the image
    :param border_correct: range of pixels that can be corrected over
    :return: border and hole filled segmenetation of the image
    """
    for i in range(w):
        p = int(top(i))
        f = int(bot(i))
        if p < 0 or f < 0:
            break
        for d in range(p, f):
            mask[i, d, 0] = 1
            mask[i, d, 1] = 1
            mask[i, d, 2] = 1
            mask[i, d, 3] = 1
        for j in range((p-border_correct), p):
            baseline_pix = image[i, p, 2]
            test_pix = image[i, j, 2]
            if test_pix > baseline_pix:
                for k in range(j, p):
                    mask[i, k, 0] = 1
                    mask[i, k, 1] = 1
                    mask[i, k, 2] = 1
                    mask[i, k, 3] = 1
        for r in range(f, (f + border_correct)):
            baseline_pix = image[i, f, 2]
            test_pix = image[i, r, 2]
            if test_pix > baseline_pix:
                for y in range(f, r):
                    mask[i, y, 0] = 1
                    mask[i, y, 1] = 1
                    mask[i, y, 2] = 1
                    mask[i, y, 3] = 1
    return mask


def run_csvwith_mat(csvname, mult=1.6, num_seed=10):
    """

    :param csvname: file that will be read and wrote, end product
    is csvname will contain top and bottom values for all ran files.
    This function show be used if files being ran are .mat files.
    :param mult: float that controls multplier for confidence interval
    :param num_seed: integer that controls how many seeds occur in the confidence
    connect method

    """
    df = pd.read_csv(csvname)
    files = sorted(os.listdir(path))
    rows, col = df.shape
    print('Total files: {}'.format(len(files)))

    tops = []
    bots = []
    # con_list = []
    for file in files:
        logging.info(file)
        img, h, w, spacing_ax, spacing_lat = load_image(file)
        itk_image = write_itk(img, spacing_ax, spacing_lat)
        seed = find_seed(img, w, h, num_seed)
        # con = get_contrast(itk_image, w, h, seed[0])
        # con_list.append(con)
        seg = confidence_connect(itk_image, seed, mult)
        array_full = get_array(seg)
        try:
            p = get_tops(array_full, h, w)
        except:
            tops.append(0)
            bots.append(0)
            continue
        try:
            q = get_bottom(array_full, h, w, p)
            seg3 = fill_bright(itk_image, seg, p, q, w, 10)
            array_full_post = get_array(seg3)
            g = get_tops(array_full_post, h, w)
            q1 = get_bottom(array_full_post, h, w, p)
        except:
            tops.append(0)
            bots.append(0)
            continue
        midpoint = w / 2
        ytop = g(midpoint)
        ybot = q1(midpoint)
        print(ytop)
        print(ybot)
        tops.append(ytop)
        bots.append(ybot)

    df.File = files
    df.Top = tops
    df.Bottom = bots
    # df.Contrast = con_list
    df.to_csv(csvname, index=False)


def run_csvwith_png(csvname, mult=1.6, num_seed=10):
    """

    :param csvname: file that will be read and wrote, end product
    is csvname will contain top and bottom values for all ran files.
    This function show be used if files being ran are .png files.
    :param mult: float that controls multplier for confidence interval
    :param num_seed: integer that controls how many seeds occur in the confidence
    connect method
    """
    df = pd.read_csv(csvname)
    files = df['Name']
    tops = []
    bots = []
    # con_list = []

    for file in files:
        logging.info(file)
        img = load_image_png(file)
        h, w = img.shape
        itk_image = write_itk_png(img)
        seed = find_seed(img, w, h, num_seed)
        seg = confidence_connect(itk_image, seed, mult)
        array_full = get_array(seg)
        try:
            p = get_tops(array_full, h, w)
        except:
            tops.append(0)
            bots.append(0)
            continue
        try:
            q = get_bottom(array_full, h, w, p)
            seg3 = fill_bright(itk_image, seg, p, q, w, 30)
            array_full_post = get_array(seg3)
            p1 = get_tops(array_full_post, h, w)
            q1 = get_bottom(array_full_post, h, w, p)
        except:
            tops.append(0)
            bots.append(0)
            continue
        midpoint = w // 2
        ytop = p1(midpoint)
        ybot = q1(midpoint)
        tops.append(ytop)
        bots.append(ybot)

    df.ITKtop = tops
    df.ITKbot = bots
    df.to_csv(csvname, index=False)


if __name__ == '__main__':
    run_csvwith_mat(csvname)