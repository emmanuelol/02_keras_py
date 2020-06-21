from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import cv2
from scipy import ndimage

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.testutils import keypoints_equal, reseed
from imgaug.augmenters import meta


def main():
    time_start = time.time()

    test_GaussianBlur()
    test_AverageBlur()
    test_MedianBlur()
    # TODO BilateralBlur

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_GaussianBlur():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 255, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]
    outer_pixels = ([], [])
    for i in sm.xrange(base_img.shape[0]):
        for j in sm.xrange(base_img.shape[1]):
            if i != j:
                outer_pixels[0].append(i)
                outer_pixels[1].append(j)

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no blur, shouldnt change anything
    aug = iaa.GaussianBlur(sigma=0)

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    # weak blur of center pixel
    aug = iaa.GaussianBlur(sigma=0.5)
    aug_det = aug.to_deterministic()

    # images as numpy array
    observed = aug.augment_images(images)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    observed = aug_det.augment_images(images)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    # images as list
    observed = aug.augment_images(images_list)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    observed = aug_det.augment_images(images_list)
    assert 100 < observed[0][1, 1] < 255
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 0).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 50).all()

    # keypoints shouldnt be changed
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # varying blur sigmas
    aug = iaa.GaussianBlur(sigma=(0, 1))
    aug_det = aug.to_deterministic()

    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    for i in sm.xrange(nb_iterations):
        observed_aug = aug.augment_images(images)
        observed_aug_det = aug_det.augment_images(images)
        if i == 0:
            last_aug = observed_aug
            last_aug_det = observed_aug_det
        else:
            if not np.array_equal(observed_aug, last_aug):
                nb_changed_aug += 1
            if not np.array_equal(observed_aug_det, last_aug_det):
                nb_changed_aug_det += 1
            last_aug = observed_aug
            last_aug_det = observed_aug_det
    assert nb_changed_aug >= int(nb_iterations * 0.8)
    assert nb_changed_aug_det == 0

    #############################
    # test other dtypes below
    # ndimage.gaussian_filter() rejects: float16
    # float64 implementation in gaussian_filter() was too inaccurate
    #############################

    # --
    # blur of various dtypes at sigma=0
    # --
    aug = iaa.GaussianBlur(sigma=0)

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == image)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
        _min_value, center_value, _max_value = meta.get_value_range_of_dtype(dtype)
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = int(center_value)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == image)

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        _min_value, center_value, _max_value = meta.get_value_range_of_dtype(dtype)
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = center_value
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.allclose(image_aug, image)

    # --
    # blur of various dtypes at sigma=1.0
    # and using an example value of 100 for int/uint/float and True for bool
    # --
    aug = iaa.GaussianBlur(sigma=1.0)

    # prototype kernel, generated via:
    #  mask = np.zeros((3, 3), dtype=np.float64)
    #  mask[1, 1] = 1.0
    #  mask = ndimage.gaussian_filter(mask, 1.0)
    kernel = np.float64([
        [0.08767308, 0.12075024, 0.08767308],
        [0.12075024, 0.16630671, 0.12075024],
        [0.08767308, 0.12075024, 0.08767308]
    ])

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image_aug = aug.augment_image(image)
    expected = kernel > 0.5
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == expected)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100
        image_aug = aug.augment_image(image)
        expected = (kernel * 100).astype(dtype)
        diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) <= 2

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100.0
        image_aug = aug.augment_image(image)
        expected = (kernel * 100.0).astype(dtype)
        diff = np.abs(image_aug.astype(np.float128) - expected.astype(np.float128))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) < 1.0

    # --
    # blur of various dtypes at sigma=0.4
    # and using an example value of 100 for int/uint/float and True for bool
    # --
    aug = iaa.GaussianBlur(sigma=0.4)

    # prototype kernel, generated via:
    #  mask = np.zeros((3, 3), dtype=np.float64)
    #  mask[1, 1] = 1.0
    #  kernel = ndimage.gaussian_filter(mask, 0.4)
    kernel = np.float64([
        [0.00163144, 0.03712817, 0.00163144],
        [0.03712817, 0.84496158, 0.03712817],
        [0.00163144, 0.03712817, 0.00163144]
    ])

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image_aug = aug.augment_image(image)
    expected = kernel > 0.5
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == expected)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100
        image_aug = aug.augment_image(image)
        expected = (kernel * 100).astype(dtype)
        diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) <= 2

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100.0
        image_aug = aug.augment_image(image)
        expected = (kernel * 100.0).astype(dtype)
        diff = np.abs(image_aug.astype(np.float128) - expected.astype(np.float128))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) < 1.0

    # --
    # blur of various dtypes at sigma=0.75
    # and values being half-way between center and maximum for each dtype (bool is skipped as it doesnt make any
    # sense here)
    # The goal of this test is to verify that no major loss of resolution happens for large dtypes.
    # Such inaccuracies appear for float64 if used.
    # --
    aug = iaa.GaussianBlur(sigma=0.75)

    # prototype kernel, generated via:
    # mask = np.zeros((3, 3), dtype=np.float64)
    # mask[1, 1] = 1.0
    # kernel = ndimage.gaussian_filter(mask, 0.75)
    kernel = np.float64([
        [0.05469418, 0.12447951, 0.05469418],
        [0.12447951, 0.28330525, 0.12447951],
        [0.05469418, 0.12447951, 0.05469418]
    ])

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
        _min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
        value = int(center_value + 0.4 * max_value)
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = value
        image_aug = aug.augment_image(image)
        expected = (kernel * value).astype(dtype)
        diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
        assert image_aug.dtype.type == dtype
        # accepts difference of 4, 8, 16 (at 1, 2, 4 bytes, i.e. 8, 16, 32 bit)
        assert np.max(diff) <= 2**(1 + np.dtype(dtype).itemsize)

    # float
    for dtype, value in zip([np.float16, np.float32, np.float64], [5000, 1000*1000, 1000*1000*1000]):
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = value
        image_aug = aug.augment_image(image)
        expected = (kernel * value).astype(dtype)
        diff = np.abs(image_aug.astype(np.float128) - expected.astype(np.float128))
        assert image_aug.dtype.type == dtype
        # accepts difference of 2.0, 4.0, 8.0, 16.0 (at 1, 2, 4, 8 bytes, i.e. 8, 16, 32, 64 bit)
        assert np.max(diff) < 2**(1 + np.dtype(dtype).itemsize)

    # assert failure on invalid dtypes
    aug = iaa.GaussianBlur(sigma=1.0)
    for dt in [np.uint64, np.int64, np.float128]:
        got_exception = False
        try:
            _ = aug.augment_image(np.zeros((1, 1), dtype=dt))
        except Exception as exc:
            assert "forbidden dtype" in str(exc)
            got_exception = True
        assert got_exception


def test_AverageBlur():
    reseed()

    base_img = np.zeros((11, 11, 1), dtype=np.uint8)
    base_img[5, 5, 0] = 200
    base_img[4, 5, 0] = 100
    base_img[6, 5, 0] = 100
    base_img[5, 4, 0] = 100
    base_img[5, 6, 0] = 100

    blur3x3 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 11, 11, 11, 0, 0, 0, 0],
        [0, 0, 0, 11, 44, 56, 44, 11, 0, 0, 0],
        [0, 0, 0, 11, 56, 67, 56, 11, 0, 0, 0],
        [0, 0, 0, 11, 44, 56, 44, 11, 0, 0, 0],
        [0, 0, 0, 0, 11, 11, 11, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    blur3x3 = np.array(blur3x3, dtype=np.uint8)[..., np.newaxis]

    blur4x4 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0],
        [0, 0, 0, 6, 25, 31, 31, 25, 6, 0, 0],
        [0, 0, 0, 6, 31, 38, 38, 31, 6, 0, 0],
        [0, 0, 0, 6, 31, 38, 38, 31, 6, 0, 0],
        [0, 0, 0, 6, 25, 31, 31, 25, 6, 0, 0],
        [0, 0, 0, 0, 6, 6, 6, 6, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    blur4x4 = np.array(blur4x4, dtype=np.uint8)[..., np.newaxis]

    blur5x5 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 0, 4, 16, 20, 20, 20, 16, 4, 0, 0],
        [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
        [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
        [0, 0, 4, 20, 24, 24, 24, 20, 4, 0, 0],
        [0, 0, 4, 16, 20, 20, 20, 16, 4, 0, 0],
        [0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    blur5x5 = np.array(blur5x5, dtype=np.uint8)[..., np.newaxis]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no blur, shouldnt change anything
    aug = iaa.AverageBlur(k=0)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, base_img)

    # k=3
    aug = iaa.AverageBlur(k=3)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, blur3x3)

    # k=5
    aug = iaa.AverageBlur(k=5)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, blur5x5)

    # k as (3, 4)
    aug = iaa.AverageBlur(k=(3, 4))
    nb_iterations = 100
    nb_seen = [0, 0]
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            nb_seen[0] += 1
        elif np.array_equal(observed, blur4x4):
            nb_seen[1] += 1
        else:
            raise Exception("Unexpected result in AverageBlur@1")
    p_seen = [v/nb_iterations for v in nb_seen]
    assert 0.4 <= p_seen[0] <= 0.6
    assert 0.4 <= p_seen[1] <= 0.6

    # k as (3, 5)
    aug = iaa.AverageBlur(k=(3, 5))
    nb_iterations = 100
    nb_seen = [0, 0, 0]
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            nb_seen[0] += 1
        elif np.array_equal(observed, blur4x4):
            nb_seen[1] += 1
        elif np.array_equal(observed, blur5x5):
            nb_seen[2] += 1
        else:
            raise Exception("Unexpected result in AverageBlur@2")
    p_seen = [v/nb_iterations for v in nb_seen]
    assert 0.23 <= p_seen[0] <= 0.43
    assert 0.23 <= p_seen[1] <= 0.43
    assert 0.23 <= p_seen[2] <= 0.43

    # k as stochastic parameter
    aug = iaa.AverageBlur(k=iap.Choice([3, 5]))
    nb_iterations = 100
    nb_seen = [0, 0]
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            nb_seen[0] += 1
        elif np.array_equal(observed, blur5x5):
            nb_seen[1] += 1
        else:
            raise Exception("Unexpected result in AverageBlur@3")
    p_seen = [v/nb_iterations for v in nb_seen]
    assert 0.4 <= p_seen[0] <= 0.6
    assert 0.4 <= p_seen[1] <= 0.6

    # k as ((3, 5), (3, 5))
    aug = iaa.AverageBlur(k=((3, 5), (3, 5)))

    possible = dict()
    for kh in [3, 4, 5]:
        for kw in [3, 4, 5]:
            key = (kh, kw)
            if kh == 0 or kw == 0:
                possible[key] = np.copy(base_img)
            else:
                possible[key] = cv2.blur(base_img, (kh, kw))[..., np.newaxis]

    nb_iterations = 250
    nb_seen = dict([(key, 0) for key, val in possible.items()])
    for i in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        for key, img_aug in possible.items():
            if np.array_equal(observed, img_aug):
                nb_seen[key] += 1
    # dont check sum here, because 0xX and Xx0 are all the same, i.e. much
    # higher sum than nb_iterations
    assert all([v > 0 for v in nb_seen.values()])

    # keypoints shouldnt be changed
    aug = iaa.AverageBlur(k=3)
    aug_det = aug.to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    #############################
    # test other dtypes below
    #############################

    # --
    # blur of various dtypes at k=0
    # --
    aug = iaa.AverageBlur(k=0)

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image[2, 2] = True
    image_aug = aug.augment_image(image)
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == image)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        _min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = int(center_value + 0.4 * max_value)
        image[2, 2] = int(center_value + 0.4 * max_value)
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.all(image_aug == image)

    # float
    for dtype, value in zip([np.float16, np.float32, np.float64], [5000, 1000*1000, 1000*1000*1000]):
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = value
        image[2, 2] = value
        image_aug = aug.augment_image(image)
        assert image_aug.dtype.type == dtype
        assert np.allclose(image_aug, image)

    # --
    # blur of various dtypes at k=3
    # and using an example value of 100 for int/uint/float and True for bool
    # --
    aug = iaa.AverageBlur(k=3)

    # prototype mask
    # we place values in a 3x3 grid at positions (row=1, col=1) and (row=2, col=2) (beginning with 0)
    # AverageBlur uses cv2.blur(), which uses BORDER_REFLECT_101 as its default padding mode,
    # see https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html
    # the matrix below shows the 3x3 grid and the padded row/col values around it
    # [1, 0, 1, 0, 1]
    # [0, 0, 0, 0, 0]
    # [1, 0, 1, 0, 1]
    # [0, 0, 0, 1, 0]
    # [1, 0, 1, 0, 1]
    mask = np.float64([
        [4/9, 2/9, 4/9],
        [2/9, 2/9, 3/9],
        [4/9, 3/9, 5/9]
    ])

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image[2, 2] = True
    image_aug = aug.augment_image(image)
    expected = mask > 0.5
    assert image_aug.dtype.type == np.bool_
    assert np.all(image_aug == expected)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100
        image[2, 2] = 100
        image_aug = aug.augment_image(image)
        expected = np.round(mask * 100).astype(dtype)  # cv2.blur() applies rounding for int/uint dtypes
        diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) <= 2

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100.0
        image[2, 2] = 100.0
        image_aug = aug.augment_image(image)
        expected = (mask * 100.0).astype(dtype)
        diff = np.abs(image_aug.astype(np.float128) - expected.astype(np.float128))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) < 1.0

    # --
    # blur of various dtypes at k=3
    # and values being half-way between center and maximum for each dtype (bool is skipped as it doesnt make any
    # sense here)
    # The goal of this test is to verify that no major loss of resolution happens for large dtypes.
    # --
    aug = iaa.AverageBlur(k=3)

    # prototype mask (see above)
    mask = np.float64([
        [4/9, 2/9, 4/9],
        [2/9, 2/9, 3/9],
        [4/9, 3/9, 5/9]
    ])

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        _min_value, center_value, max_value = meta.get_value_range_of_dtype(dtype)
        value = int(center_value + 0.4 * max_value)
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = value
        image[2, 2] = value
        image_aug = aug.augment_image(image)
        expected = (mask * value).astype(dtype)
        diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
        assert image_aug.dtype.type == dtype
        # accepts difference of 4, 8, 16 (at 1, 2, 4 bytes, i.e. 8, 16, 32 bit)
        assert np.max(diff) <= 2**(1 + np.dtype(dtype).itemsize)

    # float
    for dtype, value in zip([np.float16, np.float32, np.float64], [5000, 1000*1000, 1000*1000*1000]):
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = value
        image[2, 2] = value
        image_aug = aug.augment_image(image)
        expected = (mask * value).astype(dtype)
        diff = np.abs(image_aug.astype(np.float128) - expected.astype(np.float128))
        assert image_aug.dtype.type == dtype
        # accepts difference of 2.0, 4.0, 8.0, 16.0 (at 1, 2, 4, 8 bytes, i.e. 8, 16, 32, 64 bit)
        assert np.max(diff) < 2**(1 + np.dtype(dtype).itemsize)

    # assert failure on invalid dtypes
    aug = iaa.AverageBlur(k=3)
    for dt in [np.uint32, np.uint64, np.int32, np.int64]:
        got_exception = False
        try:
            _ = aug.augment_image(np.zeros((1, 1), dtype=dt))
        except Exception as exc:
            assert "forbidden dtype" in str(exc)
            got_exception = True
        assert got_exception


def test_MedianBlur():
    reseed()

    base_img = np.zeros((11, 11, 1), dtype=np.uint8)
    base_img[3:8, 3:8, 0] = 1
    base_img[4:7, 4:7, 0] = 2
    base_img[5:6, 5:6, 0] = 3

    blur3x3 = np.zeros_like(base_img)
    blur3x3[3:8, 3:8, 0] = 1
    blur3x3[4:7, 4:7, 0] = 2
    blur3x3[4, 4, 0] = 1
    blur3x3[4, 6, 0] = 1
    blur3x3[6, 4, 0] = 1
    blur3x3[6, 6, 0] = 1
    blur3x3[3, 3, 0] = 0
    blur3x3[3, 7, 0] = 0
    blur3x3[7, 3, 0] = 0
    blur3x3[7, 7, 0] = 0

    blur5x5 = np.copy(blur3x3)
    blur5x5[4, 3, 0] = 0
    blur5x5[3, 4, 0] = 0
    blur5x5[6, 3, 0] = 0
    blur5x5[7, 4, 0] = 0
    blur5x5[4, 7, 0] = 0
    blur5x5[3, 6, 0] = 0
    blur5x5[6, 7, 0] = 0
    blur5x5[7, 6, 0] = 0
    blur5x5[blur5x5 > 1] = 1

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    # no blur, shouldnt change anything
    aug = iaa.MedianBlur(k=1)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, base_img)

    # k=3
    aug = iaa.MedianBlur(k=3)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, blur3x3)

    # k=5
    aug = iaa.MedianBlur(k=5)
    observed = aug.augment_image(base_img)
    assert np.array_equal(observed, blur5x5)

    # k as (3, 5)
    aug = iaa.MedianBlur(k=(3, 5))
    seen = [False, False]
    for i in sm.xrange(100):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            seen[0] = True
        elif np.array_equal(observed, blur5x5):
            seen[1] = True
        else:
            raise Exception("Unexpected result in MedianBlur@1")
        if all(seen):
            break
    assert all(seen)

    # k as stochastic parameter
    aug = iaa.MedianBlur(k=iap.Choice([3, 5]))
    seen = [False, False]
    for i in sm.xrange(100):
        observed = aug.augment_image(base_img)
        if np.array_equal(observed, blur3x3):
            seen[0] += True
        elif np.array_equal(observed, blur5x5):
            seen[1] += True
        else:
            raise Exception("Unexpected result in MedianBlur@2")
        if all(seen):
            break
    assert all(seen)

    # keypoints shouldnt be changed
    aug = iaa.MedianBlur(k=3)
    aug_det = aug.to_deterministic()
    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)


def test_MotionBlur():
    reseed()

    # simple scenario
    aug = iaa.MotionBlur(k=3, angle=0, direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 1.0/3, 0],
        [0, 1.0/3, 0],
        [0, 1.0/3, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected)

    # 90deg angle
    aug = iaa.MotionBlur(k=3, angle=90, direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 0, 0],
        [1.0/3, 1.0/3, 1.0/3],
        [0, 0, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected)

    # 45deg angle
    aug = iaa.MotionBlur(k=3, angle=45, direction=0.0, order=0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 0, 1.0/3],
        [0, 1.0/3, 0],
        [1.0/3, 0, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected)

    # random angle
    aug = iaa.MotionBlur(k=3, angle=[0, 90], direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(50)]
    expected1 = np.float32([
        [0, 1.0/3, 0],
        [0, 1.0/3, 0],
        [0, 1.0/3, 0]
    ])
    expected2 = np.float32([
        [0, 0, 0],
        [1.0/3, 1.0/3, 1.0/3],
        [0, 0, 0],
    ])
    nb_seen = [0, 0]
    for matrices_image in matrices:
        assert np.allclose(matrices_image[0], matrices_image[1])
        assert np.allclose(matrices_image[1], matrices_image[2])
        for matrix_channel in matrices_image:
            if np.allclose(matrix_channel, expected1):
                nb_seen[0] += 1
            elif np.allclose(matrix_channel, expected2):
                nb_seen[1] += 1
    assert nb_seen[0] > 0
    assert nb_seen[1] > 0

    # 5x5
    aug = iaa.MotionBlur(k=5, angle=90, direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected)

    # random k
    aug = iaa.MotionBlur(k=[3, 5], angle=90, direction=0.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(50)]
    expected1 = np.float32([
        [0, 0, 0],
        [1.0/3, 1.0/3, 1.0/3],
        [0, 0, 0],
    ])
    expected2 = np.float32([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1.0/5, 1.0/5, 1.0/5, 1.0/5, 1.0/5],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    nb_seen = [0, 0]
    for matrices_image in matrices:
        assert np.allclose(matrices_image[0], matrices_image[1])
        assert np.allclose(matrices_image[1], matrices_image[2])
        for matrix_channel in matrices_image:
            if matrix_channel.shape == expected1.shape and np.allclose(matrix_channel, expected1):
                nb_seen[0] += 1
            elif matrix_channel.shape == expected2.shape and np.allclose(matrix_channel, expected2):
                nb_seen[1] += 1
    assert nb_seen[0] > 0
    assert nb_seen[1] > 0

    # k with choice [a, b, c, ...] must error in case of non-discrete values
    got_exception = False
    try:
        _ = iaa.MotionBlur(k=[3, 3.5, 4])
    except Exception as exc:
        assert "to only contain integer" in str(exc)
        got_exception = True
    assert got_exception

    # no error in case of (a, b), checks for #215
    aug = iaa.MotionBlur(k=(3, 7))
    for _ in range(10):
        _ = aug.augment_image(np.zeros((11, 11, 3), dtype=np.uint8))

    # direction 1.0
    aug = iaa.MotionBlur(k=3, angle=0, direction=1.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 1.0/1.5, 0],
        [0, 0.5/1.5, 0],
        [0, 0.0/1.5, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected, rtol=0, atol=1e-2)

    # direction -1.0
    aug = iaa.MotionBlur(k=3, angle=0, direction=-1.0)
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(10)]
    expected = np.float32([
        [0, 0.0/1.5, 0],
        [0, 0.5/1.5, 0],
        [0, 1.0/1.5, 0]
    ])
    for matrices_image in matrices:
        for matrix_channel in matrices_image:
            assert np.allclose(matrix_channel, expected, rtol=0, atol=1e-2)

    # random direction
    aug = iaa.MotionBlur(k=3, angle=[0, 90], direction=[-1.0, 1.0])
    matrix_func = aug.matrix
    matrices = [matrix_func(np.zeros((128, 128, 3), dtype=np.uint8), 3, ia.new_random_state(i)) for i in range(50)]
    expected1 = np.float32([
        [0, 1.0/1.5, 0],
        [0, 0.5/1.5, 0],
        [0, 0.0/1.5, 0]
    ])
    expected2 = np.float32([
        [0, 0.0/1.5, 0],
        [0, 0.5/1.5, 0],
        [0, 1.0/1.5, 0]
    ])
    nb_seen = [0, 0]
    for matrices_image in matrices:
        assert np.allclose(matrices_image[0], matrices_image[1])
        assert np.allclose(matrices_image[1], matrices_image[2])
        for matrix_channel in matrices_image:
            if np.allclose(matrix_channel, expected1, rtol=0, atol=1e-2):
                nb_seen[0] += 1
            elif np.allclose(matrix_channel, expected2, rtol=0, atol=1e-2):
                nb_seen[1] += 1
    assert nb_seen[0] > 0
    assert nb_seen[1] > 0

    # test of actual augmenter
    img = np.zeros((7, 7, 3), dtype=np.uint8)
    img[3-1:3+2, 3-1:3+2, :] = 255
    aug = iaa.MotionBlur(k=3, angle=90, direction=0.0)
    img_aug = aug.augment_image(img)
    v1 = (255*(1/3))
    v2 = (255*(1/3)) * 2
    v3 = (255*(1/3)) * 3
    expected = np.float32([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, v1, v2, v3, v2, v1, 0],
        [0, v1, v2, v3, v2, v1, 0],
        [0, v1, v2, v3, v2, v1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]).astype(np.uint8)
    expected = np.tile(expected[..., np.newaxis], (1, 1, 3))
    assert np.allclose(img_aug, expected)


if __name__ == "__main__":
    main()
