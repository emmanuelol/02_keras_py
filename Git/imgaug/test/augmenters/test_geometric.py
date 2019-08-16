from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import skimage.morphology

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug.testutils import array_equal_lists, keypoints_equal, reseed


def main():
    time_start = time.time()

    test_Affine()
    test_AffineCv2()
    test_PiecewiseAffine()
    test_PerspectiveTransform()
    test_ElasticTransformation()
    test_Rot90()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_Affine():
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

    # no translation/scale/rotate/shear, shouldnt change nothing
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # ---------------------
    # scale
    # ---------------------
    # zoom in
    aug = iaa.Affine(scale=1.75, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    # zoom in only on x axis
    aug = iaa.Affine(scale={"x": 1.75, "y": 1.0}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    # zoom in only on y axis
    aug = iaa.Affine(scale={"x": 1.0, "y": 1.75}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    # zoom out
    # this one uses a 4x4 area of all 255, which is zoomed out to a 4x4 area
    # in which the center 2x2 area is 255
    # zoom in should probably be adapted to this style
    # no separate tests here for x/y axis, should work fine if zoom in works with that
    aug = iaa.Affine(scale=0.49, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.ones((4, 4, 1), dtype=np.uint8) * 255
    images = np.array([image])
    images_list = [image]
    outer_pixels = ([], [])
    for y in sm.xrange(4):
        xs = sm.xrange(4) if y in [0, 3] else [0, 3]
        for x in xs:
            outer_pixels[0].append(y)
            outer_pixels[1].append(x)
    inner_pixels = ([1, 1, 2, 2], [1, 2, 1, 2])
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=0),
                                      ia.Keypoint(x=0, y=3), ia.Keypoint(x=3, y=3)],
                                     shape=image.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=0.765, y=0.765), ia.Keypoint(x=2.235, y=0.765),
                                          ia.Keypoint(x=0.765, y=2.235), ia.Keypoint(x=2.235, y=2.235)],
                                         shape=image.shape)]

    observed = aug.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # varying scales
    aug = iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, translate_px=0,
                     rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.array([[0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 2, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0]], dtype=np.uint8) * 100
    image = image[:, :, np.newaxis]
    images = np.array([image])

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

    aug = iaa.Affine(scale=iap.Uniform(0.7, 0.9))
    assert isinstance(aug.scale, iap.Uniform)
    assert isinstance(aug.scale.a, iap.Deterministic)
    assert isinstance(aug.scale.b, iap.Deterministic)
    assert 0.7 - 1e-8 < aug.scale.a.value < 0.7 + 1e-8
    assert 0.9 - 1e-8 < aug.scale.b.value < 0.9 + 1e-8

    # ---------------------
    # translate
    # ---------------------
    # move one pixel to the right
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move one pixel to the right
    # with backend = skimage
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="skimage")
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = skimage
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="skimage")
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = skimage, order=ALL
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="skimage", order=ia.ALL)
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = skimage, order=list
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="skimage", order=[0, 1, 3])
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = cv2, order=list
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="cv2", order=[0, 1, 3])
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with backend = cv2, order=StochasticParameter
    aug = iaa.Affine(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, backend="cv2", order=iap.Choice([0, 1, 3]))
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the bottom
    aug = iaa.Affine(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the right
    aug = iaa.Affine(scale=1.0, translate_percent={"x": 0.3333, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the bottom
    aug = iaa.Affine(scale=1.0, translate_percent={"x": 0, "y": 0.3333}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # 0-1px to left/right and 0-1px to top/bottom
    aug = iaa.Affine(scale=1.0, translate_px={"x": (-1, 1), "y": (-1, 1)}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    centers_aug = np.copy(image).astype(np.int32) * 0
    centers_aug_det = np.copy(image).astype(np.int32) * 0
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

        assert len(observed_aug[0].nonzero()[0]) == 1
        assert len(observed_aug_det[0].nonzero()[0]) == 1
        centers_aug += (observed_aug[0] > 0)
        centers_aug_det += (observed_aug_det[0] > 0)

    assert nb_changed_aug >= int(nb_iterations * 0.7)
    assert nb_changed_aug_det == 0
    assert (centers_aug > int(nb_iterations * (1/9 * 0.6))).all()
    assert (centers_aug < int(nb_iterations * (1/9 * 1.4))).all()

    aug = iaa.Affine(translate_percent=iap.Uniform(0.7, 0.9))
    assert isinstance(aug.translate, iap.Uniform)
    assert isinstance(aug.translate.a, iap.Deterministic)
    assert isinstance(aug.translate.b, iap.Deterministic)
    assert 0.7 - 1e-8 < aug.translate.a.value < 0.7 + 1e-8
    assert 0.9 - 1e-8 < aug.translate.b.value < 0.9 + 1e-8

    aug = iaa.Affine(translate_px=iap.DiscreteUniform(1, 10))
    assert isinstance(aug.translate, iap.DiscreteUniform)
    assert isinstance(aug.translate.a, iap.Deterministic)
    assert isinstance(aug.translate.b, iap.Deterministic)
    assert aug.translate.a.value == 1
    assert aug.translate.b.value == 10

    # ---------------------
    # translate heatmaps
    # ---------------------
    heatmaps = ia.HeatmapsOnImage(
        np.float32([
            [0.0, 0.5, 0.75],
            [0.0, 0.5, 0.75],
            [0.75, 0.75, 0.75],
        ]),
        shape=(3, 3, 3)
    )
    arr_expected_1px_right = np.float32([
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.75, 0.75],
    ])
    aug = iaa.Affine(translate_px={"x": 1})
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    # should still use mode=constant cval=0 even when other settings chosen
    aug = iaa.Affine(translate_px={"x": 1}, cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    aug = iaa.Affine(translate_px={"x": 1}, mode="edge", cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    # ---------------------
    # rotate
    # ---------------------
    # rotate by 45 degrees
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=90, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, :] = 255
    image_aug[0, 1] = 255
    image_aug[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # rotate by StochasticParameter
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=iap.Uniform(10, 20), shear=0)
    assert isinstance(aug.rotate, iap.Uniform)
    assert isinstance(aug.rotate.a, iap.Deterministic)
    assert aug.rotate.a.value == 10
    assert isinstance(aug.rotate.b, iap.Deterministic)
    assert aug.rotate.b.value == 20

    # random rotation 0-364 degrees
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=(0, 364), shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    pixels_sums_aug = np.copy(image).astype(np.int32) * 0
    pixels_sums_aug_det = np.copy(image).astype(np.int32) * 0
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

        pixels_sums_aug += (observed_aug[0] > 100)
        pixels_sums_aug_det += (observed_aug_det[0] > 100)

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)

    # outer pixels, should sometimes be white
    # the values here had to be set quite tolerant, the middle pixels at top/left/bottom/right get more activation
    # than expected
    outer_pixels = ([0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2])
    assert (pixels_sums_aug[outer_pixels] > int(nb_iterations * (2/8 * 0.4))).all()
    assert (pixels_sums_aug[outer_pixels] < int(nb_iterations * (2/8 * 2.0))).all()

    for backend in ["auto", "cv2", "skimage"]:
        # measure alignment between images and heatmaps when rotating
        aug = iaa.Affine(rotate=45, backend=backend)
        image = np.zeros((7, 6), dtype=np.uint8)
        image[:, 2:3+1] = 255
        hm = ia.HeatmapsOnImage(image.astype(np.float32)/255, shape=(7, 6))
        img_aug = aug.augment_image(image)
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert hm_aug.shape == (7, 6)
        assert hm_aug.arr_0to1.shape == (7, 6, 1)
        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = hm_aug.arr_0to1 > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert (same / img_aug_mask.size) >= 0.99

        # measure alignment between images and heatmaps when rotating
        # here with smaller heatmaps
        aug = iaa.Affine(rotate=45, backend=backend)
        image = np.zeros((56, 48), dtype=np.uint8)
        image[:, 16:24+1] = 255
        hm = ia.HeatmapsOnImage(
            ia.imresize_single_image(image, (28, 24), interpolation="cubic").astype(np.float32)/255,
            shape=(56, 48)
        )
        img_aug = aug.augment_image(image)
        hm_aug = aug.augment_heatmaps([hm])[0]
        assert hm_aug.shape == (56, 48)
        assert hm_aug.arr_0to1.shape == (28, 24, 1)
        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = ia.imresize_single_image(hm_aug.arr_0to1, img_aug.shape[0:2], interpolation="cubic") > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert (same / img_aug_mask.size) >= 0.9

    # ---------------------
    # shear
    # ---------------------
    # TODO

    # shear by StochasticParameter
    aug = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=iap.Uniform(10, 20))
    assert isinstance(aug.shear, iap.Uniform)
    assert isinstance(aug.shear.a, iap.Deterministic)
    assert aug.shear.a.value == 10
    assert isinstance(aug.shear.b, iap.Deterministic)
    assert aug.shear.b.value == 20

    # ---------------------
    # cval
    # ---------------------
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=128)
    aug_det = aug.to_deterministic()

    image = np.ones((3, 3, 1), dtype=np.uint8) * 255
    images = np.array([image])
    images_list = [image]

    observed = aug.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    # random cvals
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=(0, 255))
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    averages = []
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

        averages.append(int(np.average(observed_aug)))

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)
    assert len(set(averages)) > 200

    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=ia.ALL)
    assert isinstance(aug.cval, iap.Uniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 255

    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=iap.DiscreteUniform(1, 5))
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 1
    assert aug.cval.b.value == 5

    # ------------
    # mode
    # ------------
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=ia.ALL)
    assert isinstance(aug.mode, iap.Choice)
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode="edge")
    assert isinstance(aug.mode, iap.Deterministic)
    assert aug.mode.value == "edge"
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=["constant", "edge"])
    assert isinstance(aug.mode, iap.Choice)
    assert len(aug.mode.a) == 2 and "constant" in aug.mode.a and "edge" in aug.mode.a
    aug = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=iap.Choice(["constant", "edge"]))
    assert isinstance(aug.mode, iap.Choice)
    assert len(aug.mode.a) == 2 and "constant" in aug.mode.a and "edge" in aug.mode.a

    # ------------
    # fit_output
    # ------------
    for backend in ["auto", "cv2", "skimage"]:
        aug = iaa.Affine(scale=1.0, translate_px=100, fit_output=True, backend=backend)
        assert aug.fit_output is True
        observed = aug.augment_images(images)
        expected = images
        assert np.array_equal(observed, expected)
        observed = aug.augment_keypoints(keypoints)
        expected = keypoints
        assert keypoints_equal(observed, expected)
        observed = aug.augment_heatmaps([heatmaps])[0]
        expected = heatmaps
        assert np.allclose(observed.arr_0to1, expected.arr_0to1)

        # fit_output with rotation
        aug = iaa.Affine(rotate=45, fit_output=True, backend=backend)
        img = np.zeros((10, 10), dtype=np.uint8)
        img[0:2, 0:2] = 255
        img[-2:, 0:2] = 255
        img[0:2, -2:] = 255
        img[-2:, -2:] = 255
        hm = ia.HeatmapsOnImage(img.astype(np.float32)/255, shape=(10, 10))
        img_aug = aug.augment_image(img)
        hm_aug = aug.augment_heatmaps([hm])[0]
        _labels, nb_labels = skimage.morphology.label(img_aug > 240, return_num=True, connectivity=2)
        assert nb_labels == 4
        _labels, nb_labels = skimage.morphology.label(hm_aug.arr_0to1 > 240/255, return_num=True, connectivity=2)
        assert nb_labels == 4

        # fit_output with differently sized heatmaps
        aug = iaa.Affine(rotate=45, fit_output=True, backend=backend)
        img = np.zeros((80, 80), dtype=np.uint8)
        img[0:5, 0:5] = 255
        img[-5:, 0:5] = 255
        img[0:5, -5:] = 255
        img[-5:, -5:] = 255
        hm = ia.HeatmapsOnImage(
            ia.imresize_single_image(img, (40, 40), interpolation="cubic").astype(np.float32)/255,
            shape=(80, 80)
        )
        img_aug = aug.augment_image(img)
        hm_aug = aug.augment_heatmaps([hm])[0]
        # these asserts are deactivated because the image size can change under fit_output=True
        # assert hm_aug.shape == (80, 80)
        # assert hm_aug.arr_0to1.shape == (40, 40, 1)
        _labels, nb_labels = skimage.morphology.label(img_aug > 240, return_num=True, connectivity=2)
        assert nb_labels == 4
        _labels, nb_labels = skimage.morphology.label(hm_aug.arr_0to1 > 200/255, return_num=True, connectivity=2)
        assert nb_labels == 4

        img_aug_mask = img_aug > 255*0.1
        hm_aug_mask = ia.imresize_single_image(hm_aug.arr_0to1, img_aug.shape[0:2], interpolation="cubic") > 0.1
        same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
        assert (same / img_aug_mask.size) >= 0.95

    # ------------
    # exceptions for bad inputs
    # ------------
    # scale
    got_exception = False
    try:
        _ = iaa.Affine(scale=False)
    except Exception:
        got_exception = True
    assert got_exception

    # translate_px
    got_exception = False
    try:
        _ = iaa.Affine(translate_px=False)
    except Exception:
        got_exception = True
    assert got_exception

    # translate_percent
    got_exception = False
    try:
        _ = iaa.Affine(translate_percent=False)
    except Exception:
        got_exception = True
    assert got_exception

    # rotate
    got_exception = False
    try:
        _ = iaa.Affine(scale=1.0, translate_px=0, rotate=False, shear=0, cval=0)
    except Exception:
        got_exception = True
    assert got_exception

    # shear
    got_exception = False
    try:
        _ = iaa.Affine(scale=1.0, translate_px=0, rotate=0, shear=False, cval=0)
    except Exception:
        got_exception = True
    assert got_exception

    # cval
    got_exception = False
    try:
        _ = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=None)
    except Exception:
        got_exception = True
    assert got_exception

    # mode
    got_exception = False
    try:
        _ = iaa.Affine(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=False)
    except Exception:
        got_exception = True
    assert got_exception

    # non-existent order in case of backend=cv2
    got_exception = False
    try:
        _ = iaa.Affine(backend="cv2", order=-1)
    except Exception:
        got_exception = True
    assert got_exception

    # bad order datatype in case of backend=cv2
    got_exception = False
    try:
        _ = iaa.Affine(backend="cv2", order="test")
    except Exception:
        got_exception = True
    assert got_exception

    # ----------
    # get_parameters
    # ----------
    aug = iaa.Affine(scale=1, translate_px=2, rotate=3, shear=4, order=1, cval=0, mode="constant", backend="cv2",
                     fit_output=True)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)  # scale
    assert isinstance(params[1], iap.Deterministic)  # translate
    assert isinstance(params[2], iap.Deterministic)  # rotate
    assert isinstance(params[3], iap.Deterministic)  # shear
    assert params[0].value == 1  # scale
    assert params[1].value == 2  # translate
    assert params[2].value == 3  # rotate
    assert params[3].value == 4  # shear
    assert params[4].value == 1  # order
    assert params[5].value == 0  # cval
    assert params[6].value == "constant"  # mode
    assert params[7] == "cv2"  # backend
    assert params[8] is True  # fit_output


def test_AffineCv2():
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

    # no translation/scale/rotate/shear, shouldnt change nothing
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug_det.augment_images(images)
    expected = images
    assert np.array_equal(observed, expected)

    observed = aug.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug_det.augment_images(images_list)
    expected = images_list
    assert array_equal_lists(observed, expected)

    observed = aug.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    observed = aug_det.augment_keypoints(keypoints)
    expected = keypoints
    assert keypoints_equal(observed, expected)

    # ---------------------
    # scale
    # ---------------------
    # zoom in
    aug = iaa.AffineCv2(scale=1.75, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][outer_pixels[0], outer_pixels[1]] > 20).all()
    assert (observed[0][outer_pixels[0], outer_pixels[1]] < 150).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y > 2

    # zoom in only on x axis
    aug = iaa.AffineCv2(scale={"x": 1.75, "y": 1.0}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[1, 1], [0, 2]] > 20).all()
    assert (observed[0][[1, 1], [0, 2]] < 150).all()
    assert (observed[0][0, :] < 5).all()
    assert (observed[0][2, :] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x < 0
    assert observed[0].keypoints[0].y == 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x > 2
    assert observed[0].keypoints[2].y == 2

    # zoom in only on y axis
    aug = iaa.AffineCv2(scale={"x": 1.0, "y": 1.75}, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    observed = aug.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug_det.augment_images(images_list)
    assert observed[0][1, 1] > 250
    assert (observed[0][[0, 2], [1, 1]] > 20).all()
    assert (observed[0][[0, 2], [1, 1]] < 150).all()
    assert (observed[0][:, 0] < 5).all()
    assert (observed[0][:, 2] < 5).all()

    observed = aug.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    observed = aug_det.augment_keypoints(keypoints)
    assert observed[0].keypoints[0].x == 0
    assert observed[0].keypoints[0].y < 0
    assert observed[0].keypoints[1].x == 1
    assert observed[0].keypoints[1].y == 1
    assert observed[0].keypoints[2].x == 2
    assert observed[0].keypoints[2].y > 2

    # zoom out
    # this one uses a 4x4 area of all 255, which is zoomed out to a 4x4 area
    # in which the center 2x2 area is 255
    # zoom in should probably be adapted to this style
    # no separate tests here for x/y axis, should work fine if zoom in works with that
    aug = iaa.AffineCv2(scale=0.49, translate_px=0, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.ones((4, 4, 1), dtype=np.uint8) * 255
    images = np.array([image])
    images_list = [image]
    outer_pixels = ([], [])
    for y in sm.xrange(4):
        xs = sm.xrange(4) if y in [0, 3] else [0, 3]
        for x in xs:
            outer_pixels[0].append(y)
            outer_pixels[1].append(x)
    inner_pixels = ([1, 1, 2, 2], [1, 2, 1, 2])
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=0),
                                      ia.Keypoint(x=0, y=3), ia.Keypoint(x=3, y=3)],
                                     shape=image.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=0.765, y=0.765), ia.Keypoint(x=2.235, y=0.765),
                                          ia.Keypoint(x=0.765, y=2.235), ia.Keypoint(x=2.235, y=2.235)],
                                         shape=image.shape)]

    observed = aug.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0][outer_pixels] < 25).all()
    assert (observed[0][inner_pixels] > 200).all()

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # varying scales
    aug = iaa.AffineCv2(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, translate_px=0,
                        rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.array([[0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 2, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0]], dtype=np.uint8) * 100
    image = image[:, :, np.newaxis]
    images = np.array([image])

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

    aug = iaa.AffineCv2(scale=iap.Uniform(0.7, 0.9))
    assert isinstance(aug.scale, iap.Uniform)
    assert isinstance(aug.scale.a, iap.Deterministic)
    assert isinstance(aug.scale.b, iap.Deterministic)
    assert 0.7 - 1e-8 < aug.scale.a.value < 0.7 + 1e-8
    assert 0.9 - 1e-8 < aug.scale.b.value < 0.9 + 1e-8

    # ---------------------
    # translate
    # ---------------------
    # move one pixel to the right
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move one pixel to the right
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0)
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with order=ALL
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, order=ia.ALL)
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with order=list
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, order=[0, 1, 2])
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the right
    # with order=StochasticParameter
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 1, "y": 0}, rotate=0, shear=0, order=iap.Choice([0, 1, 2]))
    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    # move one pixel to the bottom
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": 0, "y": 1}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the right
    aug = iaa.AffineCv2(scale=1.0, translate_percent={"x": 0.3333, "y": 0}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[1, 2] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=2, y=1)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # move 33% (one pixel) to the bottom
    aug = iaa.AffineCv2(scale=1.0, translate_percent={"x": 0, "y": 0.3333}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # 0-1px to left/right and 0-1px to top/bottom
    aug = iaa.AffineCv2(scale=1.0, translate_px={"x": (-1, 1), "y": (-1, 1)}, rotate=0, shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    centers_aug = np.copy(image).astype(np.int32) * 0
    centers_aug_det = np.copy(image).astype(np.int32) * 0
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

        assert len(observed_aug[0].nonzero()[0]) == 1
        assert len(observed_aug_det[0].nonzero()[0]) == 1
        centers_aug += (observed_aug[0] > 0)
        centers_aug_det += (observed_aug_det[0] > 0)

    assert nb_changed_aug >= int(nb_iterations * 0.7)
    assert nb_changed_aug_det == 0
    assert (centers_aug > int(nb_iterations * (1/9 * 0.6))).all()
    assert (centers_aug < int(nb_iterations * (1/9 * 1.4))).all()

    aug = iaa.AffineCv2(translate_percent=iap.Uniform(0.7, 0.9))
    assert isinstance(aug.translate, iap.Uniform)
    assert isinstance(aug.translate.a, iap.Deterministic)
    assert isinstance(aug.translate.b, iap.Deterministic)
    assert 0.7 - 1e-8 < aug.translate.a.value < 0.7 + 1e-8
    assert 0.9 - 1e-8 < aug.translate.b.value < 0.9 + 1e-8

    aug = iaa.AffineCv2(translate_px=iap.DiscreteUniform(1, 10))
    assert isinstance(aug.translate, iap.DiscreteUniform)
    assert isinstance(aug.translate.a, iap.Deterministic)
    assert isinstance(aug.translate.b, iap.Deterministic)
    assert aug.translate.a.value == 1
    assert aug.translate.b.value == 10

    # ---------------------
    # translate heatmaps
    # ---------------------
    heatmaps = ia.HeatmapsOnImage(
        np.float32([
            [0.0, 0.5, 0.75],
            [0.0, 0.5, 0.75],
            [0.75, 0.75, 0.75],
        ]),
        shape=(3, 3, 3)
    )
    arr_expected_1px_right = np.float32([
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.75, 0.75],
    ])
    aug = iaa.AffineCv2(translate_px={"x": 1})
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    # should still use mode=constant cval=0 even when other settings chosen
    aug = iaa.AffineCv2(translate_px={"x": 1}, cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    aug = iaa.AffineCv2(translate_px={"x": 1}, mode="replicate", cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed.get_arr(), arr_expected_1px_right)

    # ---------------------
    # rotate
    # ---------------------
    # rotate by 45 degrees
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=90, shear=0)
    aug_det = aug.to_deterministic()

    image = np.zeros((3, 3, 1), dtype=np.uint8)
    image_aug = np.copy(image)
    image[1, :] = 255
    image_aug[0, 1] = 255
    image_aug[1, 1] = 255
    image_aug[2, 1] = 255
    images = np.array([image])
    images_aug = np.array([image_aug])
    images_list = [image]
    images_aug_list = [image_aug]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=1)], shape=base_img.shape)]
    keypoints_aug = [ia.KeypointsOnImage([ia.Keypoint(x=1, y=0), ia.Keypoint(x=1, y=1),
                                          ia.Keypoint(x=1, y=2)], shape=base_img.shape)]

    observed = aug.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug_det.augment_images(images)
    observed[observed >= 100] = 255
    observed[observed < 100] = 0
    assert np.array_equal(observed, images_aug)

    observed = aug.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug_det.augment_images(images_list)
    observed[0][observed[0] >= 100] = 255
    observed[0][observed[0] < 100] = 0
    assert array_equal_lists(observed, images_aug_list)

    observed = aug.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    observed = aug_det.augment_keypoints(keypoints)
    assert keypoints_equal(observed, keypoints_aug)

    # rotate by StochasticParameter
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=iap.Uniform(10, 20), shear=0)
    assert isinstance(aug.rotate, iap.Uniform)
    assert isinstance(aug.rotate.a, iap.Deterministic)
    assert aug.rotate.a.value == 10
    assert isinstance(aug.rotate.b, iap.Deterministic)
    assert aug.rotate.b.value == 20

    # random rotation 0-364 degrees
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=(0, 364), shear=0)
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    pixels_sums_aug = np.copy(image).astype(np.int32) * 0
    pixels_sums_aug_det = np.copy(image).astype(np.int32) * 0
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

        pixels_sums_aug += (observed_aug[0] > 100)
        pixels_sums_aug_det += (observed_aug_det[0] > 100)

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)

    # outer pixels, should sometimes be white
    # the values here had to be set quite tolerant, the middle pixels at top/left/bottom/right get more activation
    # than expected
    outer_pixels = ([0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2])
    assert (pixels_sums_aug[outer_pixels] > int(nb_iterations * (2/8 * 0.4))).all()
    assert (pixels_sums_aug[outer_pixels] < int(nb_iterations * (2/8 * 2.0))).all()

    # ---------------------
    # shear
    # ---------------------
    # TODO

    # shear by StochasticParameter
    aug = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0, shear=iap.Uniform(10, 20))
    assert isinstance(aug.shear, iap.Uniform)
    assert isinstance(aug.shear.a, iap.Deterministic)
    assert aug.shear.a.value == 10
    assert isinstance(aug.shear.b, iap.Deterministic)
    assert aug.shear.b.value == 20

    # ---------------------
    # cval
    # ---------------------
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=128)
    aug_det = aug.to_deterministic()

    image = np.ones((3, 3, 1), dtype=np.uint8) * 255
    image_aug = np.copy(image)
    images = np.array([image])
    images_list = [image]

    observed = aug.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    observed = aug_det.augment_images(images_list)
    assert (observed[0] > 128 - 30).all()
    assert (observed[0] < 128 + 30).all()

    # random cvals
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=(0, 255))
    aug_det = aug.to_deterministic()
    last_aug = None
    last_aug_det = None
    nb_changed_aug = 0
    nb_changed_aug_det = 0
    nb_iterations = 1000
    averages = []
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

        averages.append(int(np.average(observed_aug)))

    assert nb_changed_aug >= int(nb_iterations * 0.9)
    assert nb_changed_aug_det == 0
    # center pixel, should always be white when rotating line around center
    assert pixels_sums_aug[1, 1] > (nb_iterations * 0.98)
    assert pixels_sums_aug[1, 1] < (nb_iterations * 1.02)
    assert len(set(averages)) > 200

    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=ia.ALL)
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 255

    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=iap.DiscreteUniform(1, 5))
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 1
    assert aug.cval.b.value == 5

    # ------------
    # mode
    # ------------
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=ia.ALL)
    assert isinstance(aug.mode, iap.Choice)
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode="replicate")
    assert isinstance(aug.mode, iap.Deterministic)
    assert aug.mode.value == "replicate"
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=["replicate", "reflect"])
    assert isinstance(aug.mode, iap.Choice)
    assert len(aug.mode.a) == 2 and "replicate" in aug.mode.a and "reflect" in aug.mode.a
    aug = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0,
                        mode=iap.Choice(["replicate", "reflect"]))
    assert isinstance(aug.mode, iap.Choice)
    assert len(aug.mode.a) == 2 and "replicate" in aug.mode.a and "reflect" in aug.mode.a

    # ------------
    # exceptions for bad inputs
    # ------------
    # scale
    got_exception = False
    try:
        _ = iaa.AffineCv2(scale=False)
    except Exception:
        got_exception = True
    assert got_exception

    # translate_px
    got_exception = False
    try:
        _ = iaa.AffineCv2(translate_px=False)
    except Exception:
        got_exception = True
    assert got_exception

    # translate_percent
    got_exception = False
    try:
        _ = iaa.AffineCv2(translate_percent=False)
    except Exception:
        got_exception = True
    assert got_exception

    # rotate
    got_exception = False
    try:
        _ = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=False, shear=0, cval=0)
    except Exception:
        got_exception = True
    assert got_exception

    # shear
    got_exception = False
    try:
        _ = iaa.AffineCv2(scale=1.0, translate_px=0, rotate=0, shear=False, cval=0)
    except Exception:
        got_exception = True
    assert got_exception

    # cval
    got_exception = False
    try:
        _ = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=None)
    except Exception:
        got_exception = True
    assert got_exception

    # mode
    got_exception = False
    try:
        _ = iaa.AffineCv2(scale=1.0, translate_px=100, rotate=0, shear=0, cval=0, mode=False)
    except Exception:
        got_exception = True
    assert got_exception

    # non-existent order
    got_exception = False
    try:
        _ = iaa.AffineCv2(order=-1)
    except Exception:
        got_exception = True
    assert got_exception

    # bad order datatype
    got_exception = False
    try:
        _ = iaa.AffineCv2(order="test")
    except Exception:
        got_exception = True
    assert got_exception

    # ----------
    # get_parameters
    # ----------
    aug = iaa.AffineCv2(scale=1, translate_px=2, rotate=3, shear=4, order=1, cval=0, mode="constant")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)  # scale
    assert isinstance(params[1], iap.Deterministic)  # translate
    assert isinstance(params[2], iap.Deterministic)  # rotate
    assert isinstance(params[3], iap.Deterministic)  # shear
    assert params[0].value == 1  # scale
    assert params[1].value == 2  # translate
    assert params[2].value == 3  # rotate
    assert params[3].value == 4  # shear
    assert params[4].value == 1  # order
    assert params[5].value == 0  # cval
    assert params[6].value == "constant"  # mode


def test_PiecewiseAffine():
    reseed()

    img = np.zeros((60, 80), dtype=np.uint8)
    img[:, 9:11+1] = 255
    img[:, 69:71+1] = 255
    mask = img > 0
    heatmaps = ia.HeatmapsOnImage((img / 255.0).astype(np.float32), shape=(60, 80, 3))
    heatmaps_arr = heatmaps.get_arr()

    # -----
    # scale
    # -----
    # basic test
    aug = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    observed = aug.augment_image(img)
    assert 100.0 < np.average(observed[mask]) < np.average(img[mask])
    assert 75.0 > np.average(observed[~mask]) > np.average(img[~mask])

    # basic test, heatmaps
    aug = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    observed = aug.augment_heatmaps([heatmaps])[0]
    observed_arr = observed.get_arr()
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert 100.0/255.0 < np.average(observed_arr[mask]) < np.average(heatmaps_arr[mask])
    assert 75.0/255.0 > np.average(observed_arr[~mask]) > np.average(heatmaps_arr[~mask])

    # scale 0
    aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    # scale 0, heatmaps
    aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)
    observed = aug.augment_heatmaps([heatmaps])[0]
    observed_arr = observed.get_arr()
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.array_equal(observed_arr, heatmaps_arr)

    # scale 0, keypoints
    aug = iaa.PiecewiseAffine(scale=0, nb_rows=12, nb_cols=4)
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=5, y=3), ia.Keypoint(x=3, y=8)], shape=(14, 14, 3))
    kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (14, 14, 3)
    assert np.allclose(kpsoi_aug.keypoints[0].x, 5)
    assert np.allclose(kpsoi_aug.keypoints[0].y, 3)
    assert np.allclose(kpsoi_aug.keypoints[1].x, 3)
    assert np.allclose(kpsoi_aug.keypoints[1].y, 8)

    # stronger scale should lead to stronger changes
    aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
    observed1 = aug1.augment_image(img)
    observed2 = aug2.augment_image(img)
    assert np.average(observed1[~mask]) < np.average(observed2[~mask])

    # stronger scale should lead to stronger changes, heatmaps
    aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
    observed1 = aug1.augment_heatmaps([heatmaps])[0]
    observed1_arr = observed1.get_arr()
    observed2 = aug2.augment_heatmaps([heatmaps])[0]
    observed2_arr = observed2.get_arr()
    assert observed1.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed1.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed1.max_value < heatmaps.max_value + 1e-6
    assert observed2.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed2.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed2.max_value < heatmaps.max_value + 1e-6
    assert np.average(observed1_arr[~mask]) < np.average(observed2_arr[~mask])

    # strong scale, measure alignment between images and heatmaps
    aug = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
    aug_det = aug.to_deterministic()
    img_aug = aug_det.augment_image(img)
    hm_aug = aug_det.augment_heatmaps([heatmaps])[0]
    assert hm_aug.shape == (60, 80, 3)
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    img_aug_mask = img_aug > 255*0.1
    hm_aug_mask = hm_aug.arr_0to1 > 0.1
    same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
    assert (same / img_aug_mask.size) >= 0.98

    # strong scale, measure alignment between images and heatmaps
    # heatmaps here smaller than image
    aug_det = aug.to_deterministic()
    heatmaps_small = ia.HeatmapsOnImage(
        (ia.imresize_single_image(img, (30, 40+10), interpolation="cubic") / 255.0).astype(np.float32),
        shape=(60, 80, 3)
    )
    img_aug = aug_det.augment_image(img)
    hm_aug = aug_det.augment_heatmaps([heatmaps_small])[0]
    assert hm_aug.shape == (60, 80, 3)
    assert hm_aug.arr_0to1.shape == (30, 40+10, 1)
    img_aug_mask = img_aug > 255*0.1
    hm_aug_mask = ia.imresize_single_image(hm_aug.arr_0to1, (60, 80), interpolation="cubic") > 0.1
    same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
    assert (same / img_aug_mask.size) >= 0.9  # seems to be 0.948 actually

    # strong scale, measure alignment between images and keypoints
    aug = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
    aug_det = aug.to_deterministic()
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=5, y=15), ia.Keypoint(x=17, y=12)], shape=(24, 30, 3))
    img_kps = np.zeros((24, 30, 3), dtype=np.uint8)
    img_kps = kpsoi.draw_on_image(img_kps, color=[255, 255, 255])
    img_kps_aug = aug_det.augment_image(img_kps)
    kpsoi_aug = aug_det.augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (24, 30, 3)
    bb1 = ia.BoundingBox(x1=kpsoi_aug.keypoints[0].x-1, y1=kpsoi_aug.keypoints[0].y-1,
                         x2=kpsoi_aug.keypoints[0].x+1, y2=kpsoi_aug.keypoints[0].y+1)
    bb2 = ia.BoundingBox(x1=kpsoi_aug.keypoints[1].x-1, y1=kpsoi_aug.keypoints[1].y-1,
                         x2=kpsoi_aug.keypoints[1].x+1, y2=kpsoi_aug.keypoints[1].y+1)
    patch1 = bb1.extract_from_image(img_kps_aug)
    patch2 = bb2.extract_from_image(img_kps_aug)
    assert np.max(patch1) > 150
    assert np.max(patch2) > 150
    assert np.average(img_kps_aug) < 40

    # scale as list
    aug1 = iaa.PiecewiseAffine(scale=0.01, nb_rows=12, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.10, nb_rows=12, nb_cols=4)
    aug = iaa.PiecewiseAffine(scale=[0.01, 0.10], nb_rows=12, nb_cols=4)
    assert isinstance(aug.scale, iap.Choice)
    assert 0.01 - 1e-8 < aug.scale.a[0] < 0.01 + 1e-8
    assert 0.10 - 1e-8 < aug.scale.a[1] < 0.10 + 1e-8

    avg1 = np.average([np.average(aug1.augment_image(img) * (~mask).astype(np.float32)) for _ in sm.xrange(3)])
    avg2 = np.average([np.average(aug2.augment_image(img) * (~mask).astype(np.float32)) for _ in sm.xrange(3)])
    seen = [0, 0]
    for _ in sm.xrange(15):
        observed = aug.augment_image(img)
        avg = np.average(observed * (~mask).astype(np.float32))
        diff1 = abs(avg - avg1)
        diff2 = abs(avg - avg2)
        if diff1 < diff2:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 0
    assert seen[1] > 0

    # scale as tuple
    aug = iaa.PiecewiseAffine(scale=(0.01, 0.10), nb_rows=12, nb_cols=4)
    assert isinstance(aug.jitter.scale, iap.Uniform)
    assert isinstance(aug.jitter.scale.a, iap.Deterministic)
    assert isinstance(aug.jitter.scale.b, iap.Deterministic)
    assert 0.01 - 1e-8 < aug.jitter.scale.a.value < 0.01 + 1e-8
    assert 0.10 - 1e-8 < aug.jitter.scale.b.value < 0.10 + 1e-8

    # scale as StochasticParameter
    aug = iaa.PiecewiseAffine(scale=iap.Uniform(0.01, 0.10), nb_rows=12, nb_cols=4)
    assert isinstance(aug.jitter.scale, iap.Uniform)
    assert isinstance(aug.jitter.scale.a, iap.Deterministic)
    assert isinstance(aug.jitter.scale.b, iap.Deterministic)
    assert 0.01 - 1e-8 < aug.jitter.scale.a.value < 0.01 + 1e-8
    assert 0.10 - 1e-8 < aug.jitter.scale.b.value < 0.10 + 1e-8

    # bad datatype for scale
    got_exception = False
    try:
        _ = iaa.PiecewiseAffine(scale=False, nb_rows=12, nb_cols=4)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # rows and cols
    # -----
    # verify effects of rows/cols
    aug1 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.05, nb_rows=30, nb_cols=4)
    std1 = []
    std2 = []
    for _ in sm.xrange(3):
        observed1 = aug1.augment_image(img)
        observed2 = aug2.augment_image(img)
        grad_vert1 = observed1[1:, :].astype(np.float32) - observed1[:-1, :].astype(np.float32)
        grad_vert2 = observed2[1:, :].astype(np.float32) - observed2[:-1, :].astype(np.float32)
        grad_vert1 = grad_vert1 * (~mask[1:, :]).astype(np.float32)
        grad_vert2 = grad_vert2 * (~mask[1:, :]).astype(np.float32)
        std1.append(np.std(grad_vert1))
        std2.append(np.std(grad_vert2))
    std1 = np.average(std1)
    std2 = np.average(std2)
    assert std1 < std2

    # -----
    # rows
    # -----
    # rows as list
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=[4, 20], nb_cols=4)
    assert isinstance(aug.nb_rows, iap.Choice)
    assert aug.nb_rows.a[0] == 4
    assert aug.nb_rows.a[1] == 20

    seen = [0, 0]
    for _ in sm.xrange(20):
        observed = aug.augment_image(img)
        grad_vert = observed[1:, :].astype(np.float32) - observed[:-1, :].astype(np.float32)
        grad_vert = grad_vert * (~mask[1:, :]).astype(np.float32)
        std = np.std(grad_vert)
        diff1 = abs(std - std1)
        diff2 = abs(std - std2)
        if diff1 < diff2:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 0
    assert seen[1] > 0

    # rows as tuple
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=(4, 20), nb_cols=4)
    assert isinstance(aug.nb_rows, iap.DiscreteUniform)
    assert isinstance(aug.nb_rows.a, iap.Deterministic)
    assert isinstance(aug.nb_rows.b, iap.Deterministic)
    assert aug.nb_rows.a.value == 4
    assert aug.nb_rows.b.value == 20

    # rows as StochasticParameter
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=iap.DiscreteUniform(4, 20), nb_cols=4)
    assert isinstance(aug.nb_rows, iap.DiscreteUniform)
    assert isinstance(aug.nb_rows.a, iap.Deterministic)
    assert isinstance(aug.nb_rows.b, iap.Deterministic)
    assert aug.nb_rows.a.value == 4
    assert aug.nb_rows.b.value == 20

    # bad datatype for rows
    got_exception = False
    try:
        _ = iaa.PiecewiseAffine(scale=0.05, nb_rows=False, nb_cols=4)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # nb_cols
    # -----

    # cols as list
    img_cols = img.T
    mask_cols = mask.T
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=[4, 20])
    assert isinstance(aug.nb_cols, iap.Choice)
    assert aug.nb_cols.a[0] == 4
    assert aug.nb_cols.a[1] == 20

    aug1 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=4)
    aug2 = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=20)

    std1 = []
    std2 = []
    for _ in sm.xrange(3):
        observed1 = aug1.augment_image(img_cols)
        observed2 = aug2.augment_image(img_cols)
        grad_hori1 = observed1[:, 1:].astype(np.float32) - observed1[:, :-1].astype(np.float32)
        grad_hori2 = observed2[:, 1:].astype(np.float32) - observed2[:, :-1].astype(np.float32)
        grad_hori1 = grad_hori1 * (~mask_cols[:, 1:]).astype(np.float32)
        grad_hori2 = grad_hori2 * (~mask_cols[:, 1:]).astype(np.float32)
        std1.append(np.std(grad_hori1))
        std2.append(np.std(grad_hori2))
    std1 = np.average(std1)
    std2 = np.average(std2)

    seen = [0, 0]
    for _ in sm.xrange(15):
        observed = aug.augment_image(img_cols)

        grad_hori = observed[:, 1:].astype(np.float32) - observed[:, :-1].astype(np.float32)
        grad_hori = grad_hori * (~mask_cols[:, 1:]).astype(np.float32)
        std = np.std(grad_hori)
        diff1 = abs(std - std1)
        diff2 = abs(std - std2)

        if diff1 < diff2:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 0
    assert seen[1] > 0

    # cols as tuple
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=(4, 20))
    assert isinstance(aug.nb_cols, iap.DiscreteUniform)
    assert isinstance(aug.nb_cols.a, iap.Deterministic)
    assert isinstance(aug.nb_cols.b, iap.Deterministic)
    assert aug.nb_cols.a.value == 4
    assert aug.nb_cols.b.value == 20

    # cols as StochasticParameter
    aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=iap.DiscreteUniform(4, 20))
    assert isinstance(aug.nb_cols, iap.DiscreteUniform)
    assert isinstance(aug.nb_cols.a, iap.Deterministic)
    assert isinstance(aug.nb_cols.b, iap.Deterministic)
    assert aug.nb_cols.a.value == 4
    assert aug.nb_cols.b.value == 20

    # bad datatype for cols
    got_exception = False
    try:
        aug = iaa.PiecewiseAffine(scale=0.05, nb_rows=4, nb_cols=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # order
    # -----
    # single int for order
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=0)
    assert isinstance(aug.order, iap.Deterministic)
    assert aug.order.value == 0

    # list for order
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=[0, 1, 3])
    assert isinstance(aug.order, iap.Choice)
    assert all([v in aug.order.a for v in [0, 1, 3]])

    # StochasticParameter for order
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=iap.Choice([0, 1, 3]))
    assert isinstance(aug.order, iap.Choice)
    assert all([v in aug.order.a for v in [0, 1, 3]])

    # ALL for order
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=ia.ALL)
    assert isinstance(aug.order, iap.Choice)
    assert all([v in aug.order.a for v in [0, 1, 3, 4, 5]])

    # bad datatype for order
    got_exception = False
    try:
        _ = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, order=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # cval
    # -----
    # cval as deterministic
    img = np.zeros((50, 50, 3), dtype=np.uint8) + 255
    aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=10, nb_cols=10, mode="constant", cval=0)
    observed = aug.augment_image(img)
    assert np.sum([observed[:, :] == [0, 0, 0]]) > 0

    # cval as deterministic, heatmaps should always use cval=0
    heatmaps = ia.HeatmapsOnImage(np.zeros((50, 50, 1), dtype=np.float32), shape=(50, 50, 3))
    aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=10, nb_cols=10, mode="constant", cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert np.sum([observed.get_arr()[:, :] >= 0.01]) == 0

    # cval as list
    img = np.zeros((20, 20), dtype=np.uint8) + 255
    aug = iaa.PiecewiseAffine(scale=0.7, nb_rows=5, nb_cols=5, mode="constant", cval=[0, 10])
    assert isinstance(aug.cval, iap.Choice)
    assert aug.cval.a[0] == 0
    assert aug.cval.a[1] == 10

    seen = [0, 0, 0]
    for _ in sm.xrange(30):
        observed = aug.augment_image(img)
        nb_0 = np.sum([observed[:, :] == 0])
        nb_10 = np.sum([observed[:, :] == 10])
        if nb_0 > 0:
            seen[0] += 1
        elif nb_10 > 0:
            seen[1] += 1
        else:
            seen[2] += 1
    assert seen[0] > 5
    assert seen[1] > 5
    assert seen[2] <= 4

    # cval as tuple
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode="constant", cval=(0, 10))
    assert isinstance(aug.cval, iap.Uniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 10

    # cval as StochasticParameter
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode="constant", cval=iap.DiscreteUniform(0, 10))
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 10

    # ALL as cval
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode="constant", cval=ia.ALL)
    assert isinstance(aug.cval, iap.Uniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 255

    # bas datatype for cval
    got_exception = False
    try:
        _ = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, cval=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # -----
    # mode
    # -----
    # single string for mode
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode="nearest")
    assert isinstance(aug.mode, iap.Deterministic)
    assert aug.mode.value == "nearest"

    # list for mode
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=["nearest", "edge", "symmetric"])
    assert isinstance(aug.mode, iap.Choice)
    assert all([v in aug.mode.a for v in ["nearest", "edge", "symmetric"]])

    # StochasticParameter for mode
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=iap.Choice(["nearest", "edge", "symmetric"]))
    assert isinstance(aug.mode, iap.Choice)
    assert all([v in aug.mode.a for v in ["nearest", "edge", "symmetric"]])

    # ALL for mode
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=ia.ALL)
    assert isinstance(aug.mode, iap.Choice)
    assert all([v in aug.mode.a for v in ["constant", "edge", "symmetric", "reflect", "wrap"]])

    # bad datatype for mode
    got_exception = False
    try:
        _ = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=8, mode=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # ---------
    # keypoints
    # ---------
    # basic test
    img = np.zeros((100, 80), dtype=np.uint8)
    img[:, 9:11+1] = 255
    img[:, 69:71+1] = 255
    mask = img > 0
    kps = [ia.Keypoint(x=10, y=20), ia.Keypoint(x=10, y=40),
           ia.Keypoint(x=70, y=20), ia.Keypoint(x=70, y=40)]
    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)

    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
    aug_det = aug.to_deterministic()
    observed_img = aug_det.augment_image(img)
    observed_kpsoi = aug_det.augment_keypoints([kpsoi])
    assert not keypoints_equal([kpsoi], observed_kpsoi)
    for kp in observed_kpsoi[0].keypoints:
        assert observed_img[int(kp.y), int(kp.x)] > 0

    # scale 0
    aug = iaa.PiecewiseAffine(scale=0, nb_rows=10, nb_cols=10)
    observed = aug.augment_keypoints([kpsoi])
    assert keypoints_equal([kpsoi], observed)

    # keypoints outside of image
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=10, nb_cols=10)
    kps = [ia.Keypoint(x=-10, y=-20)]
    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)
    observed = aug.augment_keypoints([kpsoi])
    assert keypoints_equal([kpsoi], observed)

    # ---------
    # get_parameters
    # ---------
    aug = iaa.PiecewiseAffine(scale=0.1, nb_rows=8, nb_cols=10, order=1, cval=2, mode="nearest", absolute_scale=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert isinstance(params[2], iap.Deterministic)
    assert isinstance(params[3], iap.Deterministic)
    assert isinstance(params[4], iap.Deterministic)
    assert isinstance(params[5], iap.Deterministic)
    assert params[6] is False
    assert 0.1 - 1e-8 < params[0].value < 0.1 + 1e-8
    assert params[1].value == 8
    assert params[2].value == 10
    assert params[3].value == 1
    assert params[4].value == 2
    assert params[5].value == "nearest"


def test_PerspectiveTransform():
    reseed()

    img = np.zeros((30, 30), dtype=np.uint8)
    img[10:20, 10:20] = 255
    heatmaps = ia.HeatmapsOnImage((img / 255.0).astype(np.float32), shape=img.shape)

    # without keep_size
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_image(img)
    y1 = int(30*0.2)
    y2 = int(30*0.8)
    x1 = int(30*0.2)
    x2 = int(30*0.8)
    expected = img[y1:y2, x1:x2]
    assert all([abs(s1-s2) <= 1 for s1, s2 in zip(observed.shape, expected.shape)])
    if observed.shape != expected.shape:
        observed = ia.imresize_single_image(observed, expected.shape[0:2], interpolation="cubic")
    # differences seem to mainly appear around the border of the inner rectangle, possibly
    # due to interpolation
    assert np.average(np.abs(observed.astype(np.int32) - expected.astype(np.int32))) < 30.0

    hm = ia.HeatmapsOnImage(img.astype(np.float32)/255.0, shape=(30, 30))
    hm_aug = aug.augment_heatmaps([hm])[0]
    expected = (y2 - y1, x2 - x1)
    assert all([abs(s1-s2) <= 1 for s1, s2 in zip(hm_aug.shape, expected)])
    assert all([abs(s1-s2) <= 1 for s1, s2 in zip(hm_aug.arr_0to1.shape, expected + (1,))])
    img_aug_mask = observed > 255*0.1
    hm_aug_mask = hm_aug.arr_0to1 > 0.1
    same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
    assert (same / img_aug_mask.size) >= 0.99

    # without keep_size, different heatmap size
    img_small = ia.imresize_single_image(img, (20, 25), interpolation="cubic")
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
    aug.jitter = iap.Deterministic(0.2)
    img_aug = aug.augment_image(img)
    y1 = int(30*0.2)
    y2 = int(30*0.8)
    x1 = int(30*0.2)
    x2 = int(30*0.8)
    x1_small = int(25*0.2)
    x2_small = int(25*0.8)
    y1_small = int(20*0.2)
    y2_small = int(20*0.8)
    hm = ia.HeatmapsOnImage(img_small.astype(np.float32)/255.0, shape=(30, 30))
    hm_aug = aug.augment_heatmaps([hm])[0]
    expected = (y2 - y1, x2 - x1)
    expected_small = (y2_small - y1_small, x2_small - x1_small, 1)
    assert all([abs(s1-s2) <= 1 for s1, s2 in zip(hm_aug.shape, expected)])
    assert all([abs(s1-s2) <= 1 for s1, s2 in zip(hm_aug.arr_0to1.shape, expected_small)])
    img_aug_mask = img_aug > 255*0.1
    hm_aug_mask = ia.imresize_single_image(hm_aug.arr_0to1, img_aug.shape[0:2], interpolation="cubic") > 0.1
    same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
    assert (same / img_aug_mask.size) >= 0.96

    # with keep_size
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_image(img)
    expected = img[int(30*0.2):int(30*0.8), int(30*0.2):int(30*0.8)]
    expected = ia.imresize_single_image(expected, img.shape[0:2], interpolation="cubic")
    assert observed.shape == img.shape
    # differences seem to mainly appear around the border of the inner rectangle, possibly
    # due to interpolation
    assert np.average(np.abs(observed.astype(np.int32) - expected.astype(np.int32))) < 30.0

    # with keep_size, heatmaps
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = heatmaps.get_arr()[int(30*0.2):int(30*0.8), int(30*0.2):int(30*0.8)]
    expected = ia.imresize_single_image((expected*255).astype(np.uint8), img.shape[0:2], interpolation="cubic")
    expected = (expected / 255.0).astype(np.float32)
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    # differences seem to mainly appear around the border of the inner rectangle, possibly
    # due to interpolation
    assert np.average(np.abs(observed.get_arr() - expected)) < 30.0

    # with keep_size, RGB images
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
    aug.jitter = iap.Deterministic(0.2)
    imgs = np.tile(img[np.newaxis, :, :, np.newaxis], (2, 1, 1, 3))
    observed = aug.augment_images(imgs)
    for img_idx in sm.xrange(2):
        for c in sm.xrange(3):
            observed_i = observed[img_idx, :, :, c]
            expected = imgs[img_idx, int(30*0.2):int(30*0.8), int(30*0.2):int(30*0.8), c]
            expected = ia.imresize_single_image(expected, imgs.shape[1:3], interpolation="cubic")
            assert observed_i.shape == imgs.shape[1:3]
            # differences seem to mainly appear around the border of the inner rectangle, possibly
            # due to interpolation
            assert np.average(np.abs(observed_i.astype(np.int32) - expected.astype(np.int32))) < 30.0

    # tuple for scale
    aug = iaa.PerspectiveTransform(scale=(0.1, 0.2))
    assert isinstance(aug.jitter.scale, iap.Uniform)
    assert isinstance(aug.jitter.scale.a, iap.Deterministic)
    assert isinstance(aug.jitter.scale.b, iap.Deterministic)
    assert 0.1 - 1e-8 < aug.jitter.scale.a.value < 0.1 + 1e-8
    assert 0.2 - 1e-8 < aug.jitter.scale.b.value < 0.2 + 1e-8

    # list for scale
    aug = iaa.PerspectiveTransform(scale=[0.1, 0.2, 0.3])
    assert isinstance(aug.jitter.scale, iap.Choice)
    assert len(aug.jitter.scale.a) == 3
    assert 0.1 - 1e-8 < aug.jitter.scale.a[0] < 0.1 + 1e-8
    assert 0.2 - 1e-8 < aug.jitter.scale.a[1] < 0.2 + 1e-8
    assert 0.3 - 1e-8 < aug.jitter.scale.a[2] < 0.3 + 1e-8

    # StochasticParameter for scale
    aug = iaa.PerspectiveTransform(scale=iap.Choice([0.1, 0.2, 0.3]))
    assert isinstance(aug.jitter.scale, iap.Choice)
    assert len(aug.jitter.scale.a) == 3
    assert 0.1 - 1e-8 < aug.jitter.scale.a[0] < 0.1 + 1e-8
    assert 0.2 - 1e-8 < aug.jitter.scale.a[1] < 0.2 + 1e-8
    assert 0.3 - 1e-8 < aug.jitter.scale.a[2] < 0.3 + 1e-8

    # bad datatype for scale
    got_exception = False
    try:
        _ = iaa.PerspectiveTransform(scale=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # keypoint augmentation without keep_size
    # TODO deviations of around 0.4-0.7 in this and the next test (between expected and observed
    # coordinates) -- why?
    kps = [ia.Keypoint(x=10, y=10), ia.Keypoint(x=14, y=11)]
    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=False)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_keypoints([kpsoi])
    kps_expected = [
        ia.Keypoint(x=10-0.2*30, y=10-0.2*30),
        ia.Keypoint(x=14-0.2*30, y=11-0.2*30)
    ]
    for kp_observed, kp_expected in zip(observed[0].keypoints, kps_expected):
        assert kp_expected.x - 1.5 < kp_observed.x < kp_expected.x + 1.5
        assert kp_expected.y - 1.5 < kp_observed.y < kp_expected.y + 1.5

    # keypoint augmentation with keep_size
    kps = [ia.Keypoint(x=10, y=10), ia.Keypoint(x=14, y=11)]
    kpsoi = ia.KeypointsOnImage(kps, shape=img.shape)
    aug = iaa.PerspectiveTransform(scale=0.2, keep_size=True)
    aug.jitter = iap.Deterministic(0.2)
    observed = aug.augment_keypoints([kpsoi])
    kps_expected = [
        ia.Keypoint(x=((10-0.2*30)/(30*0.6))*30, y=((10-0.2*30)/(30*0.6))*30),
        ia.Keypoint(x=((14-0.2*30)/(30*0.6))*30, y=((11-0.2*30)/(30*0.6))*30)
    ]
    for kp_observed, kp_expected in zip(observed[0].keypoints, kps_expected):
        assert kp_expected.x - 1.5 < kp_observed.x < kp_expected.x + 1.5
        assert kp_expected.y - 1.5 < kp_observed.y < kp_expected.y + 1.5

    # get_parameters
    aug = iaa.PerspectiveTransform(scale=0.1, keep_size=False)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Normal)
    assert isinstance(params[0].scale, iap.Deterministic)
    assert 0.1 - 1e-8 < params[0].scale.value < 0.1 + 1e-8
    assert params[1] is False


def test_ElasticTransformation():
    reseed()

    img = np.zeros((50, 50), dtype=np.uint8) + 255
    img = np.pad(img, ((100, 100), (100, 100)), mode="constant", constant_values=0)
    mask = img > 0
    heatmaps = ia.HeatmapsOnImage((img / 255.0).astype(np.float32), shape=img.shape)

    img_nonsquare = np.zeros((50, 100), dtype=np.uint8) + 255
    img_nonsquare = np.pad(img_nonsquare, ((100, 100), (100, 100)), mode="constant", constant_values=0)
    mask_nonsquare = img_nonsquare > 0

    # test basic funtionality
    aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
    observed = aug.augment_image(img)
    # assume that some white/255 pixels have been moved away from the center and replaced by black/0 pixels
    assert np.sum(observed[mask]) < np.sum(img[mask])
    # assume that some black/0 pixels have been moved away from the outer area and replaced by white/255 pixels
    assert np.sum(observed[~mask]) > np.sum(img[~mask])

    # test basic funtionality with non-square images
    aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
    observed = aug.augment_image(img_nonsquare)
    assert np.sum(observed[mask_nonsquare]) < np.sum(img_nonsquare[mask_nonsquare])
    assert np.sum(observed[~mask_nonsquare]) > np.sum(img_nonsquare[~mask_nonsquare])

    # test basic funtionality, heatmaps
    aug = iaa.ElasticTransformation(alpha=0.5, sigma=0.25)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.sum(observed.get_arr()[mask]) < np.sum(heatmaps.get_arr()[mask])
    assert np.sum(observed.get_arr()[~mask]) > np.sum(heatmaps.get_arr()[~mask])

    # test effects of increased alpha strength
    aug1 = iaa.ElasticTransformation(alpha=0.1, sigma=0.25)
    aug2 = iaa.ElasticTransformation(alpha=5.0, sigma=0.25)
    observed1 = aug1.augment_image(img)
    observed2 = aug2.augment_image(img)
    # assume that the inner area has become more black-ish when using high alphas (more white pixels were moved out of
    # the inner area)
    assert np.sum(observed1[mask]) > np.sum(observed2[mask])
    # assume that the outer area has become more white-ish when using high alphas (more black pixels were moved into
    # the inner area)
    assert np.sum(observed1[~mask]) < np.sum(observed2[~mask])

    # test effects of increased alpha strength, heatmaps
    aug1 = iaa.ElasticTransformation(alpha=0.1, sigma=0.25)
    aug2 = iaa.ElasticTransformation(alpha=5.0, sigma=0.25)
    observed1 = aug1.augment_heatmaps([heatmaps])[0]
    observed2 = aug2.augment_heatmaps([heatmaps])[0]
    assert observed1.shape == heatmaps.shape
    assert observed2.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed1.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed1.max_value < heatmaps.max_value + 1e-6
    assert heatmaps.min_value - 1e-6 < observed2.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed2.max_value < heatmaps.max_value + 1e-6
    assert np.sum(observed1.get_arr()[mask]) > np.sum(observed2.get_arr()[mask])
    assert np.sum(observed1.get_arr()[~mask]) < np.sum(observed2.get_arr()[~mask])

    # test effectsof increased sigmas
    aug1 = iaa.ElasticTransformation(alpha=3.0, sigma=0.1)
    aug2 = iaa.ElasticTransformation(alpha=3.0, sigma=3.0)
    observed1 = aug1.augment_image(img)
    observed2 = aug2.augment_image(img)
    observed1_std_hori = np.std(observed1.astype(np.float32)[:, 1:] - observed1.astype(np.float32)[:, :-1])
    observed2_std_hori = np.std(observed2.astype(np.float32)[:, 1:] - observed2.astype(np.float32)[:, :-1])
    observed1_std_vert = np.std(observed1.astype(np.float32)[1:, :] - observed1.astype(np.float32)[:-1, :])
    observed2_std_vert = np.std(observed2.astype(np.float32)[1:, :] - observed2.astype(np.float32)[:-1, :])
    observed1_std = (observed1_std_hori + observed1_std_vert) / 2
    observed2_std = (observed2_std_hori + observed2_std_vert) / 2
    assert observed1_std > observed2_std

    # test alpha being iap.Choice
    aug = iaa.ElasticTransformation(alpha=iap.Choice([0.001, 5.0]), sigma=0.25)
    seen = [0, 0]
    for _ in sm.xrange(100):
        observed = aug.augment_image(img)
        diff = np.average(np.abs(img.astype(np.float32) - observed.astype(np.float32)))
        if diff < 1.0:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 10
    assert seen[1] > 10

    # test alpha being tuple
    aug = iaa.ElasticTransformation(alpha=(1.0, 2.0), sigma=0.25)
    assert isinstance(aug.alpha, iap.Uniform)
    assert isinstance(aug.alpha.a, iap.Deterministic)
    assert isinstance(aug.alpha.b, iap.Deterministic)
    assert 1.0 - 1e-8 < aug.alpha.a.value < 1.0 + 1e-8
    assert 2.0 - 1e-8 < aug.alpha.b.value < 2.0 + 1e-8

    # test alpha having bad datatype
    got_exception = False
    try:
        _ = iaa.ElasticTransformation(alpha=False, sigma=0.25)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # test sigma being iap.Choice
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=iap.Choice([0.01, 5.0]))
    seen = [0, 0]
    for _ in sm.xrange(100):
        observed = aug.augment_image(img)

        observed_std_hori = np.std(observed.astype(np.float32)[:, 1:] - observed.astype(np.float32)[:, :-1])
        observed_std_vert = np.std(observed.astype(np.float32)[1:, :] - observed.astype(np.float32)[:-1, :])
        observed_std = (observed_std_hori + observed_std_vert) / 2

        if observed_std > 10.0:
            seen[0] += 1
        else:
            seen[1] += 1
    assert seen[0] > 10
    assert seen[1] > 10

    # test sigma being tuple
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=(1.0, 2.0))
    assert isinstance(aug.sigma, iap.Uniform)
    assert isinstance(aug.sigma.a, iap.Deterministic)
    assert isinstance(aug.sigma.b, iap.Deterministic)
    assert 1.0 - 1e-8 < aug.sigma.a.value < 1.0 + 1e-8
    assert 2.0 - 1e-8 < aug.sigma.b.value < 2.0 + 1e-8

    # test sigma having bad datatype
    got_exception = False
    try:
        _ = iaa.ElasticTransformation(alpha=0.25, sigma=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # order
    # no proper tests here, because unclear how to test
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=ia.ALL)
    assert isinstance(aug.order, iap.Choice)
    assert all([order in aug.order.a for order in [0, 1, 2, 3, 4, 5]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=1)
    assert isinstance(aug.order, iap.Deterministic)
    assert aug.order.value == 1

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=[0, 1, 2])
    assert isinstance(aug.order, iap.Choice)
    assert all([order in aug.order.a for order in [0, 1, 2]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=iap.Choice([0, 1, 2, 3]))
    assert isinstance(aug.order, iap.Choice)
    assert all([order in aug.order.a for order in [0, 1, 2, 3]])

    got_exception = False
    try:
        _ = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # cval
    # few proper tests here, because unclear how to test
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=ia.ALL)
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 0
    assert aug.cval.b.value == 255

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=128)
    assert isinstance(aug.cval, iap.Deterministic)
    assert aug.cval.value == 128

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=(128, 255))
    assert isinstance(aug.cval, iap.DiscreteUniform)
    assert isinstance(aug.cval.a, iap.Deterministic)
    assert isinstance(aug.cval.b, iap.Deterministic)
    assert aug.cval.a.value == 128
    assert aug.cval.b.value == 255

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=[16, 32, 64])
    assert isinstance(aug.cval, iap.Choice)
    assert all([cval in aug.cval.a for cval in [16, 32, 64]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=iap.Choice([16, 32, 64]))
    assert isinstance(aug.cval, iap.Choice)
    assert all([cval in aug.cval.a for cval in [16, 32, 64]])

    aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant", cval=255)
    img = np.zeros((50, 50), dtype=np.uint8)
    observed = aug.augment_image(img)
    assert np.sum(observed == 255) > 0

    aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant", cval=0)
    img = np.zeros((50, 50), dtype=np.uint8)
    observed = aug.augment_image(img)
    assert np.sum(observed == 255) == 0

    got_exception = False
    try:
        _ = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, cval=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # cval with heatmaps
    heatmaps = ia.HeatmapsOnImage(np.zeros((32, 32, 1), dtype=np.float32), shape=(32, 32, 3))
    aug = iaa.ElasticTransformation(alpha=3.0, sigma=3.0, mode="constant", cval=255)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == heatmaps.shape
    assert heatmaps.min_value - 1e-6 < observed.min_value < heatmaps.min_value + 1e-6
    assert heatmaps.max_value - 1e-6 < observed.max_value < heatmaps.max_value + 1e-6
    assert np.sum(observed.get_arr() > 0.01) == 0

    # mode
    # no proper tests here, because unclear how to test
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=ia.ALL)
    assert isinstance(aug.mode, iap.Choice)
    assert all([mode in aug.mode.a for mode in ["constant", "nearest", "reflect", "wrap"]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode="nearest")
    assert isinstance(aug.mode, iap.Deterministic)
    assert aug.mode.value == "nearest"

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=["constant", "nearest"])
    assert isinstance(aug.mode, iap.Choice)
    assert all([mode in aug.mode.a for mode in ["constant", "nearest"]])

    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=iap.Choice(["constant", "nearest"]))
    assert isinstance(aug.mode, iap.Choice)
    assert all([mode in aug.mode.a for mode in ["constant", "nearest"]])

    got_exception = False
    try:
        _ = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, mode=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # keypoints
    # for small alpha, should not move if below threshold
    alpha_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH
    sigma_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = 1.0
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = 0
    kps = [ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25), ia.Keypoint(x=5, y=5),
           ia.Keypoint(x=7, y=4), ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
           ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8), ia.Keypoint(x=12, y=21),
           ia.Keypoint(x=3, y=45), ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
    aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)
    observed = aug.augment_keypoints([kpsoi])[0]
    d = kpsoi.get_coords_array() - observed.get_coords_array()
    d[:, 0] = d[:, 0] ** 2
    d[:, 1] = d[:, 1] ** 2
    d = np.sum(d, axis=1)
    d = np.average(d, axis=0)
    assert d < 1e-8
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = alpha_thresh_orig
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = sigma_thresh_orig

    # for small sigma, should not move if below threshold
    alpha_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH
    sigma_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = 0.0
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = 1.0
    kps = [ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25), ia.Keypoint(x=5, y=5),
           ia.Keypoint(x=7, y=4), ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
           ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8), ia.Keypoint(x=12, y=21),
           ia.Keypoint(x=3, y=45), ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
    aug = iaa.ElasticTransformation(alpha=1.0, sigma=0.001)
    observed = aug.augment_keypoints([kpsoi])[0]
    d = kpsoi.get_coords_array() - observed.get_coords_array()
    d[:, 0] = d[:, 0] ** 2
    d[:, 1] = d[:, 1] ** 2
    d = np.sum(d, axis=1)
    d = np.average(d, axis=0)
    assert d < 1e-8
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = alpha_thresh_orig
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = sigma_thresh_orig

    # for small alpha (at sigma 1.0), should barely move
    alpha_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH
    sigma_thresh_orig = iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = 0
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = 0
    kps = [ia.Keypoint(x=1, y=1), ia.Keypoint(x=15, y=25), ia.Keypoint(x=5, y=5),
           ia.Keypoint(x=7, y=4), ia.Keypoint(x=48, y=5), ia.Keypoint(x=21, y=37),
           ia.Keypoint(x=32, y=39), ia.Keypoint(x=6, y=8), ia.Keypoint(x=12, y=21),
           ia.Keypoint(x=3, y=45), ia.Keypoint(x=45, y=3), ia.Keypoint(x=7, y=48)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(50, 50))
    aug = iaa.ElasticTransformation(alpha=0.001, sigma=1.0)
    observed = aug.augment_keypoints([kpsoi])[0]
    d = kpsoi.get_coords_array() - observed.get_coords_array()
    d[:, 0] = d[:, 0] ** 2
    d[:, 1] = d[:, 1] ** 2
    d = np.sum(d, axis=1)
    d = np.average(d, axis=0)
    assert d < 0.5
    iaa.ElasticTransformation.KEYPOINT_AUG_ALPHA_THRESH = alpha_thresh_orig
    iaa.ElasticTransformation.KEYPOINT_AUG_SIGMA_THRESH = sigma_thresh_orig

    # test alignment between images and heatmaps
    img = np.zeros((80, 80), dtype=np.uint8)
    img[:, 30:50] = 255
    img[30:50, :] = 255
    hm = ia.HeatmapsOnImage(img.astype(np.float32)/255.0, shape=(80, 80))
    aug = iaa.ElasticTransformation(alpha=60.0, sigma=4.0, mode="constant", cval=0)
    aug_det = aug.to_deterministic()
    img_aug = aug_det.augment_image(img)
    hm_aug = aug_det.augment_heatmaps([hm])[0]
    assert hm_aug.shape == (80, 80)
    assert hm_aug.arr_0to1.shape == (80, 80, 1)
    img_aug_mask = img_aug > 255*0.1
    hm_aug_mask = hm_aug.arr_0to1 > 0.1
    same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
    assert (same / img_aug_mask.size) >= 0.99

    # test alignment between images and heatmaps
    # here with heatmaps that are smaller than the image
    img = np.zeros((80, 80), dtype=np.uint8)
    img[:, 30:50] = 255
    img[30:50, :] = 255
    img_small = ia.imresize_single_image(img, (40, 40), interpolation="nearest")
    hm = ia.HeatmapsOnImage(img_small.astype(np.float32)/255.0, shape=(80, 80))
    aug = iaa.ElasticTransformation(alpha=60.0, sigma=4.0, mode="constant", cval=0)
    aug_det = aug.to_deterministic()
    img_aug = aug_det.augment_image(img)
    hm_aug = aug_det.augment_heatmaps([hm])[0]
    assert hm_aug.shape == (80, 80)
    assert hm_aug.arr_0to1.shape == (40, 40, 1)
    img_aug_mask = img_aug > 255*0.1
    hm_aug_mask = ia.imresize_single_image(hm_aug.arr_0to1, (80, 80), interpolation="nearest") > 0.1
    same = np.sum(img_aug_mask == hm_aug_mask[:, :, 0])
    assert (same / img_aug_mask.size) >= 0.96

    # get_parameters()
    aug = iaa.ElasticTransformation(alpha=0.25, sigma=1.0, order=2, cval=10, mode="constant")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert isinstance(params[2], iap.Deterministic)
    assert isinstance(params[3], iap.Deterministic)
    assert isinstance(params[4], iap.Deterministic)
    assert 0.25 - 1e-8 < params[0].value < 0.25 + 1e-8
    assert 1.0 - 1e-8 < params[1].value < 1.0 + 1e-8
    assert params[2].value == 2
    assert params[3].value == 10
    assert params[4].value == "constant"


def test_Rot90():
    img = np.arange(4*4*3).reshape((4, 4, 3)).astype(np.uint8)
    hms = ia.HeatmapsOnImage(img[..., 0:1].astype(np.float32) / 255, shape=(4, 4, 3))
    hms_smaller = ia.HeatmapsOnImage(np.float32([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), shape=(4, 8, 3))
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=2, y=3)], shape=(4, 8, 3))

    # k=0, k=4
    for k in [0, 4]:
        aug = iaa.Rot90(k, keep_size=False)

        img_aug = aug.augment_image(img)
        assert img_aug.dtype == np.uint8
        assert np.array_equal(img_aug, img)

        hms_aug = aug.augment_heatmaps([hms])[0]
        assert hms_aug.arr_0to1.dtype == hms.arr_0to1.dtype
        assert np.allclose(hms_aug.arr_0to1, hms.arr_0to1)
        assert hms_aug.shape == hms.shape

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
        assert kpsoi_aug.shape == kpsoi.shape
        for kp_aug, kp in zip(kpsoi_aug.keypoints, kpsoi.keypoints):
            assert np.allclose([kp_aug.x, kp_aug.y], [kp.x, kp.y])

    # k=1, k=5
    for k in [1, 5]:
        aug = iaa.Rot90(k, keep_size=False)

        img_aug = aug.augment_image(img)
        assert img_aug.dtype == np.uint8
        assert np.array_equal(img_aug, np.rot90(img, 1, axes=(1, 0)))

        hms_aug = aug.augment_heatmaps([hms])[0]
        assert hms_aug.arr_0to1.dtype == hms.arr_0to1.dtype
        assert np.allclose(hms_aug.arr_0to1, np.rot90(hms.arr_0to1, 1, axes=(1, 0)))
        assert hms_aug.shape == (4, 4, 3)

        hms_smaller_aug = aug.augment_heatmaps([hms_smaller])[0]
        assert hms_smaller_aug.arr_0to1.dtype == hms_smaller.arr_0to1.dtype
        assert np.allclose(hms_smaller_aug.arr_0to1, np.rot90(hms_smaller.arr_0to1, 1, axes=(1, 0)))
        assert hms_smaller_aug.shape == (8, 4, 3)

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
        assert kpsoi_aug.shape == (8, 4, 3)
        expected = [(1, 1), (0, 2)]
        for kp_aug, kp in zip(kpsoi_aug.keypoints, expected):
            assert np.allclose([kp_aug.x, kp_aug.y], [kp[0], kp[1]])

    # k=2
    aug = iaa.Rot90(2, keep_size=False)

    img_aug = aug.augment_image(img)
    assert img_aug.dtype == np.uint8
    assert np.array_equal(img_aug, np.rot90(img, 2, axes=(1, 0)))

    hms_aug = aug.augment_heatmaps([hms])[0]
    assert hms_aug.arr_0to1.dtype == hms.arr_0to1.dtype
    assert np.allclose(hms_aug.arr_0to1, np.rot90(hms.arr_0to1, 2, axes=(1, 0)))
    assert hms_aug.shape == (4, 4, 3)

    hms_smaller_aug = aug.augment_heatmaps([hms_smaller])[0]
    assert hms_smaller_aug.arr_0to1.dtype == hms_smaller.arr_0to1.dtype
    assert np.allclose(hms_smaller_aug.arr_0to1, np.rot90(hms_smaller.arr_0to1, 2, axes=(1, 0)))
    assert hms_smaller_aug.shape == (4, 8, 3)

    kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (4, 8, 3)
    expected = [(6, 1), (5, 0)]
    for kp_aug, kp in zip(kpsoi_aug.keypoints, expected):
        assert np.allclose([kp_aug.x, kp_aug.y], [kp[0], kp[1]])

    # k=3, k=-1
    for k in [3, -1]:
        aug = iaa.Rot90(k, keep_size=False)

        img_aug = aug.augment_image(img)
        assert img_aug.dtype == np.uint8
        assert np.array_equal(img_aug, np.rot90(img, 3, axes=(1, 0)))

        hms_aug = aug.augment_heatmaps([hms])[0]
        assert hms_aug.arr_0to1.dtype == hms.arr_0to1.dtype
        assert np.allclose(hms_aug.arr_0to1, np.rot90(hms.arr_0to1, 3, axes=(1, 0)))
        assert hms_aug.shape == (4, 4, 3)

        hms_smaller_aug = aug.augment_heatmaps([hms_smaller])[0]
        assert hms_smaller_aug.arr_0to1.dtype == hms_smaller.arr_0to1.dtype
        assert np.allclose(hms_smaller_aug.arr_0to1, np.rot90(hms_smaller.arr_0to1, 3, axes=(1, 0)))
        assert hms_smaller_aug.shape == (8, 4, 3)

        kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
        assert kpsoi_aug.shape == (8, 4, 3)
        expected = [(2, 6), (3, 5)]
        for kp_aug, kp in zip(kpsoi_aug.keypoints, expected):
            assert np.allclose([kp_aug.x, kp_aug.y], [kp[0], kp[1]])

    # verify once without np.rot90
    img_aug = iaa.Rot90(k=1, keep_size=False).augment_image(
        np.uint8([[1, 0, 0],
                  [0, 2, 0]])
    )
    expected = np.uint8([[0, 1], [2, 0], [0, 0]])
    assert np.array_equal(img_aug, expected)

    # keep_size=True
    aug = iaa.Rot90(1, keep_size=True)

    img_nonsquare = np.arange(5*4*3).reshape((5, 4, 3)).astype(np.uint8)
    img_aug = aug.augment_image(img_nonsquare)
    assert img_aug.dtype == np.uint8
    assert np.array_equal(
        img_aug, ia.imresize_single_image(np.rot90(img_nonsquare, 1, axes=(1, 0)), (5, 4), interpolation="cubic")
    )

    hms_aug = aug.augment_heatmaps([hms])[0]
    assert hms_aug.arr_0to1.dtype == hms.arr_0to1.dtype
    assert np.allclose(hms_aug.arr_0to1, np.rot90(hms.arr_0to1, 1, axes=(1, 0)))
    assert hms_aug.shape == (4, 4, 3)

    hms_smaller_aug = aug.augment_heatmaps([hms_smaller])[0]
    assert hms_smaller_aug.arr_0to1.dtype == hms_smaller.arr_0to1.dtype
    hms_smaller_rot = np.rot90(hms_smaller.arr_0to1, 1, axes=(1, 0))
    hms_smaller_rot = np.clip(ia.imresize_single_image(hms_smaller_rot, (2, 3), interpolation="cubic"), 0.0, 1.0)
    assert np.allclose(hms_smaller_aug.arr_0to1, hms_smaller_rot)
    assert hms_smaller_aug.shape == (4, 8, 3)

    kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (4, 8, 3)
    expected = [(1, 1), (0, 2)]
    expected = [(8*x/4, 4*y/8) for x, y in expected]
    for kp_aug, kp in zip(kpsoi_aug.keypoints, expected):
        assert np.allclose([kp_aug.x, kp_aug.y], [kp[0], kp[1]])

    # test parameter stochasticity
    aug = iaa.Rot90([1, 3])
    assert isinstance(aug.k, iap.Choice)
    assert len(aug.k.a) == 2
    assert aug.k.a[0] == 1
    assert aug.k.a[1] == 3

    class _TwoValueParam(iap.StochasticParameter):
        def __init__(self, v1, v2):
            super(_TwoValueParam, self).__init__()
            self.v1 = v1
            self.v2 = v2

        def _draw_samples(self, size, random_state):
            arr = np.full(size, self.v1, dtype=np.int32)
            arr[1::2] = self.v2
            return arr

    aug = iaa.Rot90(_TwoValueParam(1, 2), keep_size=False)
    imgs_aug = aug.augment_images([img] * 4)
    assert np.array_equal(imgs_aug[0], np.rot90(img, 1, axes=(1, 0)))
    assert np.array_equal(imgs_aug[1], np.rot90(img, 2, axes=(1, 0)))
    assert np.array_equal(imgs_aug[2], np.rot90(img, 1, axes=(1, 0)))
    assert np.array_equal(imgs_aug[3], np.rot90(img, 2, axes=(1, 0)))

    hms_aug = aug.augment_heatmaps([hms_smaller] * 4)
    assert hms_aug[0].shape == (8, 4, 3)
    assert hms_aug[1].shape == (4, 8, 3)
    assert hms_aug[2].shape == (8, 4, 3)
    assert hms_aug[3].shape == (4, 8, 3)
    assert np.allclose(hms_aug[0].arr_0to1, np.rot90(hms_smaller.arr_0to1, 1, axes=(1, 0)))
    assert np.allclose(hms_aug[1].arr_0to1, np.rot90(hms_smaller.arr_0to1, 2, axes=(1, 0)))
    assert np.allclose(hms_aug[2].arr_0to1, np.rot90(hms_smaller.arr_0to1, 1, axes=(1, 0)))
    assert np.allclose(hms_aug[3].arr_0to1, np.rot90(hms_smaller.arr_0to1, 2, axes=(1, 0)))

    kpsoi_aug = aug.augment_keypoints([kpsoi] * 4)
    assert kpsoi_aug[0].shape == (8, 4, 3)
    assert kpsoi_aug[1].shape == (4, 8, 3)
    assert kpsoi_aug[2].shape == (8, 4, 3)
    assert kpsoi_aug[3].shape == (4, 8, 3)
    assert np.allclose([kpsoi_aug[0].keypoints[0].x, kpsoi_aug[0].keypoints[0].y], [1, 1])
    assert np.allclose([kpsoi_aug[0].keypoints[1].x, kpsoi_aug[0].keypoints[1].y], [0, 2])
    assert np.allclose([kpsoi_aug[1].keypoints[0].x, kpsoi_aug[1].keypoints[0].y], [6, 1])
    assert np.allclose([kpsoi_aug[1].keypoints[1].x, kpsoi_aug[1].keypoints[1].y], [5, 0])
    assert np.allclose([kpsoi_aug[2].keypoints[0].x, kpsoi_aug[2].keypoints[0].y], [1, 1])
    assert np.allclose([kpsoi_aug[2].keypoints[1].x, kpsoi_aug[2].keypoints[1].y], [0, 2])
    assert np.allclose([kpsoi_aug[3].keypoints[0].x, kpsoi_aug[3].keypoints[0].y], [6, 1])
    assert np.allclose([kpsoi_aug[3].keypoints[1].x, kpsoi_aug[3].keypoints[1].y], [5, 0])

    # get_parameters()
    aug = iaa.Rot90([1, 3], keep_size=False)
    assert aug.get_parameters()[0] == aug.k
    assert aug.get_parameters()[1] is False


if __name__ == "__main__":
    main()
