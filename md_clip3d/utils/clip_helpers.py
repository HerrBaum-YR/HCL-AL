import os
import glob
import numpy as np
import SimpleITK as sitk


def last_checkpoint(chk_root):
    """
    find the directory of last check point
    :param chk_root: the check point root directory, which may contain multiple checkpoints
    :return: the last check point directory
    """

    last_epoch = -1
    chk_folders = os.path.join(chk_root, 'chk_*')
    for folder in glob.glob(chk_folders):
        folder_name = os.path.basename(folder)
        tokens = folder_name.split('_')
        epoch = int(tokens[-1])
        if epoch > last_epoch:
            last_epoch = epoch

    if last_epoch == -1:
        raise OSError('No checkpoint folder found!')

    return os.path.join(chk_root, 'chk_{}'.format(last_epoch))

def get_checkpoint(chk_root, epoch=-1):
    if epoch > 0:
        chk_path = os.path.join(chk_root, 'chk_{}'.format(epoch))
        assert os.path.isdir(chk_path), "checkpoints not exist: " + chk_path
    else:
        chk_path = last_checkpoint(chk_root)

    return chk_path

def image_crop(im, crop_center, crop_spacing, crop_size, crop_axes=np.eye(3), method='nn', default_value=0):
    """
    Args:
        im (sitk.Image): input sitk image
        crop_center (list/tuple/np.ndarray): [x,y,z]
        crop_spacing (list/tuple/np.ndarray): [sp_x, sp_y, sp_z]
        crop_size (list/tuple/np.ndarray): [size_x, size_y, size_z]
        crop_axes (np.ndarray): default np.eye(3)
        method (str): interpolation method, nn/linear

    Return:
        np.ndarray: cropped image [W, H, D]
    """
    method = method.lower()
    assert method in ['nn', "linear"], "method must be 'nn' or 'linear'"
    assert isinstance(crop_axes, np.ndarray) and crop_axes.shape == (3, 3), "crop_axes must be 3x3 numpy array"
    assert len(crop_center) == len(crop_spacing) == len(crop_size) == 3, "Parameters must be 3D"
    crop_center = np.asarray(crop_center, dtype=np.float32)
    crop_spacing = np.asarray(crop_spacing, dtype=np.float32)
    crop_size = np.asarray(crop_size, dtype=np.int32)

    #  set interpolator
    if method == 'nn':
        sitk_interpolator = sitk.sitkNearestNeighbor
    elif method == "linear":
        sitk_interpolator = sitk.sitkLinear
    else:
        raise ValueError(f"Invalid interpolation method：{method}")

    # set reference image
    reference_image = sitk.Image(int(crop_size[0]), int(crop_size[1]), int(crop_size[2]), im.GetPixelID())
    reference_image.SetSpacing(crop_spacing.tolist())
    reference_image.SetOrigin((0, 0, 0))
    reference_image.SetDirection(crop_axes.flatten().tolist())

    # calculate crop center
    crop_center_index = [0, 0, 0]
    for i in range(3):
        crop_center_index[i] = crop_center[i] - (crop_size[i] * crop_spacing[i]/ 2)

    # crop image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetOutputOrigin(crop_center_index)
    resampler.SetInterpolator(sitk_interpolator)
    resampler.SetDefaultPixelValue(default_value)
    cropped_im = resampler.Execute(im)
    return cropped_im

def read_crop_adaptive(impath, crop_center, crop_spacing, crop_axes, crop_size):
    """ read image crop from disk adaptively, also support image3dd format
    :param impath           the path of input image
    :param crop_center      the crop center in world space
    :param crop_spacing     the crop spacing in mm
    :param crop_size        the crop size in voxels
    :param crop_axes        the crop axes (None if use RAI coordinate)
    :return an image crop
    """
    assert os.path.isfile(impath), 'image path does not exist: {}'.format(impath)
    crop = sitk.ReadImage(impath, outputPixelType=sitk.sitkFloat32)
    crop = image_crop(crop, crop_center, crop_spacing, crop_size, crop_axes, 'linear')
    return crop

def read_crop_adaptive_nn(impath, crop_center, crop_spacing, crop_axes, crop_size):
    """ read image crop from disk adaptively, also support image3dd format
    :param impath           the path of input image
    :param crop_center      the crop center in world space
    :param crop_spacing     the crop spacing in mm
    :param crop_size        the crop size in voxels
    :param crop_axes        the crop axes (None if use RAI coordinate)
    :return an image crop
    """
    assert os.path.isfile(impath), 'image path does not exist: {}'.format(impath)
    crop = sitk.ReadImage(impath, outputPixelType=sitk.sitkFloat32)
    crop = image_crop(crop, crop_center, crop_spacing, crop_size, crop_axes, 'nn')
    return crop

def intensity_normalize(image: sitk.Image, mean: float, stddev: float, 
                        clip: bool=False, clip_min: float=-1, clip_max: float=1):
   image_npy = sitk.GetArrayFromImage(image)
   image_npy = image_npy.astype(np.float32)
   normalized = (image_npy - mean) / (stddev + 1e-8)
   if clip:
      normalized = np.clip(normalized, clip_min, clip_max)
   result = sitk.GetImageFromArray(normalized)
   result.CopyInformation(image)
   return result

class FixedNormalizer(object):
    """
    use fixed mean and stddev to normalize image intensities
    intensity = (intensity - mean) / stddev
    if clip is enabled:
        intensity = np.clip((intensity - mean) / stddev, -1, 1)
    """
    def __init__(self, mean, stddev, clip=False):
        """ constructor """
        assert stddev > 0, 'stddev must be positive'
        assert isinstance(clip, bool), 'clip must be a boolean'
        self.mean = mean
        self.stddev = stddev
        self.clip = clip

    def __call__(self, image):
        """ normalize image """
        if isinstance(image, sitk.Image):
            image = intensity_normalize(image, self.mean, self.stddev, self.clip)
        elif isinstance(image, (list, tuple)):
            for idx, im in enumerate(image):
                assert isinstance(im, sitk.Image)
                image[idx] = intensity_normalize(im, self.mean, self.stddev, self.clip)
        else:
            raise ValueError('Unknown type of input. Normalizer only supports sitk.Image or sitk.Image list/tuple')
        return image

    def static_obj(self):
        """ get a static normalizer object by removing randomness """
        obj = FixedNormalizer(self.mean, self.stddev, self.clip)
        return obj

    def to_dict(self):
        """ convert parameters to dictionary """
        obj = {'type': 0, 'mean': self.mean, 'stddev': self.stddev, 'clip': self.clip}
        return obj

if __name__ == '__main__':
    pass

