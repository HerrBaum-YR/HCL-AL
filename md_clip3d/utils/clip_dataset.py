import math

import torch
from torch.utils.data import Dataset

from md_clip3d.utils.base.rotation import axis_angle_to_rotation_matrix
from md_clip3d.utils.base.dilate import imdilate
from md_clip3d.utils.clip_fileio import load_json_as_dict, load_im_clip_list
from md_clip3d.utils.clip_helpers import *
from md_clip3d.utils.clip_utils import *
from md_clip3d.tokenizer.clip_tokenize import tokenize


class ClipClsDataSet(Dataset):
    """ training data set for volumetric clip """

    def __init__(self, im_clip_list, translate_json, target_header, input_channels, 
                 crop_size, crop_normalizers, sample_method, interpolation, 
                 spacing, box_center_random, box_percent_padding,
                 rotate_config, scale_config, random_flip, lesion_idx,
                 net_name, pretrained_model_dir, mode):
        """
        :param im_clip_list: image-clip list file
        :param translate_json:
        :param target_header:
        :param input_channels: the number of input image
        :param crop_size: crop voxel size, e.g., [96, 96, 96]
        :param crop_normalizers: used to normalize the image crops
        :param sample_method: image crop method
        :param interpolation:
        :param spacing: spacing information
        :param box_center_random: random range of center point for fixed length sample
        :param box_center_padding:
        :param rotate_config: random rotate input crop in degrees or not
        :param scale_config:
        :param random_flip: random flip or not
        :param lesion_idx:
        :param net_name: 
        :param mode: 
        """
        if im_clip_list.endswith('csv'):
            self.ims_list, self.bbox_info_list, self.other_info_list = load_im_clip_list(
                im_clip_list, input_channels)
        else:
            raise ValueError('im_clip_list must be a csv file')
        
        if translate_json.endswith('json'):
            self.words_to_texts_dict = load_json_as_dict(translate_json)
        else:
            raise ValueError('location_json must be a json file')

        self.target_header = target_header
        self.input_channels = input_channels

        self.crop_size = np.array(crop_size, dtype=np.int32)
        assert self.crop_size.size == 3, 'only 3-element of crop size is supported'

        self.crop_normalizers = crop_normalizers 

        self.sample_method = sample_method
        if isinstance(interpolation, str):
            self.interpolation = [interpolation] * self.input_channels
        else:
            assert len(interpolation) == self.input_channels, \
                "The number of interpolation methods does not match the input channels."
            self.interpolation = interpolation

        self.spacing = spacing
        self.box_center_random = box_center_random
        self.box_percent_padding = box_percent_padding

        self.random_flip = random_flip

        self.rot_prob = rotate_config['rot_prob']
        self.rot_axis = rotate_config['rot_axis']
        self.rot_angle_degree = abs(rotate_config['rot_angle_degree'])

        self.scale_prob = scale_config['scale_prob']
        self.scale_min_ratio = abs(scale_config['scale_min_ratio'])
        self.scale_max_ratio = abs(scale_config['scale_max_ratio'])
        self.scale_isotropic = scale_config['scale_isotropic']

        self.lesion_idx = lesion_idx if lesion_idx else []

        self.net_name = net_name
        self.pretrained_model_dir = pretrained_model_dir
        self.mode = mode

    def __len__(self):
        """ get the number of images in this data set """
        return len(self.ims_list)

    def __getitem__(self, index):
        """ get a training sample - image and label pair
        :param index:  the sample index
        :return cropped image, cropped mask
        """
        images_path, bbox_info = self.ims_list[index], self.bbox_info_list[index]

        crop_images = []
        crop_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.double)
        crop_size = self.crop_size
        crop_scale_ratio = np.array([1.0, 1.0, 1.0])

        for image_path in images_path:
            assert os.path.isfile(image_path), 'image path does not exist: {}'.format(image_path)

        # base on the prob set in config to do roatation and scale
        rotate_flag = np.random.choice([False, True], p=[1 - self.rot_prob, self.rot_prob]) if 0 <= self.rot_prob <= 1 else False
        scale_flag = np.random.choice([False, True], p=[1 - self.scale_prob, self.scale_prob]) if 0 <= self.scale_prob <= 1 else False

        if rotate_flag or scale_flag:
            if rotate_flag:
                # random rotate input image in degrees
                rot_axis = uniform_sample_point_from_unit_sphere() if self.rot_axis is None else np.array(self.rot_axis)
                rot_axis = rot_axis[0]
                angle = np.random.random() * self.rot_angle_degree * math.pi / 180.0
                crop_axes = axis_angle_to_rotation_matrix(rot_axis, angle)

            if scale_flag:
                # random scale input image with scale ratio
                if self.scale_isotropic:
                    scale_ratio = np.random.uniform(self.scale_min_ratio, self.scale_max_ratio)
                    scale_ratio = np.array([scale_ratio] * 3)
                else:
                    scale_ratio = np.random.uniform(self.scale_min_ratio, self.scale_max_ratio, (3,))
                crop_scale_ratio *= scale_ratio

        # calculate crop center and spacing
        if self.sample_method == "fixed_length":
            box_center = bbox_info[0:3]
            center_new = [box_center[0] + np.random.uniform(-self.box_center_random[0], self.box_center_random[0]),
                          box_center[1] + np.random.uniform(-self.box_center_random[1], self.box_center_random[1]),
                          box_center[2] + np.random.uniform(-self.box_center_random[2], self.box_center_random[2])]
            crop_spacing = self.spacing / crop_scale_ratio
        else:
            raise ValueError("Unknown sample method: " + self.sample_method)
        
        for idx in range(len(images_path)):
            if self.interpolation[idx] == 'nn':
                crop_image = read_crop_adaptive_nn(images_path[idx], center_new, crop_spacing, crop_axes, crop_size)
            elif self.interpolation[idx] == 'linear':
                crop_image = read_crop_adaptive(images_path[idx], center_new, crop_spacing, crop_axes, crop_size)
            else:
                raise ValueError("Unknown interpolation method: " + self.interpolation[idx])
            crop_images.append(crop_image)
  
        # filter out non-target lesions
        for idx in self.lesion_idx:
            _, crop_images[idx] = label_connected_component(crop_images[idx], connectivity=26)
            target_label, _ = search_label_by_bbox(crop_images[idx], bbox_info)
            if target_label:
                crop_images[idx] = convert_multi_label_to_binary(crop_images[idx], target_label)
                continue
            crop_lesion = read_crop_adaptive_nn(images_path[idx], center_new, crop_spacing / 2, crop_axes, crop_size * 2)
            _, crop_lesion = label_connected_component(crop_lesion, connectivity=26)
            target_label, labels = search_label_by_bbox(crop_lesion, bbox_info)
            if target_label:
                crop_lesion = convert_multi_label_to_binary(crop_lesion, target_label)
                crop_lesion = imdilate(crop_lesion, 1, 1, connectivity=6)
                crop_lesion = image_crop(crop_lesion, center_new, crop_spacing, crop_size, crop_axes)
                crop_images[idx] = crop_lesion
                continue
            print(f'Warning: Cannot match bbox and lesion mask!')
            crop_images[idx] = convert_labels_to_target_label(crop_images[idx], labels, 0)

        for idx in range(len(crop_images)):
            if self.crop_normalizers[idx] is not None:
                crop_images[idx] = self.crop_normalizers[idx](crop_images[idx])
        
        # convert to numpy
        for idx in range(len(crop_images)):
            img_npy = sitk.GetArrayFromImage(crop_images[idx])
            crop_images[idx] = np.transpose(img_npy, (2, 1, 0))

        # random flip the crop_image
        if self.random_flip:
            # value in flip flag is 1 or -1
            flip_flag = np.random.randint(low=0, high=2, size=3) * 2 - 1
            for idx in range(len(crop_images)):
                crop_images[idx] = crop_image[::flip_flag[0], ::flip_flag[1], ::flip_flag[2]]

        crop_images = torch.stack([torch.from_numpy(crop_image) for crop_image in crop_images])
        
        word = self.other_info_list[index][self.target_header]
        text = self.words_to_texts_dict[word]
        encoded_text = tokenize(text, self.net_name, self.pretrained_model_dir)
        input_ids = encoded_text.squeeze(0)

        if self.mode == 'train':
            return crop_images, input_ids
        elif self.mode == 'train_hardnegsample':
            assert isinstance(self.target_header, str)
            other_texts = [self.words_to_texts_dict[w] for w in self.words_to_texts_dict if w != word]
            encoded_other_text = tokenize(other_texts, self.net_name, self.pretrained_model_dir)
            other_ids = encoded_other_text
            return crop_images, input_ids, other_ids
        

