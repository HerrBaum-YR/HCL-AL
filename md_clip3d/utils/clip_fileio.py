import os
import csv
import json
from collections import namedtuple

def load_json_as_dict(json_path):
   assert os.path.exists(json_path), f"Cannot find {json_path}."
   json_dict = {}
   with open(json_path, encoding="utf-8") as f:
      json_to_dict = json.load(f)
   for key in json_to_dict:
      json_dict[key] = json_to_dict[key]
   return json_dict

def write_dict_as_json(data, json_path):
   assert json_path.endswith(".json"), 'Invalid save json path'
   if not os.path.isdir(os.path.dirname(json_path)):
      os.makedirs(os.path.dirname(json_path))
   
   with open(json_path, 'w', encoding='utf-8') as fp:
      json.dump(data, fp, ensure_ascii=False, indent=2)

def write_list_as_csv(data, csv_path):
   assert csv_path.endswith(".csv"), 'Invalid save csv path'
   if not os.path.isdir(os.path.dirname(csv_path)):
      os.makedirs(os.path.dirname(csv_path))

   if isinstance(data, dict):
      data = [data]
   elif isinstance(data, list):
      assert len(data), 'Empty list.'
      output_file = open(csv_path, "w")
      writer = csv.DictWriter(output_file, fieldnames=list(data[0].keys()))
      writer.writeheader()
      writer.writerows(data)
   else:
      raise ValueError('Invalid data type.') 

def load_im_clip_list(csv_path, input_channels):
   ims_list, bbox_info_list, other_info_list = [], [], []

   ims_headers = ['image_path'] + ["image_path"+str(i) for i in range(input_channels) if i]
   bbox_headers = ['x', 'y', 'z', 'width', 'height', 'depth']

   with open(csv_path, 'r', encoding='utf-8') as fp:
      reader = csv.DictReader(fp)
      other_headers = [h for h in reader.fieldnames if h not in ims_headers + bbox_headers]
      for line in reader:
         ims_list.append([line[ims_header] for ims_header in ims_headers])
         bbox_info_list.append([float(line[bbox_header]) for bbox_header in bbox_headers])
         other_info_dict = dict()
         for other_header in other_headers:
            other_info_dict[other_header] = line[other_header]
         other_info_list.append(other_info_dict)
         
   return ims_list, bbox_info_list, other_info_list

def read_clip_csv(csv_file, input_channels):
   """
   :param csv_file:        csv file path
   :param input_channels:  the number of input image in one case
   :return: return image path list, label name and bounding box information
   """
   ims_list, labels, bbox_info_list = [], [], []
   with open(csv_file, 'r') as fp:
      reader = csv.reader(fp)
      head = next(reader)
      for i in range(len(head)):
         if head[i] == "class":              
            head[i] = 'type'
      Row = namedtuple('Row', head)

      for line in reader:
         row = Row(*line)
         im_list = list()
         for i in range(input_channels):
            if i == 0:
               im_list.append(row.__getattribute__("image_path"))
            else:
               im_list.append(row.__getattribute__("image_path"+str(i)))
         ims_list.append(im_list)
         bbox_info_list.append([float(row.x), float(row.y), float(row.z), 
                                float(row.width), float(row.height), float(row.depth)])
         try:
            labels.append(row.type)
         except Exception as e:
            labels.append(None)

   return ims_list, labels, bbox_info_list

def get_input_fieldname_from_csv(csv_path):
   """
   single modality: image_path
   multi modality: image_path, image_path1, image_path2...
   :param csv_path:   csv file path
   :return:           input channel
   """
   input_file = open(csv_path, "r", encoding='utf-8-sig')
   input_reader = csv.DictReader(input_file)

   fieldname = []
   for read in input_reader:
      fieldname = list(read.keys())
      break

   return fieldname

def get_input_channels_from_fieldname(fieldname):
   channels = 0

   while True:
      path = "image_path"
      if channels != 0:
         path = path + str(channels)

      if path in fieldname:
         channels += 1
      else:
         return channels

def load_case_csv(input_csv, input_channels=1):
   """
   load case csv, organize nodules by image
   :param input_csv:    a input csv file
   :return:
      case_imnames:   the case images list, organized by input channels image name.
      case_box_list:      if "prob" in nodule_csv, return a list of box[impath, probability, x, y, z, sx, sy, sz];
                        else, return a list of nodules[impath, x, y, z, sx, sy, sz]
   """
   case_imnames, case_list = [], []

   with open(input_csv, 'r', encoding='utf-8-sig') as fp:
      reader = csv.DictReader(fp)
      for read in reader:
         images_path = []
         for i in range(input_channels):
            if i == 0:
               images_path.append(read["image_path"])
            else:
               images_path.append(read["image_path" + str(i)])
         
         if images_path not in case_imnames:
            case_imnames.append(images_path)
            case_list.append([read])
         else:
            idx = case_imnames.index(images_path)
            case_list[idx].append(read)

   return case_imnames, case_list