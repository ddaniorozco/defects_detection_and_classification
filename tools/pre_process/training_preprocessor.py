import random
import numpy as np
from glob import glob
import shutil
import os
import argparse
from labelme2coco import labelme2coco
import json
import cv2
import base64


class TrainingPreprocessor:

    def __init__(self, source_dir, train_dir, test_dir, val_dir, test_percent, val_percent):

        self.source_dir = source_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.test_percent = test_percent
        self.val_percent = val_percent

    def export(self):

        for filename in glob(os.path.join(self.source_dir, '*.*')):
            shutil.copy(filename, self.train_dir)
        files_images = sorted(glob(os.path.join(self.train_dir, '*[!.json]')))
        files_json = sorted(glob(os.path.join(self.train_dir, '*[.json]')))

        if len(files_images) > len(files_json):
            print('Error: Not the same amount of files. More images than JSON.')
        elif len(files_images) < len(files_json):
            print('Error: Not the same amount of files. More JSON than images.')
        elif len(files_images) == len(files_json):
            print('Processing images and JSON')

        self.resize_normalize_many(files_images, files_json)
        self.split_train_test_val_with_jsons(files_images)
        self.labelme_2_coco()

    def split_train_test_val_with_jsons(self, files_images):

        n = len(files_images)
        test_percent = int(self.test_percent)
        test_files = random.sample(files_images, int(np.round(test_percent * n / 100.)))

        for file in test_files:
            json_file = file.replace(os.path.splitext(file)[1], '.json')
            shutil.move(file, self.test_dir)
            shutil.move(json_file, self.test_dir)

        files = glob(os.path.join(self.train_dir + '*[!.json]'))
        val_percent = int(self.val_percent)
        val_files = random.sample(files, int(np.round(val_percent * n / 100.)))
        for file in val_files:
            json_file = file.replace(os.path.splitext(file)[1], '.json')
            shutil.move(file, self.val_dir)
            shutil.move(json_file, self.val_dir)

    def labelme_2_coco(self):

        paths_list = [self.train_dir, self.test_dir, self.val_dir]
        for path in paths_list:
            only_json = glob(os.path.join(path + '*.json'))
            json_name_file = os.path.basename(os.path.normpath(path)) + '.json'
            labelme2coco(only_json, path + json_name_file)

    def resize_images(self, img, json_path):

        im = cv2.imread(img, cv2.IMREAD_COLOR)

        original_height = im.shape[0]
        original_width = im.shape[1]
        self.fixed_side = 768

        if original_height < original_width:
            f = (self.fixed_side / original_width)
            self.hsize = int(original_height * f)
            im_resized = cv2.resize(im, (self.fixed_side, self.hsize))
        elif original_height > original_width:
            f = (self.fixed_side / original_height)
            self.hsize = int(original_width * f)
            im_resized = cv2.resize(im, (self.hsize, self.fixed_side))
        elif original_height == original_width:
            im_resized = cv2.resize(im, (self.fixed_side, self.fixed_side))

        cv2.imwrite(img, im_resized)

        self.resize_height = im_resized.shape[0]
        self.resize_width = im_resized.shape[1]
        self.ratio_height = self.resize_height / original_height
        self.ratio_width = self.resize_width / original_width

        if original_width != self.resize_width and original_height != self.resize_height:
            im_np = cv2.imread(img, cv2.IMREAD_COLOR)
            with open(json_path) as f:
                data = json.loads(f.read())
                json_modified = self.normalize_bbox(data, im_np)
            with open((os.path.join(json_path)), 'w') as outputfile:
                json.dump(json_modified, outputfile, indent=2)

    def normalize_bbox(self, data, im_np):

        access_shapes = data['shapes']
        for data_shapes in access_shapes:

            bbox = data_shapes['points']
            bbox[0][0], bbox[1][0] = bbox[0][0] * self.ratio_height, bbox[1][0] * self.ratio_height
            bbox[0][1], bbox[1][1] = bbox[0][1] * self.ratio_width, bbox[1][1] * self.ratio_width
            data_shapes["points"] = [[bbox[0][0], bbox[0][1]], [bbox[1][0], bbox[1][1]]]

        data['imageHeight'] = self.resize_height
        data['imageWidth'] = self.resize_width
        img_b64_str = self.base64_encode(im_np, ext='.png').decode('utf-8')

        data['imageData'] = img_b64_str

        return data

    @staticmethod
    def base64_encode(image_rgb, ext='.jpg'):
        retval, buffer = cv2.imencode(ext, image_rgb)
        if retval:
            image_str = base64.b64encode(buffer)
            return image_str

    def resize_normalize_many(self, files_images, files_json):

        for (file_image, file_json) in zip(files_images, files_json):
            self.resize_images(file_image, file_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training preprocessing")
    parser.add_argument('--source_folder', type=str, required=True)
    parser.add_argument('--train_folder', type=str, required=True)
    parser.add_argument('--test_folder', type=str, required=True)
    parser.add_argument('--val_folder', type=str, required=True)
    parser.add_argument('--test_percentage', type=str, required=False)
    parser.add_argument('--val_percentage', type=str, required=False)
    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()

    source_folder = args.source_folder
    train_folder = args.train_folder
    test_folder = args.test_folder
    val_folder = args.val_folder
    test_percentage = args.test_percentage
    val_percentage = args.val_percentage

    training_preprocessor = TrainingPreprocessor(source_folder, train_folder, test_folder, val_folder, test_percentage,
                                                 val_percentage)

    training_preprocessor.export()
