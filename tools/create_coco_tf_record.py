# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw COCO dataset to TFRecord for object_detection.

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record_2.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --test_annotations_file="${TEST_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import contextlib2
import numpy as np
from PIL import Image

from tensorflow.python.framework.versions import VERSION

if VERSION >= "2.0.0a0":
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf

from pycocotools import mask
import sys
from tqdm import tqdm
from glob import glob
import cv2

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', True,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: True.')
tf.flags.DEFINE_boolean('shards', True,
                        'Whether to to make one tfrecord or shards ')
tf.flags.DEFINE_string('train_image_dir', '',
                       'Training image directory.')
tf.flags.DEFINE_string('test_image_dir', '',
                       'Test image directory.')
tf.flags.DEFINE_string('train_annotations_file', '',
                       'Training annotations JSON file.')
tf.flags.DEFINE_string('test_annotations_file', '',
                       'Test-dev annotations JSON file.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

labels_colors = {1: (0, 255, 0), 2: (0, 0, 255), 'else': (0, 0, 255)}


def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False):
    """Converts image and annotations to a tf.Example proto.

  Args:
    image: dict with keys:
      [u'license', u'file_name', u'coco_url', u'height', u'width',
      u'date_captured', u'flickr_url', u'id']
    annotations_list:
      list of dicts with keys:
      [u'segmentation', u'area', u'iscrowd', u'image_id',
      u'bbox', u'category_id', u'id']
      Notice that bounding box coordinates in the official COCO dataset are
      given as [x, y, width, height] tuples using absolute coordinates where
      x, y represent the top-left (0-indexed) corner.  This function converts
      to the format expected by the Tensorflow Object Detection API (which is
      which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
      to image size).
    image_dir: directory containing the image files.
    category_index: a dict containing COCO category information keyed
      by the 'id' field of each category.  See the
      label_map_util.create_category_index function.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
  Returns:
    example: The converted tf.Example
    num_annotations_skipped: Number of (invalid) annotations that were ignored.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        area.append(object_annotations['area'])

        if include_masks:
            run_len_encoding = mask.frPyObjects(object_annotations['segmentation'],
                                                image_height, image_width)
            binary_mask = mask.decode(run_len_encoding)
            if not object_annotations['iscrowd']:
                binary_mask = np.amax(binary_mask, axis=2)
            pil_image = Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())
    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        # 'image/object/is_crowd':
        #     dataset_util.int64_list_feature(is_crowd),
        # 'image/object/area':
        #     dataset_util.float_list_feature(area)
    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return key, example, num_annotations_skipped


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards
    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx + 1, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.python_io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def read_from_folder(tfrecord_files_dir, images_dst, labels_to_index_map, masks):
    files = glob(os.path.join(tfrecord_files_dir, '*.record*'))
    for file in files:
        read_tfrecord_file(file, images_dst, labels_to_index_map, masks)


def read_tfrecord_file(tfrecord_file_name, folder_name, labels_to_index_map, masks):
    record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=tfrecord_file_name)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = example.features.feature['image/height'].int64_list.value[0]
        width = example.features.feature['image/width'].int64_list.value[0]

        file_name = example.features.feature['image/filename'].bytes_list.value[0].decode("utf-8")

        encoded_image = example.features.feature['image/encoded'].bytes_list.value[0]

        image_np = np.frombuffer(encoded_image, dtype=np.uint8)
        decoded_image = cv2.imdecode(image_np, flags=cv2.IMREAD_UNCHANGED)

        image = np.array(decoded_image[:, :, :3])

        texts = example.features.feature['image/object/class/text'].bytes_list.value

        xmins = example.features.feature['image/object/bbox/xmin'].float_list.value
        ymins = example.features.feature['image/object/bbox/ymin'].float_list.value
        xmaxs = example.features.feature['image/object/bbox/xmax'].float_list.value
        ymaxs = example.features.feature['image/object/bbox/ymax'].float_list.value

        classes = example.features.feature['image/object/class/text'].bytes_list.value

        if masks:
            masks = example.features.feature['image/object/mask'].bytes_list

        for (xmin, ymin, xmax, ymax, text, class_name) in zip(xmins, ymins, xmaxs, ymaxs, texts, classes):
            min_p = np.array([xmin, ymin])
            max_p = np.array([xmax, ymax])

            min_p = np.round(min_p * [width, height]).astype(np.int32)
            max_p = np.round(max_p * [width, height]).astype(np.int32)

            if class_name.decode() in labels_to_index_map and labels_to_index_map[class_name.decode()] in labels_colors:
                color = labels_colors[labels_to_index_map[class_name.decode()]]
            else:
                color = labels_colors['else']
            cv2.rectangle(image, tuple(min_p), tuple(max_p), color, 2)

            draw_text(image, str(text.decode()), tuple([min_p[0], max_p[1]]), color)

        cv2.imwrite(folder_name + file_name, image)

        if masks:
            for i, e in enumerate(masks.value):
                # image_0_1 = Image.open(io.BytesIO(aaa))
                image_0_1 = Image.open(io.BytesIO(e))
                masked = np.asarray(image_0_1)
                masked = (masked * 255).astype(np.uint8)
                cv2.imwrite(folder_name + str(i) + file_name, masked)


# def read_tfrecord_file(tfrecord_file_name, folder_name, labels_to_index_map, labels_colors):
#
#     record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_file_name)
#
#     for string_record in record_iterator:
#         example = tf.train.Example()
#         example.ParseFromString(string_record)
#
#         # print(example)
#
#         height = example.features.feature['image/height'].int64_list.value[0]
#         width = example.features.feature['image/width'].int64_list.value[0]
#
#         file_name = example.features.feature['image/filename'].bytes_list.value[0].decode("utf-8")
#
#         encoded_image = example.features.feature['image/encoded'].bytes_list.value[0]
#
#         image_format = example.features.feature['image/format'].bytes_list.value[0].decode("utf-8")
#
#         image_np = np.frombuffer(encoded_image, dtype=np.uint8)
#         decoded_image = cv2.imdecode(image_np, flags=cv2.IMREAD_UNCHANGED)
#
#         classes = example.features.feature['image/object/class/text'].bytes_list.value
#         labels = example.features.feature['image/object/class/label'].int64_list.value
#
#         xmins = example.features.feature['image/object/bbox/xmin'].float_list.value
#         ymins = example.features.feature['image/object/bbox/ymin'].float_list.value
#         xmaxs = example.features.feature['image/object/bbox/xmax'].float_list.value
#         ymaxs = example.features.feature['image/object/bbox/ymax'].float_list.value
#
#         for (xmin, ymin, xmax, ymax, label, class_name) in zip(xmins, ymins, xmaxs, ymaxs, labels, classes):
#
#             min_p = np.array([xmin, ymin])
#             max_p = np.array([xmax, ymax])
#
#             min_p = np.round(min_p * [width, height]).astype(np.int32)
#             max_p = np.round(max_p * [width, height]).astype(np.int32)
#
#             if labels_to_index_map[class_name.decode()] in labels_colors:
#                 color = labels_colors[labels_to_index_map[class_name.decode()]]
#             else:
#                 color = labels_colors['else']
#             cv2.rectangle(decoded_image, tuple(min_p), tuple(max_p), color, 2)
#             draw_text(decoded_image, str(label) + ' - ' + class_name.decode("utf-8"), tuple([min_p[0], max_p[1]]))
#
#         cv2.imwrite(folder_name + file_name, decoded_image)


def draw_text(img, text, origin, color):
    thickness = 2
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    textsize, baseline = cv2.getTextSize(text, fontface, fontscale, thickness)
    baseline += thickness
    text_origin = np.array((origin[0], origin[1] - textsize[1]))
    # cv2.rectangle(img, tuple((text_origin + (0, baseline)).astype(int)),
    #               tuple((text_origin + (textsize[0], -textsize[1])).astype(int)), (128, 128, 128), -1)
    cv2.putText(img, text, tuple((text_origin + (0, baseline / 2)).astype(int)), fontface, fontscale,
                color, thickness, 8)
    return img


def _create_tf_record_from_coco_annotations_with_shards(
        annotations_file, image_dir, output_path, include_masks):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    """
    num_shards = 10

    with contextlib2.ExitStack() as tf_record_close_stack, tf.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = open_sharded_output_tfrecords(tf_record_close_stack, output_path, num_shards)
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            tf.logging.info(
                'Found groundtruth annotations. Building annotations index.')
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)
        missing_annotation_count = 0
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        tf.logging.info('%d images are missing annotations.',
                        missing_annotation_count)

        total_num_annotations_skipped = 0
        for idx, image in enumerate(tqdm(images)):
            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(images))
            annotations_list = annotations_index[image['id']]
            _, tf_example, num_annotations_skipped = create_tf_example(
                image, annotations_list, image_dir, category_index, include_masks)
            total_num_annotations_skipped += num_annotations_skipped
            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        tf.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)


def _create_tf_record_from_coco_annotations(
        annotations_file, image_dir, output_path, include_masks):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    """
    with tf.gfile.GFile(annotations_file, 'r') as fid:
        output_tfrecords = tf.python_io.TFRecordWriter(output_path)
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']
        category_index = label_map_util.create_category_index(
            groundtruth_data['categories'])

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            tf.logging.info(
                'Found groundtruth annotations. Building annotations index.')
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)
        missing_annotation_count = 0
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        tf.logging.info('%d images are missing annotations.',
                        missing_annotation_count)

        total_num_annotations_skipped = 0
        for idx, image in tqdm(enumerate(images)):
            if idx % 100 == 0:
                tf.logging.info('On image %d of %d', idx, len(images))
            annotations_list = annotations_index[image['id']]
            _, tf_example, num_annotations_skipped = create_tf_example(
                image, annotations_list, image_dir, category_index, include_masks)
            total_num_annotations_skipped += num_annotations_skipped
            output_tfrecords.write(tf_example.SerializeToString())
        tf.logging.info('Finished writing, skipped %d annotations.',
                        total_num_annotations_skipped)


def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.test_image_dir, '`test_image_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    assert FLAGS.test_annotations_file, '`test_annotations_file` missing.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    testdev_output_path = os.path.join(FLAGS.output_dir, 'test.record')

    if FLAGS.shards:
        create_tf_record_from_coco_annotations = _create_tf_record_from_coco_annotations_with_shards
    else:
        create_tf_record_from_coco_annotations = _create_tf_record_from_coco_annotations

    create_tf_record_from_coco_annotations(
        FLAGS.train_annotations_file,
        FLAGS.train_image_dir,
        train_output_path,
        FLAGS.include_masks)
    create_tf_record_from_coco_annotations(
        FLAGS.test_annotations_file,
        FLAGS.test_image_dir,
        testdev_output_path,
        FLAGS.include_masks)


if __name__ == '__main__':
    FLAGS.train_image_dir = ''
    FLAGS.test_image_dir = ''
    FLAGS.test_annotations_file = ''
    FLAGS.train_annotations_file = ''
    FLAGS.output_dir = ''
    FLAGS.include_masks = True
    FLAGS.shards = False

    tfrecord_dir = ''
    images_dst = ''
    included_masks = True
    labels_to_index_map_dict = {}

    tf.app.run()

    # read_from_folder(tfrecord_files_dir, images_dst, labels_to_index_map, included_masks)

