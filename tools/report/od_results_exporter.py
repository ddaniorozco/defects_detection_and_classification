import os
import numpy as np
import cv2
import json
import glob
import base64
import argparse
from tf_object_detector import TfObjectDetector


class OdResultsExporter:

    def __init__(self, frozen_graph_path, labels_path, min_thresh_percent, images_path,
                 results_images=None, results_json=None):

        self.tf_object_detector = TfObjectDetector(frozen_graph_path, labels_path)
        self.min_thresh_percent = min_thresh_percent
        self.images_path = images_path
        self.results_json = results_json
        self.results_images = results_images
        self.chunks_size = 100
        self.image_resize = 512

    def export(self):

        images_paths_total = sorted(glob.glob(images_path, recursive=False))
        chunks_images = self.generate_chunks(images_paths_total, self.chunks_size)

        images_processed = 0

        for i, chunk_images in enumerate(chunks_images):

            images_transformation_and_detection = self.images_transformation_and_detect(chunk_images)
            images_np, detection_results, images_paths, images_np_for_detection = \
                images_transformation_and_detection

            images_processed += len(images_np)
            print(f"Processing images = {images_processed}/{len(images_paths_total)} ")

            self.tf_2_labelme_many(images_paths, images_np, detection_results, results_json)

            if results_images is not None:
                draw_results_many = self.tf_object_detector.draw_results_many(images_np,
                                                                              detection_results,
                                                                              min_thresh_percent)

                self.save_images_net(images_paths, draw_results_many, results_images)

    def images_transformation_and_detect(self, images_paths):

        images_np = [cv2.imread(file) for file in images_paths]
        images_np_for_detection = [cv2.resize(img, (self.image_resize, self.image_resize)) for img in images_np]
        images_np_for_detection = np.array(list(images_np_for_detection))
        images_np_for_detection = images_np_for_detection[..., ::-1]
        detect = self.tf_object_detector.detect_many(images_np_for_detection)
        detection_results, category_index = detect
        images_np_for_detection = images_np_for_detection[..., ::-1]
        self.category_index = category_index

        return images_np, detection_results, images_paths, images_np_for_detection

    @staticmethod
    def generate_chunks(data, batch_size):

        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def data_normalization(self, image_np, detection_result, min_thresh_percent):

        boxes, scores, classes, num_detections = detection_result

        category_index = self.category_index
        class_id_to_name_dict = {row['id']: row['name'] for row in list(category_index.values())}
        classes_labels = [class_id_to_name_dict[cls] for cls in classes]

        min_thresh_percent = int(min_thresh_percent)
        min_score_thresh = (min_thresh_percent / 100)
        count_n = np.count_nonzero(scores > min_score_thresh)
        boxes_transformed = []
        width = int(image_np.shape[1])
        height = int(image_np.shape[0])

        for bbox, score in zip(boxes.astype(float), scores):

            if score < min_score_thresh:
                pass
            else:
                bbox[0], bbox[2] = bbox[0] * height, bbox[2] * height
                bbox[1], bbox[3] = bbox[1] * width, bbox[3] * width
                box_transformed = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]

                boxes_transformed.append(box_transformed)

        return classes_labels, boxes_transformed, count_n

    def tf_2_labelme(self, images_path, image_np, detection_result, results_json, min_thresh_percent):

        data_normalization = self.data_normalization(image_np, detection_result, min_thresh_percent)

        classes_labels, boxes_transformed, count_n = data_normalization

        img_file_name = images_path.split('/')[-1]
        width = int(image_np.shape[1])
        height = int(image_np.shape[0])

        shapes = []
        for i in range(count_n):
            shape = {
                'label': str(classes_labels[i]),
                'points': boxes_transformed[i],
                'group_id': None,
                'shape_type': 'rectangle',
                'flags': {}
            }

            shapes.append(shape)

        img_b64_str = self.base64_encode(image_np, ext='.png').decode('utf-8')

        labelme_dict = {
            'version': '4.5.7',
            'flags': {},
            'shapes': shapes,
            'imagePath': img_file_name,
            'imageData': img_b64_str,
            'imageHeight': height,
            'imageWidth': width
        }

        labelme_file_name = os.path.join(results_json, img_file_name.split('.')[0] + '.json')
        with open(labelme_file_name, 'w') as json_file:
            json.dump(labelme_dict, json_file, indent=2)

        return labelme_dict

    @staticmethod
    def base64_encode(image_rgb, ext='.jpg'):
        retval, buffer = cv2.imencode(ext, image_rgb)
        if retval:
            image_str = base64.b64encode(buffer)
            return image_str

    def tf_2_labelme_many(self, images_path, images_np, detection_results, results_json):

        if results_json is not None:
            ([self.tf_2_labelme(image_path, image_np, detection_result, results_json, min_thresh_percent) for
              (image_path, image_np, detection_result) in
              zip(images_path, images_np, detection_results)])
        else:
            pass

    @staticmethod
    def save_images_net(images_paths, images_net, results_images):

        for path, img in zip(images_paths, images_net):
            file_name = os.path.join(results_images, path.split('/')[-1])
            cv2.imwrite(file_name, img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Object Detection results exporter")
    parser.add_argument('--frozen_graph_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--min_thresh_percent', type=str, required=True)
    parser.add_argument('--images_path', type=str, required=True)
    parser.add_argument('--results_json', type=str, required=False)
    parser.add_argument('--results_images', type=str, required=False)
    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()

    frozen_graph_path = args.frozen_graph_path
    labels_path = args.labels_path
    min_thresh_percent = args.min_thresh_percent
    images_path = args.images_path
    results_json = args.results_json
    results_images = args.results_images

    print(frozen_graph_path)
    print(labels_path)
    print(min_thresh_percent)
    print(images_path)
    print(results_json)
    print(results_images)
    print(os.environ)

    od_results_exporter = OdResultsExporter(frozen_graph_path, labels_path, min_thresh_percent, images_path,
                                            results_json, results_images)

    od_results_exporter.export()
