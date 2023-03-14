import numpy as np
import tensorflow as tf


class TfObjectDetector(object):

    def __init__(self, frozen_graph_path, labels_path):

        from object_detection.utils import label_map_util

        self.frozen_graph_path = frozen_graph_path
        self.labels_path = labels_path

        self.detection_graph = self._build_graph()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        label_map = label_map_util.load_labelmap(self.labels_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=100, use_display_name=True)

        self.category_index = label_map_util.create_category_index(categories)

    def _build_graph(self):

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = detection_graph.as_graph_def()
            with tf.io.gfile.GFile(self.frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        return detection_graph

    def detect_many(self, images_np):

        graph = self.detection_graph
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        boxes = graph.get_tensor_by_name('detection_boxes:0')
        scores = graph.get_tensor_by_name('detection_scores:0')
        classes = graph.get_tensor_by_name('detection_classes:0')
        num_detections = graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: images_np})

        detection_results = [(boxes[i, :], scores[i, :], classes[i, :].astype(int), num_detections[i]) for i in
                             range(len(num_detections))]

        category_index = self.category_index

        return detection_results, category_index

    @staticmethod
    def filter_class_from_detection_results(detection_results, class_num):

        filtered_detection_results = []

        for detection_result in detection_results:
            class_filter = detection_result[2] == class_num
            boxes = detection_result[0][class_filter]
            scores = detection_result[1][class_filter]
            classes = detection_result[2][class_filter]
            num_detections = len(classes)

            filtered_detection_results.append((boxes, scores, classes, num_detections))

        return filtered_detection_results

    def detect(self, image_np):

        dims = len(image_np.shape)
        if dims == 2:
            # image_np_correct_dims = np.expand_dims(np.stack([image_np] * 3, axis=-1), axis=0)
            return None
        elif dims == 3:
            image_np_correct_dims = np.expand_dims(image_np, axis=0)
        elif dims == 4:
            image_np_correct_dims = image_np
        else:
            return None

        results = self.detect_many(image_np_correct_dims)

        if dims == 2:
            return results[0][0]
        elif dims == 3:
            return results[0]
        elif dims == 4:
            return results

    def draw_results(self, image_rgb, detection_result, min_thresh_percent):

        from object_detection.utils import visualization_utils as vis_util

        min_thresh_percent = int(min_thresh_percent)
        boxes, scores, classes, num_detections = detection_result
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_rgb,
            boxes,
            classes,
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=min_thresh_percent / 100,
            max_boxes_to_draw=100)

        return image_rgb

    def draw_results_many(self, images_np, detections_results, min_thresh_percent):

        return [self.draw_results(image_np, detection_result, min_thresh_percent) for
                (image_np, detection_result)
                in zip(images_np, detections_results)]
