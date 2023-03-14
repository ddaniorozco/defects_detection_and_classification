import json
import os
from glob import glob
import numpy as np
import argparse


class MetricReport:

    def __init__(self, jsons_gt, jsons_net, dest_folder):

        self.jsons_gt = jsons_gt
        self.jsons_net = jsons_net
        self.dest_folder = dest_folder

    def generate_report(self):

        dict_metrics_total = self.generate_results()
        self.save_to_json(dict_metrics_total)

    def generate_results(self):

        jsons_net = glob(self.jsons_net)

        true_positive_total = 0
        false_positive_total = 0
        false_negative_total = 0
        true_positive_total_fruit = 0
        false_positive_total_fruit = 0
        false_negative_total_fruit = 0
        fruit_number_total = 0
        fruit_with_defect_total = 0
        fruit_with_defect_total_gt = 0
        fruit_no_defect_total = 0
        fruit_no_defect_total_gt = 0

        defects = []

        metrics_defect = []

        for defect in defects:

            true_positive_per_defect = 0
            false_positive_per_defect = 0
            false_negative_per_defect = 0
            true_positive_per_defect_fruit = 0
            false_positive_per_defect_fruit = 0
            false_negative_per_defect_fruit = 0
            fruit_number_per_defect = 0
            fruit_with_defect_per_defect = 0
            fruit_with_defect_per_defect_gt = 0
            fruit_no_defect_per_defect = 0
            fruit_no_defect_per_defect_gt = 0

            for j_net in jsons_net:
                j_gt = os.path.join(self.jsons_gt, j_net.split('/')[-1])
                with open(j_net, 'r') as json_input_net:
                    json_net = json.load(json_input_net)
                with open(j_gt, 'r') as json_input_gt:
                    json_gt = json.load(json_input_gt)

                results_defects = self.compare_images(defect, json_gt, json_net)
                true_positive, false_positive, false_negative = results_defects
                true_positive_per_defect += true_positive
                false_positive_per_defect += false_positive
                false_negative_per_defect += false_negative

                results_fruits_defects = self.defect_inside_fruit(defect, json_gt, json_net)

                fruit_number, fruit_with_defect_net, fruit_with_defect_gt, true_positive_fruit, false_positive_fruit, \
                    false_negative_fruit = results_fruits_defects

                fruit_number_per_defect += fruit_number
                fruit_with_defect_per_defect += fruit_with_defect_net
                fruit_with_defect_per_defect_gt += fruit_with_defect_gt
                fruit_no_defect_per_defect = fruit_number_per_defect - fruit_with_defect_per_defect
                fruit_no_defect_per_defect_gt = fruit_number_per_defect - fruit_with_defect_per_defect_gt
                true_positive_per_defect_fruit += true_positive_fruit
                false_positive_per_defect_fruit += false_positive_fruit
                false_negative_per_defect_fruit += false_negative_fruit

            variables = (true_positive_per_defect, false_positive_per_defect, false_negative_per_defect,
                         true_positive_per_defect_fruit, false_negative_per_defect_fruit,
                         false_positive_per_defect_fruit)
            metrics = self.calc_precision_recall_f1(defect, variables)
            precision, recall, f1, precision_fruit, recall_fruit, f1_fruit = metrics

            true_positive_total += true_positive_per_defect
            false_positive_total += false_positive_per_defect
            false_negative_total += false_negative_per_defect
            true_positive_total_fruit += true_positive_per_defect_fruit
            false_positive_total_fruit += false_positive_per_defect_fruit
            false_negative_total_fruit += false_negative_per_defect_fruit
            fruit_number_total += fruit_number_per_defect
            fruit_with_defect_total += fruit_with_defect_per_defect
            fruit_no_defect_total += fruit_no_defect_per_defect
            fruit_with_defect_total_gt += fruit_with_defect_per_defect_gt
            fruit_no_defect_total_gt += fruit_no_defect_per_defect_gt

            dict_metrics_defect = {
                'Class': defect,
                'Precision': precision,
                'Recall': recall,
                'f1': f1,
                'True positive': true_positive_per_defect,
                'False positive': false_positive_per_defect,
                'False negative': false_negative_per_defect,
                'Precision fruit': precision_fruit,
                'Recall fruit': recall_fruit,
                'f1 fruit': f1_fruit,
                'True positive fruit': true_positive_per_defect_fruit,
                'False positive fruit': false_positive_per_defect_fruit,
                'False negative fruit': false_negative_per_defect_fruit,
                'Fruits number': fruit_number_per_defect,
                'Fruit with defect net': fruit_with_defect_per_defect,
                'Fruit without defect net': fruit_no_defect_per_defect,
                'Fruit with defect gt': fruit_with_defect_per_defect_gt,
                'Fruit without defect gt': fruit_no_defect_per_defect_gt
            }

            metrics_defect.append(dict_metrics_defect)

        variables_total = (true_positive_total, false_positive_total, false_negative_total, true_positive_total_fruit,
                           false_positive_total_fruit, false_negative_total_fruit)

        metrics_total = self.calc_precision_recall_f1('total', variables_total)
        precision, recall, f1, precision_fruit, recall_fruit, f1_fruit = metrics_total

        fruit_number_total_norm = int(fruit_number_total / len(defects))
        fruit_no_defect_total_norm = int(fruit_number_total_norm - fruit_with_defect_total)
        fruit_no_defect_total_norm_gt = int(fruit_number_total_norm - fruit_with_defect_total_gt)
        if fruit_no_defect_total_norm_gt < 0:
            fruit_no_defect_total_norm_gt = 0

        dict_metrics_total = {
            'input_data': {
                'gt_folder': self.jsons_gt,
                'net_folder': self.jsons_net
            },
            'Results': {
                'Metrics defects': metrics_defect,
                'Class': 'Total',
                'Precision': precision,
                'Recall': recall,
                'f1': f1,
                'True positive': true_positive_total,
                'False positive': false_positive_total,
                'False negative': false_negative_total,
                'Precision fruit': precision_fruit,
                'Recall fruit': recall_fruit,
                'f1 fruit': f1_fruit,
                'True positive fruit': true_positive_total_fruit,
                'False positive fruit': false_positive_total_fruit,
                'False negative fruit': false_negative_total_fruit,
                'Fruits number': fruit_number_total_norm,
                'Fruit with defect net': fruit_with_defect_total,
                'Fruit without defect net': fruit_no_defect_total_norm,
                'Fruit with defect gt': fruit_with_defect_total_gt,
                'Fruit without defect gt': fruit_no_defect_total_norm_gt
            }
        }

        return dict_metrics_total

    def save_to_json(self, dict_metrics_total):

        metrics_report_dict = os.path.join(self.dest_folder, 'metrics_report.json')
        with open(metrics_report_dict, 'w') as json_file:
            json.dump(dict_metrics_total, json_file, indent=2)

    def compare_images(self, defect, json_gt, json_net):

        true_positive = 0
        # false_positive = 0
        false_negative = 0
        gt_shapes = json_gt["shapes"]
        net_shapes = json_net["shapes"]

        gt_list = [(gt_access_shapes["label"], gt_access_shapes["points"]) for gt_access_shapes in gt_shapes if
                   gt_access_shapes["label"] != 'fruit']

        net_list = [(net_access_shapes["label"], net_access_shapes["points"]) for net_access_shapes in net_shapes if
                    net_access_shapes["label"] != 'fruit']

        for gt_label_bbox in gt_list:
            if gt_label_bbox[0] == defect:
                twin_found = False
                gt_bbox = gt_label_bbox[1]
                for net_label_bbox in net_list:
                    if net_label_bbox[0] == defect:
                        net_bbox = net_label_bbox[1]
                        calc_iou = self.calc_iou(gt_bbox, net_bbox)
                        if calc_iou > 0.3 and gt_label_bbox[0] == net_label_bbox[0]:
                            twin_found = True
                            break
                if twin_found:
                    true_positive += 1
                else:
                    false_negative += 1

        count_bboxes_net = 0
        for net_label_bbox in net_list:
            if net_label_bbox[0] == defect:
                count_bboxes_net += 1

        false_positive = count_bboxes_net - true_positive

        return true_positive, false_positive, false_negative

    def defect_inside_fruit(self, defect, json_gt, json_net):

        true_positive = 0
        false_positive = 0
        false_negative = 0
        list_fruits = []
        list_defects_net = []
        n_fruits = 0
        n_fruits_with_defects_net = 0
        n_fruits_with_defects_gt = 0

        net_shapes = json_net["shapes"]
        gt_shapes = json_gt["shapes"]

        list_defects_gt = [(gt_access_shapes["label"], gt_access_shapes["points"]) for gt_access_shapes in gt_shapes if
                           gt_access_shapes["label"] == defect]

        net_list = [(net_access_shapes["label"], net_access_shapes["points"]) for net_access_shapes in net_shapes]

        for net_label_bbox in net_list:
            if net_label_bbox[0] == 'fruit':
                list_fruits.append(net_label_bbox)
                n_fruits += 1
            elif net_label_bbox[0] == defect:
                list_defects_net.append(net_label_bbox)

        for fruit_bbox in list_fruits:
            enlarged_fruit_bbox = self.enlarge_bbox(fruit_bbox[1])
            (x_min_fruit, y_min_fruit), (x_max_fruit, y_max_fruit) = enlarged_fruit_bbox
            fruit_with_defect_net = False
            fruit_with_defect_gt = False
            for defect_bbox_net in list_defects_net:
                (x_min_defect_net, y_min_defect_net), (x_max_defect_net, y_max_defect_net) = defect_bbox_net[1]

        # Checking that the defect is inside the fruit
                if x_min_fruit < x_min_defect_net and y_max_fruit > y_max_defect_net and \
                        x_max_fruit > x_max_defect_net and y_min_fruit < y_min_defect_net:
                    fruit_with_defect_net = True
                if fruit_with_defect_net:
                    n_fruits_with_defects_net += 1
                    break
            for defect_bbox_gt in list_defects_gt:
                (x_min_defect_gt, y_min_defect_gt), (x_max_defect_gt, y_max_defect_gt) = defect_bbox_gt[1]
                if x_min_fruit < x_min_defect_gt and y_max_fruit > y_max_defect_gt and \
                        x_max_fruit > x_max_defect_gt and y_min_fruit < y_min_defect_gt:
                    fruit_with_defect_gt = True
                if fruit_with_defect_gt:
                    n_fruits_with_defects_gt += 1
                    break
            if fruit_with_defect_net is True and fruit_with_defect_gt is True:
                true_positive += 1
            elif fruit_with_defect_net is True and fruit_with_defect_gt is False:
                false_positive += 1
            elif fruit_with_defect_gt is True and fruit_with_defect_net is False:
                false_negative += 1

        return n_fruits, n_fruits_with_defects_net, n_fruits_with_defects_gt, true_positive, false_positive, \
            false_negative

    @staticmethod
    def enlarge_bbox(bbox):

        (x_min, y_min), (x_max, y_max) = bbox
        height = (y_max - y_min) * 0.05
        new_y_max = y_max + height
        new_y_min = y_min - height
        width = (x_max - x_min) * 0.05
        new_x_max = x_max + width
        new_x_min = x_min - width

        enlarged_bbox = [[new_x_min, new_y_min], [new_x_max, new_y_max]]

        return enlarged_bbox

    @staticmethod
    def calc_iou(gt_bbox, net_bbox):

        (x_topleft_gt, y_topleft_gt), (x_bottomright_gt, y_bottomright_gt) = gt_bbox
        (x_topleft_net, y_topleft_net), (x_bottomright_net, y_bottomright_net) = net_bbox

        xmin_inter = np.max([x_topleft_gt, x_topleft_net])
        ymin_inter = np.max([y_topleft_gt, y_topleft_net])
        xmax_inter = np.min([x_bottomright_gt, x_bottomright_net])
        ymax_inter = np.min([y_bottomright_gt, y_bottomright_net])

        inter_area = max(0, xmax_inter - xmin_inter + 1) * max(0, ymax_inter - ymin_inter + 1)

        actual_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
        pred_area = (x_bottomright_net - x_topleft_net + 1) * (y_bottomright_net - y_topleft_net + 1)

        iou = inter_area / float(actual_area + pred_area - inter_area)

        return iou

    @staticmethod
    def calc_precision_recall_f1(defect, variables):

        true_positive, false_positive, false_negative, true_positive_fruit, false_positive_fruit, false_negative_fruit \
            = variables

        try:
            precision = true_positive / (true_positive + false_positive)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = true_positive / (true_positive + false_negative)
        except ZeroDivisionError:
            recall = 0.0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        try:
            precision_fruit = true_positive_fruit / (true_positive_fruit + false_positive_fruit)
        except ZeroDivisionError:
            precision_fruit = 0.0
        try:
            recall_fruit = true_positive_fruit / (true_positive_fruit + false_negative_fruit)
        except ZeroDivisionError:
            recall_fruit = 0.0
        try:
            f1_fruit = 2 * (precision_fruit * recall_fruit) / (precision_fruit + recall_fruit)
        except ZeroDivisionError:
            f1_fruit = 0.0

        print(f'For {defect.capitalize()}:\n'
              f'Precision = {precision}\n'
              f'Recall = {recall}\n'
              f'F1 = {f1}\n'
              f'True positives = {true_positive}\n'
              f'False positives = {false_positive}\n'
              f'False negatives = {false_negative}\n'
              f'Precision fruit = {precision_fruit}\n'
              f'Recall fruit = {recall_fruit}\n'
              f'F1 fruit = {f1_fruit}\n'
              f'True positives fruit = {true_positive_fruit}\n'
              f'False positives fruit = {false_positive_fruit}\n'
              f'False negatives fruit = {false_negative_fruit}\n'
              )

        return precision, recall, f1, precision_fruit, recall_fruit, f1_fruit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Metrics report exporter")
    parser.add_argument('--json_folder_path_gt', type=str, required=True)
    parser.add_argument('--json_folder_path_net', type=str, required=True)
    parser.add_argument('--output_folder_path', type=str, required=True)
    args, unknown_args = parser.parse_known_args()

    json_folder_path_gt = args.json_folder_path_gt
    json_folder_path_net = args.json_folder_path_net
    output_folder_path = args.output_folder_path

    metrics_report = MetricReport(json_folder_path_gt, json_folder_path_net, output_folder_path)
    metrics_report.generate_report()
