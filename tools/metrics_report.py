import json
import os
from glob import glob
import numpy as np


class MetricReport:

    def generate(self, jsons_gt, jsons_net, dest_folder):
        """
        :param dest_folder:
        :param jsons_gt: files of Ground Truth
        :param jsons_net: path files of net detection results
        :return: Values for total images and per defect for precision, recall and f1
        """
        true_positive_total = 0
        false_positive_total = 0
        false_negative_total = 0

        defects = []
        metrics_defect = []

        for defect in defects:

            true_positive_per_defect = 0
            false_positive_per_defect = 0
            false_negative_per_defect = 0

            for j_gt in jsons_gt:
                j_net = os.path.join(jsons_net, j_gt.split('/')[-1])
                with open(j_gt, 'r') as json_input_gt:
                    json_file_gt = json.load(json_input_gt)
                with open(j_net, 'r') as json_input_net:
                    json_file_net = json.load(json_input_net)

                results = self.compare_images(defect, json_file_gt, json_file_net)
                true_positive, false_positive, false_negative = results

                true_positive_per_defect += true_positive
                false_positive_per_defect += false_positive
                false_negative_per_defect += false_negative

            variables = (true_positive_per_defect, false_positive_per_defect, false_negative_per_defect)
            metrics = self.calc_precision_recall_f1(defect, variables)
            precision, recall, f1 = metrics

            true_positive_total += true_positive_per_defect
            false_positive_total += false_positive_per_defect
            false_negative_total += false_negative_per_defect

            dict_metrics_defect = {
                'Class': defect,
                'Precision': precision,
                'Recall': recall,
                'f1': f1,
                'True positive': true_positive_per_defect,
                'False positive': false_positive_per_defect,
                'False negative': false_negative_per_defect
            }

            metrics_defect.append(dict_metrics_defect)

        variables_total = (true_positive_total, false_positive_total, false_negative_total)
        metrics_total = self.calc_precision_recall_f1('total', variables_total)
        precision, recall, f1 = metrics_total
        dict_metrics_total = {
            'Metrics defects': metrics_defect,
            'Class': 'Total',
            'Precision': precision,
            'Recall': recall,
            'f1': f1,
            'True positive': true_positive_total,
            'False positive': false_positive_total,
            'False negative': false_negative_total
        }

        metrics_report_dict = os.path.join(dest_folder, 'metrics_report.json')

        with open(metrics_report_dict, 'w') as json_file:
            json.dump(dict_metrics_total, json_file, indent=2)

    def compare_images(self, defect, gt, net):

        true_positive = 0
        # false_positive = 0
        false_negative = 0

        gt_list = []
        net_list = []
        gt_shapes = gt["shapes"]
        net_shapes = net["shapes"]

        for gt_access_shapes in gt_shapes:
            if gt_access_shapes["label"] != 'fruit':
                gt_label_bbox = (gt_access_shapes["label"], gt_access_shapes["points"])
                gt_list.append(gt_label_bbox)
        for net_access_shapes in net_shapes:
            if net_access_shapes["label"] != 'fruit':
                net_label_bbox = (net_access_shapes["label"], net_access_shapes["points"])
                net_list.append(net_label_bbox)

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

        true_positive, false_positive, false_negative = variables

        try:
            precision = true_positive / (true_positive + false_positive)
        except ZeroDivisionError:
            # print
            precision = 0.0
        try:
            recall = true_positive / (true_positive + false_negative)
        except ZeroDivisionError:
            recall = 0.0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0

        print(f'For {defect.capitalize()}:\n'
              f'Precision = {precision}\n'
              f'Recall = {recall}\n'
              f'F1 = {f1}\n'
              f'True positives = {true_positive}\n'
              f'False positives = {false_positive}\n'
              f'False negatives = {false_negative}')

        return precision, recall, f1


if __name__ == '__main__':
    jsons_gt = glob('')
    jsons_dir_net = ''
    dest_folder = ''

    metrics_report = MetricReport()
    metrics_report.generate(jsons_gt, jsons_dir_net, dest_folder)
