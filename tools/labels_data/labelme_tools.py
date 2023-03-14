import json
import os
import cv2
import base64


class ConvertersTools:

    @staticmethod
    def base64_encode(image, ext='.jpg'):
        retval, buffer = cv2.imencode(ext, image)
        if retval:
            image_str = base64.b64encode(buffer)
            return image_str
        else:
            return None

    @staticmethod
    def save_label_dict(shapes, img_file_name, img, image_height, image_width, dest_folder):

        img_b64_str = ConvertersTools.base64_encode(img, ext='.png').decode('utf-8')

        labelme_dict = {
            'version': '4.5.6',
            'flags': {},
            'shapes': shapes,
            'imagePath': img_file_name,  # The path need to be relative
            'imageData': img_b64_str,
            'imageHeight': image_height,
            'imageWidth': image_width
        }

        labelme_file_name = os.path.join(dest_folder, img_file_name.split('.')[0] + '.json')

        with open(labelme_file_name, 'w') as json_file:
            json.dump(labelme_dict, json_file, indent=2)

    @staticmethod
    def create_polygon_shape(label_name, contour_points, filter_points=1):

        points = [[x, y] for index, (x, y) in enumerate(contour_points) if label_name != 'fruit' or index % filter_points == 0]

        shape_dict = {
                        'label': label_name,
                        'points': points,
                        'group_id': None,
                        'shape_type': 'polygon',
                        'flags': {}
                      }

        return shape_dict

    @staticmethod
    def create_rectangle_shape(label_name, input_points, image_width, image_height):

        xmin, ymin, xmax, ymax = input_points
        points = [[int(round(xmax * image_width)), int(round(ymin * image_height))], [int(round(xmin * image_width)),
                                                                                      int(round(ymax * image_height))]]

        label_name = label_name.lower()

        shape_dict = {
                        'label': label_name,
                        'points': points,
                        'group_id': None,
                        'shape_type': 'rectangle',
                        'flags': {}
                      }

        return shape_dict
