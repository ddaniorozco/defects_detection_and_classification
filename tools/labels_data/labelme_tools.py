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
    def polygon_2_rectangle(data):

        access_shapes = data['shapes']
        for data_shapes in access_shapes:
            access_points = data_shapes['points']
            # print (access_points)

            min_val_x = min(x[0] for x in access_points)
            min_val_y = min(x[1] for x in access_points)
            max_val_x = max(x[0] for x in access_points)
            max_val_y = max(x[1] for x in access_points)
            new_points = [[min_val_x, min_val_y], [max_val_x, max_val_y]]

            # saved_points.append(new_points)

            data_shapes['points'] = new_points

        return data
