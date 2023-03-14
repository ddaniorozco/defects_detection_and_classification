import json
import os
from glob import glob


def append_json(jsons_2_append, jsons_appended):

    for file in jsons_2_append:
        # print(file)
        file_2 = os.path.join(jsons_appended, file.split('/')[-1])
        with open(file, 'r') as json_input:
            json_file = json.load(json_input)
            data_append = json_file["shapes"]
            data_2_change = json_file['imagePath']
        with open(file_2, 'r') as json_input_2:
            json_file_2 = json.load(json_input_2)
            json_file_2['imagePath'] = data_2_change
            data = json_file_2['shapes']
            data.extend(data_append)
            print(data_2_change)
            # print(data)

        with open(file_2, 'w') as outputfile:
            json.dump(json_file_2, outputfile, indent=2)


jsons_2_append = glob('')  # fruit
jsons_appended = ''  # defect

append_json(jsons_2_append, jsons_appended)
