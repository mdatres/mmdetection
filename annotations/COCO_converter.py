# Create COCO dataset from train.csv, validation.csv and test.csv

import csv
import json
import argparse


def convert_train(): 
    parser = argparse.ArgumentParser(description='Convert annotations form .csv to .json')
    parser.add_argument('--path_to_ann', type = str, help = 'path to .csv to convert')
    parser.add_argument('--path_to_save', type = str, help = 'path to the .json')
    args = parser.parse_args()

    data = {'images': [], 'annotations': [], 'categories': [{'id': 0, 'name': 'scafoideus_titanus'}, {'id': 1, 'name': 'planococcus_ficus_m'}]}
    

    with open(args.path_to_ann) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        id_file = 0 
        ann_id = 0
        for row in csv_reader:
            name = row[0].split('/')[-1] 
            if name not in [r['file_name'] for r in data['images']]: 
                if 'train' in args.path_to_ann:
                    data['images'].append({'file_name': str(name), 'height': 1152, 'width': 1152, 'id': id_file})
                else: 
                    data['images'].append({'file_name': str(name), 'height': 3456, 'width': 4608, 'id': id_file})
                if str(row[5])=='scafoideus_titanus': 
                    bbox_id = 0
                else: 
                    bbox_id = 1

                data['annotations'].append({'area': (float(row[3]) - float(row[1]))*(float(row[4])-float(row[2])),'image_id': id_file, 'bbox': [float(row[1]), float(row[2]), float(row[3]) - float(row[1]), float(row[4])-float(row[2])], 'category_id': bbox_id, 'id': ann_id, 'segmentation': [], 'ignore': 0, 'iscrowd': 0})
                ann_id +=1 
                id_file +=1
            else:
                if str(row[5])=='scafoideus_titanus': 
                    bbox_id = 0
                else: 
                    bbox_id = 1
                
                loc_id = 0
                for el in data['images']: 
                    if el['file_name'] == str(name): 
                        loc_id = el['id']

                data['annotations'].append({'area': (float(row[3]) - float(row[1]))*(float(row[4])-float(row[2])), 'image_id': loc_id, 'bbox': [float(row[1]), float(row[2]), float(row[3]) - float(row[1]), float(row[4])-float(row[2])], 'category_id': bbox_id, 'id': ann_id, 'segmentation': [], 'ignore': 0, 'iscrowd': 0})
                ann_id +=1 

    
    with open(args.path_to_save, 'w') as outfile:
        json.dump(data, outfile)

convert_train()


                

