import os
import random
import shutil

import json
import os
import argparse
from PIL import Image

# Define the classes and their corresponding category IDs
CLASSES = {
    0: 'Car',
    1: 'Pedestrian', 
    2: 'Cyclist', 
    3: 'Van', 
    4: 'Person_sitting', 
    5: 'Tram', 
    6: 'Truck', 
    7: 'Misc', 
    8: 'DontCare', 
    'Car': 0, 
    'Pedestrian': 1, 
    'Cyclist': 2, 
    'Van': 3, 
    'Person_sitting': 4,
    'Tram': 5, 
    'Truck': 6, 
    'Misc': 7, 
    'DontCare': 8
}
def parse_annotations(annotations_str):
    """
    Parses a string containing bounding box annotations and returns a list of annotations in COCO-format.
    """
    annotations = []
    parts = annotations_str.strip().split(' ')
    image_path = parts[0]
    bboxes = parts[1:]
    for i in range(0, len(bboxes), 5):
        x1, y1, x2, y2, class_name = bboxes[i:i+5]
        bbox = [float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)]
        category_id = CLASSES[class_name]
        annotation = {
            'bbox': bbox,
            'category_id': category_id
        }
        annotations.append(annotation)
    return image_path, annotations

def convert_to_coco(input_file, output_dir):
    """
    Converts a text file containing bounding box annotations to COCO-format label files.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line_num, line in enumerate(lines):
            try:
                image_path, annotations = parse_annotations(line)
            except Exception as e:
                print(f'Error parsing line {line_num}: {e}')
                continue
            image_name = os.path.basename(image_path)
            image_id = os.path.splitext(image_name)[0]
            image = Image.open(image_path)
            width, height = image.size
            coco_label = {
                'file_name': image_name,
                'image_id': image_id,
                'height': height,
                'width': width,
                'annotations': annotations
            }
            output_path = os.path.join(output_dir, f'{image_id}.json')
            with open(output_path, 'w') as f:
                json.dump(coco_label, f)
            print(f'Successfully converted {image_path} to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert text file of bounding box annotations to COCO-format label files')
    parser.add_argument('input_file', help='Path to input file')
    parser.add_argument('output_dir', help='Path to output directory')
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert the input file to COCO-format label files
    convert_to_coco(args.input_file, args.output_dir)