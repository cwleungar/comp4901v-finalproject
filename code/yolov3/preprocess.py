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

def convert_to_coco(input_file, output_dir, image_dir):
    """
    Converts a text file containing bounding box annotations to COCO-format label files.
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
        if not lines:
            print(f'Error: input file {input_file} is empty')
            return
        for line_num, line in enumerate(lines):
            li=line.split(' ')
            imgname = li[0]
            img = Image.open(os.path.join(image_dir, imgname))
            w,h=img.size
            with open(os.path.join(output_dir, imgname.split('.')[0] + '.txt'), 'w') as f:
                for i in range(len(li)):
                    if i==0:
                        continue
                    if li[i]=='\n':
                        break
                    l=li[i].split(',')
                    x1,y1,x2,y2=float(l[0]),float(l[1]),float(l[2]),float(l[3])
                    m1,m2=(x1+x2)/2,(y1+y2)/2
                    w1,h1=x2-x1,y2-y1
                    x,y,wn,hn=m1/w,m2/h,w1/w,h1/h
                    buffer=l[4]+' '+str(x)+' '+str(y)+' '+str(wn)+' '+str(hn)
                    if i!=len(li)-1:
                        buffer+='\n'
                    f.write(buffer)
            a='/data/cwleungar/dataset/coco/images/val'
            b='/data/cwleungar/dataset/coco/labels/val'
            os.popen('mv '+os.path.join(image_dir, imgname)+' '+os.path.join(a, imgname))
            os.popen('mv '+os.path.join(output_dir, imgname.split('.')[0] + '.txt')+' '+os.path.join(b, imgname.split('.')[0] + '.txt'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert text file of bounding box annotations to COCO-format label files')
    parser.add_argument('input_file', help='Path to input file')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('image_dir', help='Path to Image directory')

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert the input file to COCO-format label files
    convert_to_coco(args.input_file, args.output_dir, args.image_dir)