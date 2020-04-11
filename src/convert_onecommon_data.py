# coding: utf-8
  

import os
import re
import json
import numpy as np
from PIL import Image, ImageDraw


def convert(input_dir):
    
    output_dir = os.path.join(input_dir, 'converted')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for txt_name in filter(lambda x: x.endswith('.txt'), os.listdir(input_dir)):
        file_path = os.path.join(input_dir, txt_name)
        print('converting', file_path, '...')
        convert_file(file_path, output_dir)


def convert_file(file_path, output_dir, image_size=(224, 224)):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_dir, file_name + '.json')
    image_dir = os.path.join(output_dir, 'img')
    image_path_temp = os.path.join(image_dir, file_name+'_%d.png')
    
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    
    lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    tag_re = re.compile('<(.*?)>(.*?)</.*?>')    
    
    annotations = []
    for i, line in enumerate(lines):
        tags = tag_re.findall(line)
        d = {k.strip():v.strip() for k, v in tags}
        
        image, labels = make_image_and_labels([float(_) for _ in d['input'].split()], image_size)
        
        image_path = image_path_temp%i
        image.save(image_path)
        
        annotation = {}
        annotation['selected_id'] = int(d['output'])
        annotation['dialogue'] = d['dialogue']
        annotation['image_size'] = image_size
        annotation['image_path'] = image_path
        annotation['objects'] = labels
        annotations.append(annotation)
    
    with open(output_file_path, 'w') as f:
        json.dump(annotations, f)


def make_image_and_labels(ctx_raw, img_size, min_size=0.025, max_size=0.05, min_col=0.2, max_col=0.8, img_inner_factor=4, resize_method=Image.LANCZOS):
    """
    label includes x_min, y_min, x_max, y_max, size, color
    """
    
    n_point = int(len(ctx_raw)//4)
    if isinstance(img_size, int):
        nx = ny = img_size
    else:
        nx, ny = img_size
    
    full_canvas_nx = nx*img_inner_factor
    full_canvas_ny = ny*img_inner_factor
    
    margin_x = full_canvas_nx * max_size * 0.5
    margin_y = full_canvas_ny * max_size * 0.5
    canvas_nx = full_canvas_nx - 2*margin_x
    canvas_ny = full_canvas_ny - 2*margin_y
    
    labels = []
    img = Image.new('L', (full_canvas_nx, full_canvas_ny), color='white')
    draw = ImageDraw.Draw(img)
    
    draw.ellipse((margin_x, margin_y, margin_x + canvas_nx-1, margin_y + canvas_ny-1), outline=0, width=2)
    
    for i in range(0, n_point):
        x, y, s, c = ctx_raw[4*i:4*i+4]
        
        # to draw
        center_x = margin_x + canvas_nx*(1+x)*0.5
        center_y = margin_y + canvas_ny*(1+y)*0.5
        c = min_col + (1+c)*0.5*(max_col - min_col)
        s = min_size + (1+s)*0.5*(max_size - min_size)
        hx = s*0.5*canvas_nx
        hy = s*0.5*canvas_ny
        box = (center_x-hx, center_y-hy, center_x+hx, center_y+hy)
        color = int(c*255)
        draw.ellipse(box, fill=color)
        
        # annotation
        labels.append({
            'x_min': box[0]/img_inner_factor,
            'y_min': box[1]/img_inner_factor,
            'x_max': box[2]/img_inner_factor,
            'y_max': box[3]/img_inner_factor,
            'size': (2*hx/img_inner_factor, 2*hy/img_inner_factor),
            'color': color,
        })

    img = img.resize((nx, ny), resize_method)

    return img, labels


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Make json annotations and images from text files of onecommon corpus.')
    parser.add_argument('--inputdir', '-i', type=str, required=True,
                    help='path to an input directory')
    args = parser.parse_args()
    
    convert(args.inputdir)

