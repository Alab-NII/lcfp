# coding: utf-8
  

import numpy as np
from PIL import Image, ImageDraw


def is_cos_area(x, y):
    return (y - x)*(y + x) < 0


def make_view_and_label_annot(ctx_raw, ctx_label, img_size, min_size=0.025, max_size=0.05, min_col=0.2, max_col=0.8, img_inner_factor=4):

    n_point = int(len(ctx_raw)//4)
    if isinstance(img_size, int):
        nx = ny = img_size
    else:
        nx, ny = img_size
    
    edge_margin = 1 - max_size

    annot = np.zeros((n_point, ny, nx), dtype=np.int)
    img = Image.new('L', (nx*img_inner_factor, ny*img_inner_factor), color='white')
    d = ImageDraw.Draw(img)
    
    d.ellipse((0, 0, nx*img_inner_factor-1, ny*img_inner_factor-1), outline=0, width=2)

    for i in range(0, n_point):
        x, y, s, c = ctx_raw[4*i:4*i+4]

        factor = edge_margin
        if is_cos_area(x, y):
            factor = abs(x)/((x**2+y**2)**0.5)
        else:
            factor = abs(y)/((x**2+y**2)**0.5)
        x *= factor
        y *= factor

        ix = int(round((nx-1)*(1+x)*0.5))
        iy = int(round((ny-1)*(1+y)*0.5))
        x = (nx-1)*(1+x)*0.5*img_inner_factor
        y = (ny-1)*(1+y)*0.5*img_inner_factor
        
        c = min_col + (1+c)*0.5*(max_col-min_col)
        s = min_size + (1+s)*0.5*(max_size-min_size)
        hx = nx*s*0.5*img_inner_factor
        hy = ny*s*0.5*img_inner_factor
        d.ellipse((x-hx, y-hy, x+hx, y+hy), fill=int(c*255))
        annot[i, iy, ix] = 1

    img = np.asarray(img.resize((nx, ny), Image.LANCZOS), dtype=np.float32)/255

    return img, annot


def add_noise(instance):
    ctx_raw = instance['ctx_raw']           
    ctx_words = instance['words']       
    ctx_label = instance['label']
    
    n_point = int(len(ctx_raw)//4)
    ids = list(range(n_point))
    np.random.shuffle(ids)
    
    ctx_raw_new = [None]*len(ctx_raw)
    for i in range(0, n_point):
        for j in range(0, 4):
            ctx_raw_new[4*ids[i]+j] = ctx_raw[4*i+j]
    
    return {
            'ctx_raw': ctx_raw_new,
            'words': ctx_words,
            'label': ids[ctx_label],
        }

