import networkx as nx
import sys
import numpy as np
import string
from random import sample
import torch
from torchvision.ops import box_convert
import math
import os
from PIL import Image
import json
from itertools import cycle

import matplotlib.pyplot as plt
from generate_graphs import gen_graph

def map_to_lines(d):
    lines = []
    for label, box in zip(d['labels'], d['boxes']):
        lx = box[0]
        by = box[1]
        rx = box[2]
        ty = box[3]
        if label == 38:
            lines.append([lx,by,rx,ty])
        elif label == 39:
            lines.append([rx,by,lx,ty])
        elif label == 40 or label == 41:
            lines.append([lx,by,rx,ty])
        elif label == 42 or label == 43:
            lines.append([rx,by,lx,ty])
    
    return lines

def gen_dataset(n, directory):
    os.mkdir(f'./{directory}/train')
    dir_iter = cycle([True, False])
    n_iter = cycle([i + 2 for i in range(14)])
    color_iter = cycle(['#808080', '#696969', '#778899', '#708090', '#2f4f4f', '000000'])
    f_iter = cycle([5, 6, 7])

    output = {}
    i = 0
    while i < n:
        if i % 50 == 0:
            print(i)
        dir, nodes, f, color = next(dir_iter), next(n_iter), next(f_iter), next(color_iter)
        for p in np.linspace(0.05, 1.5 / nodes, 10):
            if nodes < 7 and p < 0.15:
                continue
            # print(dir,nodes,f,p)

            out = gen_graph(nodes, p, f, color, dir, False, i, f"./{directory}/train")
            lines = map_to_lines(out)
            output[i] = lines
            i += 1

    with open(f'./{directory}/train.json', 'w') as f:
        json.dump(output, f)

def main():
    """n is first arg, directory is second"""
    gen_dataset(int(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
