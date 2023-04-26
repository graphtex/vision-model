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

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250

IMG_LEN = 924

object_to_code = {'node': 36,
                  'filled_node': 37,
                  'undirected_BL_TR': 38,
                  'undirected_BR_TL': 39,
                  'directed_BL_TR': 40,
                  'directed_TR_BL': 41,
                  'directed_BR_TL': 42,
                  'directed_TL_BR': 43, }
for i in range(48, 58):
    object_to_code[chr(i)] = i - 48
for i in range(97, 123):
    object_to_code[chr(i)] = i - 87

all_labels = list(string.ascii_lowercase) + [chr(num) for num in range(48, 58)]


def gen_dataset(start_i, n, directory):
    os.makedirs(f'./{directory}/images', exist_ok=True)
    os.makedirs(f'./{directory}/labels', exist_ok=True)
    dir_iter = cycle([True, False])
    n_iter = cycle([i + 2 for i in range(14)])
    color_iter = cycle(['#808080', '#696969', '#778899', '#708090', '#2f4f4f', '000000'])
    f_iter = cycle([5, 6, 7])

    from pympler.tracker import SummaryTracker
    tracker = SummaryTracker()
    output = {}
    i = start_i
    while i < n + start_i:
        if i % 50 == 0:
            print(i)
        dir, nodes, f, color = next(dir_iter), next(n_iter), next(f_iter), next(color_iter)
        for p in np.linspace(0.05, 1.5 / nodes, 10):
            if nodes < 7 and p < 0.15:
                continue
            # print(dir,nodes,f,p)

            gen_graph(nodes, p, f, color, dir, False, i, f"./{directory}/")
            i += 1

    tracker.print_diff()


BACKGROUND = Image.open('./paper_background.jpg')


# n = number of vertices
# p = probability of an edge
# f = font size
# directed = directed/undirected graph
# show = whether to display graph
def gen_graph(n, p, f, c, directed=False, show=False, id=0, directory=''):
    # create graph
    G = nx.fast_gnp_random_graph(n, p, directed=directed)

    # relabel nodes
    labels = dict(zip(nx.spring_layout(G).keys(), sample(all_labels, n)))
    backwards_labels = {labels[key]: key for key in labels}
    nx.relabel_nodes(G, labels, copy=False)

    # remove loops
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    # adjacency matrix
    adjacency = nx.adjacency_matrix(G).toarray()

    # coordinate dictionary
    pos = nx.spring_layout(G, k=2)

    # draw settings
    radii = list(np.arange(f + 5, f + 15))
    sample_radii = np.random.choice(radii, n)
    DRAW_OPTIONS = {
        "font_size": f,
        "font_family": 'Bradley Hand',
        "node_size": sample_radii ** 2,
        # "arrowsize": 25,
        "node_color": "#eceeeb",  # white
        "edgecolors": c,
        "font_color": c,
        "edge_color": c,
        "linewidths": 1,
        "width": 1,
    }

    # draw graph
    fig = nx.draw_networkx(G, pos, **DRAW_OPTIONS)
    ax = plt.gca()
    plt.imshow(BACKGROUND, extent=[-2, 2, -2, 2])
    ax.set_aspect('equal', adjustable='box')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.axis("off")
    plt.savefig(f'{directory}/images/{id}.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    if not show:
        plt.close(fig)

    # radii for bounding boxes
    radius = sample_radii / 100.2
    letters_radius = f / 100

    # Generate bounding box points for training data
    def format_box(x1, y1, x2, y2):
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    boxes = []
    training_labels = []
    # node boxes
    for k in pos.keys():
        i = backwards_labels[k]
        boxes.append(
            format_box(pos[k][0] - radius[i], pos[k][1] - radius[i], pos[k][0] + radius[i], pos[k][1] + radius[i]))
        training_labels.append(object_to_code['node'])

    # label boxes
    for k in pos.keys():
        boxes.append(format_box(pos[k][0] - letters_radius, pos[k][1] - letters_radius, pos[k][0] + letters_radius,
                                pos[k][1] + letters_radius))
        if k in object_to_code:
            training_labels.append(object_to_code[k])

    # edge boxes
    for i in range(len(adjacency)):
        end = i + 1
        if directed:
            end = len(adjacency[i])
        for j in range(0, end):
            letter = labels[i]
            letter_2 = labels[j]
            if adjacency[i][j] == 1:
                x1 = pos[letter][0]
                y1 = pos[letter][1]
                x2 = pos[letter_2][0]
                y2 = pos[letter_2][1]

                second_above = y2 > y1
                second_right = x2 > x1
                slope = (y2 - y1) / (x2 - x1)

                x_delta_abs1 = math.sqrt(radius[i] ** 2 / (1 + slope ** 2))
                y_delta_abs1 = abs(slope * x_delta_abs1)

                x_delta_abs2 = math.sqrt(radius[j] ** 2 / (1 + slope ** 2))
                y_delta_abs2 = abs(slope * x_delta_abs2)

                box_x1 = x1 + x_delta_abs1 if second_right else x1 - x_delta_abs1
                box_y1 = y1 + y_delta_abs1 if second_above else y1 - y_delta_abs1

                box_x2 = x2 - x_delta_abs2 if second_right else x2 + x_delta_abs2
                box_y2 = y2 - y_delta_abs2 if second_above else y2 + y_delta_abs2
                if abs(box_x2 - box_x1) < 0.1:
                    add_x = abs(0.1 - abs(box_x2 - box_x1))/2
                    box_x1 -= add_x
                    box_x2 += add_x
                if abs(box_y2 - box_y1) < 0.1:
                    add_y = abs(0.1 - abs(box_y2 - box_y1))/2
                    box_y1 -= add_y
                    box_y2 += add_y
                boxes.append(format_box(box_x1, box_y1, box_x2, box_y2))

                if not directed:
                    if slope > 0:
                        training_labels.append(object_to_code['undirected_BL_TR'])
                    elif slope <= 0:
                        training_labels.append(object_to_code['undirected_BR_TL'])
                else:
                    if second_above and second_right:
                        training_labels.append(object_to_code['directed_BL_TR'])
                    elif second_above and not second_right:
                        training_labels.append(object_to_code['directed_BR_TL'])
                    elif not second_above and second_right:
                        training_labels.append(object_to_code['directed_TL_BR'])
                    elif not second_above and not second_right:
                        training_labels.append(object_to_code['directed_TR_BL'])

    out = {"labels": torch.Tensor([training_labels]), "boxes": torch.Tensor(np.array(boxes))}

    shape = [3, 1, 1]
    # Transform coordinates
    for i in range(len(out["boxes"])):
        out["boxes"][i][0] = shape[2] / 2 + out["boxes"][i][0] * shape[2] / 4
        out["boxes"][i][2] = shape[2] / 2 + out["boxes"][i][2] * shape[2] / 4
        temp = shape[1] / 2 - out["boxes"][i][1] * shape[1] / 4
        out["boxes"][i][1] = shape[1] / 2 - out["boxes"][i][3] * shape[1] / 4
        out["boxes"][i][3] = temp

    out['labels'] = [int(i) for i in out['labels'][0]]

    out['boxes'] = out['boxes'].view((-1, 4))
    out['boxes'] = box_convert(out['boxes'], 'xyxy', 'cxcywh')

    boxes = []
    for i in range(len(out['boxes'])):
        boxes.append([float(i) for i in out['boxes'][i]])

    out['boxes'] = boxes

    with open(f'{directory}/labels/{id}.txt', 'w+') as f:
        for i in range(len(out['labels'])):
            f.write('%u %f %f %f %f\n' % (out['labels'][i], out['boxes'][i][0], out['boxes'][i][1], out['boxes'][i][2], out['boxes'][i][3]))


def main():
    """n is first arg, directory is second"""
    gen_dataset(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])


if __name__ == "__main__":
    main()
