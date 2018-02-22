import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from scipy.spatial.distance import hamming
from sklearn.neighbors import BallTree
from skimage.transform import rescale
from frame_hash import FrameHash
from pyLemonFlow import Graph
from sortedcontainers import SortedSet, SortedDict
from itertools import count
from time import perf_counter as pc
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.util import view_as_blocks
from scipy.ndimage import correlate1d
from scipy.spatial.distance import hamming
import glob

TOP_K = 10      # For Temporal-Matching through Min-cost-flow


""" CODE
    ----
"""

def image_hash(img):
    """ BLOCK-HISTOGRAM based perceptual-hashing """
    # CONSTANTS
    resize_constants = (512, 384)
    block_constants = (32, 32)

    # RESIZING
    img_resized = resize(img, resize_constants)

    # GRAYSCALE
    img_resized_gray = rgb2gray(img_resized)

    # BLOCKS
    img_blocks = view_as_blocks(img_resized_gray, block_shape=block_constants)
    img_block_luminances = np.mean(img_blocks, (2,3))

    # DIFFERENCE BETWEEN ROWS
    bla = correlate1d(img_block_luminances, [-1, 1], origin = 0, axis=1)
    bits = bla[1:, 1:] >= 0.0

    return bits.flatten()

def hash_frame(frame):
    f_hasher = FrameHash(frame)
    f_hasher.calculate_hash()
    return f_hasher.get_result()

def prepare_video_new(folder, o):
    """ Uses ffmpeg's thumbnail-filter based extraction """
    hashes, imgs = [], []

    images = sorted(glob.glob(folder + '*.jpg'))
    for ind, f in enumerate(images):
        hash_A = (np.fromstring(hash_frame(imread(f)).bin,'u1') - ord('0')).astype(bool)
        hash_B = image_hash(imread(f))

        hashes.append(np.hstack((hash_A, hash_B)))
        imgs.append(rescale(imread(f), 0.1, mode='reflect'))

    with open(o + '_hash.pkl', 'wb') as f:
        pickle.dump(np.array(hashes), f)
    with open(o + '_imgs.pkl', 'wb') as f:
        pickle.dump(np.array(imgs), f)

def prepare_all(a, b):
    prepare_video_new(a, 'A')
    prepare_video_new(b, 'B')

def load_all():
    with open('A_hash.pkl', 'rb') as f:
        A = pickle.load(f)
    with open('B_hash.pkl', 'rb') as f:
        B = pickle.load(f)
    with open('A_imgs.pkl', 'rb') as f:
        A_orig = pickle.load(f)
    with open('B_imgs.pkl', 'rb') as f:
        B_orig = pickle.load(f)
    return A, B, A_orig, B_orig

def topk(A, B, A_orig=None, B_orig=None):
    """ Assumption: A is anchor -> search in B """
    print('topk')
    start_time = pc()
    N_A = A.shape[0]
    tree = BallTree(B, metric=hamming)
    top_k_scores = []
    top_k_pos = []
    for a in range(N_A):
        score, pos = tree.query([A[a]], k=TOP_K)
        top_k_scores.append(score[0])
        top_k_pos.append(pos[0])

    print(' used secs: ', pc()-start_time)

    return 1.0 - np.array(top_k_scores), np.array(top_k_pos)

def build_network_solve_lemon(A, B, top_k_scores, top_k_pos):
    # TODO lower-bound constant 0.65 is hidden; scaling-params too
    print('Build network')
    start_time = pc()
    N_A = A.shape[0]

    nodepair2nodeid = SortedDict()
    nodeid2nodepair = SortedDict()
    nodeid2nodepair[0] = 'source'
    nodeid2nodepair[1] = 'sink'

    id_gen = count(2)

    G = Graph()
    G.add_node(0)  # source
    G.add_node(1)  # sink
    G.add_arc(0, 1, 1, 0)  # weight = 0

    for i in range(N_A):
        for k_i in range(TOP_K):
            t_index = top_k_pos[i, k_i]

            id_ = None
            if (i, k_i) not in nodepair2nodeid:
                id_ = next(id_gen)
                nodepair2nodeid[(i, k_i)] = id_
                nodeid2nodepair[id_] = (i, k_i)
                G.add_node(id_)
            else:
                id_ = nodepair2nodeid[(i, k_i)]

            if top_k_scores[i, k_i] > 0.65:
                G.add_arc(0, id_, 1, -int(top_k_scores[i, k_i]**2*100.))

            G.add_arc(id_, 1, 1, 0)

            for j in range(N_A):
                if j != i:
                    for k_j in range(TOP_K):
                        id_2 = None
                        if (j, k_j) not in nodepair2nodeid:
                            id_2 = next(id_gen)
                            nodepair2nodeid[(j, k_j)] = id_2
                            nodeid2nodepair[id_2] = (j, k_j)
                            G.add_node(id_2)
                        else:
                            id_2 = nodepair2nodeid[(j, k_j)]

                        t_index_j = top_k_pos[j, k_j]
                        if t_index_j < t_index:
                            if top_k_scores[i, k_i] > 0.65:
                                G.add_arc(id_2, id_, 1, -int(top_k_scores[i, k_i]**2*100.))

    print(' used secs: ', pc() - start_time)

    print('Solve flow problem')
    start_time = pc()
    cost, path = G.min_cost_max_flow(0, 1)
    print(' used secs: ', pc() - start_time)
    print(cost)
    path = list(map(lambda x: nodeid2nodepair[x], path))
    print('processed path')
    return path

def build_network_solve(A, B, top_k_scores, top_k_pos):
    print('build network')
    N_A = A.shape[0]

    G = nx.DiGraph()

    G.add_edge((200000,200000), (100000, 100000))  # source to sink

    for i in range(N_A):
        for k_i in range(TOP_K):
            t_index = top_k_pos[i, k_i]

            if top_k_scores[i, k_i] > 0.7:
                G.add_edge((200000,200000),  # source
                       (i, k_i),
                       weight=-int(top_k_scores[i, k_i]*100.), capacity=1)

            G.add_edge((i, k_i),
                       (100000,100000),  # sink
                       weight=0, capacity=1)

            for j in range(N_A):
                if j != i:  # BROKEN?
                    for k_j in range(TOP_K):
                        t_index_j = top_k_pos[j, k_j]
                        if t_index_j < t_index:
                            if top_k_scores[i, k_i] > 0.7:
                                G.add_edge((j, k_j),
                                        (i, k_i),
                                        weight=-int(top_k_scores[i, k_i]*100.), capacity=1)

    print('solve maxflow lemon')
    import pylemonflow as plf
    solution = plf.min_cost_max_flow(list(G.nodes()),
                                 list(G.edges()),
                                 list(nx.get_edge_attributes(G,'capacity').values()),
                                 list(nx.get_edge_attributes(G,'weight').values()),
                                 (200000,200000), (100000,100000))

    my_nodes = list(G.nodes())
    solution = list(map(lambda x: my_nodes[x], solution))
    return solution


""" RUN
    ---
"""

# ASSUMES: ffmpeg thumbnail-extraction already done into these two folders
prepare_all("DATA/A/",
            "DATA/B/")

B, A, B_orig, A_orig = load_all()  # TODO swapping-logic
print(A.shape, B.shape)
top_k_scores, top_k_pos = topk(A, B, None, None)#A_orig, B_orig)
path = build_network_solve_lemon(A, B, top_k_scores, top_k_pos)

""" Heatmap-like visualization """
# Get frame scores
n_selected_frames = len(path)-2
n_scores = np.zeros(A.shape[0])

for ind, i in enumerate(path[1:-1]):
    x, y = i
    selected = top_k_pos[x, y]
    A_index = x
    B_index = selected
    score = 1.0 - hamming(A[A_index], B[B_index])
    n_scores[A_index] = score

plt.imshow(n_scores[np.newaxis], aspect = "auto", cmap="viridis")
ax = plt.gca()

plt.colorbar()

from matplotlib.ticker import FuncFormatter, MultipleLocator

def millions(x, pos):
    s = x * 0.5 * 2
    minutes = int(s / 60)
    s -= minutes * 60
    return str(minutes) + ':' + str(int(s))

loc = MultipleLocator(base=180.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)

formatter = FuncFormatter(millions)
ax.xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.show()

""" Display mapping """
A_n = A.shape[0]
B_n = B.shape[0]
max_ = max(A_n, B_n)

plt.rcParams["figure.figsize"] = (300,5)
f, arr = plt.subplots(3, max_, sharex=True, sharey=True)

for i in range(min(max_, A_n)):
    arr[0, i].imshow(A_orig[i], aspect='auto')

for i in range(min(max_, B_n)):
    arr[2, i].imshow(B_orig[i], aspect='auto')

for ind, i in enumerate(path[1:-1]):
    x, y = i
    selected = top_k_pos[x, y]
    arr[1, x].imshow(B_orig[selected], aspect='auto')

f.tight_layout()
plt.savefig('output.jpg')
plt.show()
