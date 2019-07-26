import time
import numpy as np
import solution

import matplotlib.pyplot as plt

fblocks = np.array([
    [(0,0,0),(1,0,1),(2,0,0),(3,0,1),(4,0,0)],
    [(0,0,0),(1,0,1),(2,0,0),(2,-1,1),(1,1,0)],
    [(0,0,0),(1,0,1),(2,0,0),(0,1,1),(2,-1,1)],
    [(0,0,0),(1,0,1),(2,0,0),(-1,0,1),(0,-1,1)],
    [(0,0,0),(-1,0,1),(1,0,1),(0,1,1),(0,-1,1)],
    [(0,0,1),(1,0,0),(2,0,1),(0,1,0),(0,-1,0)],
    [(0,0,0),(1,0,1),(2,0,0),(3,0,1),(0,1,1)],
    [(0,0,0),(0,1,1),(2,0,0),(2,1,1),(1,1,0)],
    [(0,0,1),(1,0,0),(2,0,1),(0,1,0),(0,2,1)],
    [(0,0,1),(-1,0,0),(0,1,0),(1,1,1),(2,1,0)],
    [(0,0,0),(1,0,1),(0,1,1),(1,1,0),(-1,1,0)],
    [(0,0,0),(1,0,1),(0,1,1),(1,1,0)],
    [(0,0,0),(1,1,0),(2,2,0),(0,1,1),(1,2,1)]
    ])

board_size = 8

blocks = []
for block in fblocks:
    blocks.append(np.array(block))
    fblocks = np.array(blocks).copy()


start_time = time.time()

blocks = np.zeros(len(fblocks)).astype(object)
counter = 0

output, c = solution.solution(0,blocks, counter)

elapsed_time = time.time() - start_time
print(f'Elapsed time {elapsed_time:.2f} s')
print(f'Tested {c} combinations')
print(output)

dcol = np.random.rand(len(fblocks),3)

def plot_block(block, setting = [0,0,0], *args, **kwargs):
    mask = block[...,-1] == 0
    plt.plot(block[mask,0], block[mask,1], marker='s', fillstyle='top', linewidth=0, *args, **kwargs)
    plt.plot(block[~mask,0], block[~mask,1], marker='s', fillstyle='bottom', linewidth=0, *args, **kwargs)

def plot_blocks(blocks, settings = None, filename = None, *args, **kwargs):
    fig = plt.figure(figsize=(10,10))
    edges = np.array([[0,0], [0,board_size-1], [board_size-1,board_size-1], [board_size-1,0], [0,0]])
    plt.plot(edges[:,0], edges[:,1], color='r')
    for idx,block in enumerate(blocks):
        setting = [0,0,0] if settings is None else settings[idx]
        plot_block(block, setting, markersize=40, color = dcol[idx])
        plt.xlim(-1, board_size)
        plt.ylim(-1, board_size)
        
    if filename is not None:
        plt.savefig(f'{filename}', bbox_inches='tight', dpi=100)
        plt.close(fig)

plot_blocks(output, filename='solution.png')
