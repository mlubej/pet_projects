import numpy as np
import matplotlib.pyplot as plt

import itertools
import time


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

def rot_mat(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.round(np.array(((c,-s), (s, c))))
    return R

def rotation(block, rot=0):
    block = block.copy()
    angles = [0,90,180,270]
    rblock = block[...,:-1]
    rblock = np.array(list(map(lambda x : np.matmul(rot_mat(angles[rot]), x), rblock))).astype(int)
    block[...,:-1] = rblock
    return block

def translation(block, vector):
    block = block.copy()
    rblock = block[...,:-1]
    rblock = np.array(list(map(lambda x : x + vector, rblock)))
    block[...,:-1] = rblock
    return block


def move_block(block, setting):
    block = rotation(block,setting[-1])
    block = translation(block,setting[:-1])
    return block

# brute force it
brute_positions_x = np.array(list(range(board_size)))
brute_positions_y = brute_positions_x
brute_rotations = np.array(list(range(4)))

iterables = [brute_positions_x, brute_positions_y, brute_rotations]

all_combs = np.array(list(itertools.product(*iterables)))

def board_check(blocks):
    rblocks = np.array([block[...,:-1] for block in blocks])
    flats = np.concatenate(rblocks)
    if any(n < 0 for n in flats.ravel()) or any(n >= board_size for n in flats.ravel()):
       return False
    return True

def duplication_check(blocks):
    rblocks = np.array([block[...,:-1] for block in blocks])
    flats = np.concatenate(rblocks)
    mat = np.zeros((board_size, board_size))
    for pos in flats:
        mat[pos[0]][pos[1]] += 1
	
    nflats = mat.ravel()
    if any(n > 1 for n in nflats.ravel()):
       return False
    return True

def single_check(case):
    if np.sum(case[:-1]) % 2 != case[-1]:
       return False
    return True


def fillstyle_check(blocks):
    flats = np.concatenate(blocks)
    if any(not single_check(case) for case in flats):
       return False
    return True

def solution(i, blocks, counter):
    for comb in all_combs:
        counter +=1
        if i < len(fblocks):
            blocks[i] = move_block(fblocks[i], comb)
            if i < len(fblocks)-1:
                if not board_check(blocks[:i+1]):
                    continue
                if not duplication_check(blocks[:i+1]):
                    continue
                if not fillstyle_check(blocks[:i+1]):
                    continue
                blocks, counter = solution(i+1, blocks, counter)
                
                if isinstance(blocks[-1], float):
                    pass
                else:
                    if not board_check(blocks):
                        continue
                    if not duplication_check(blocks):
                        continue
                    if not fillstyle_check(blocks):
                        continue
                    break
            else:
                if not board_check(blocks):
                    continue
                if not duplication_check(blocks):
                    continue
                if not fillstyle_check(blocks):
                   continue
                break
    return blocks, counter


