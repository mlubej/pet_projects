import sys
import genlib
import time

start = time.time()
evo = genlib.Evolution(int(sys.argv[1]), npcs=13, mut_threshold=0.01, instance=sys.argv[2], fdir='./')
evo.darwinize()
duration = time.time() - start
print(f'Process took {duration} s.')
print(f'Best chromosome: {evo.get_best_candidate().chromosome}')
