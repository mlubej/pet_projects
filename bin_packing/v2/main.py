import sys
import genlib
evo = genlib.Evolution(int(sys.argv[1]), npcs=13, mut_threshold=0.01, instance=sys.argv[2])
evo.darwinize()
