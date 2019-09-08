import os
import time

TREE_MAX_DEPTH = [18,21,24]

for depth in TREE_MAX_DEPTH:
	os.system('python main.py '+str(depth))
	time.sleep(5)