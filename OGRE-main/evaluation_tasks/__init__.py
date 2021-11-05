import os
import sys
sys.path.append("..")
for d in os.listdir(".."):
    sys.path.append(os.path.join("..", d))
