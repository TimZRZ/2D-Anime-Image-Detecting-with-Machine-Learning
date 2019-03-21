import sys
import os
import time
import random
 
import numpy as np
import tensorflow as tf
 
from PIL import Image
import fileinput

dic = []
for line in fileinput.input('dic.txt'):
    dic.append(str(line).strip('\n'))

SIZE = 7500
WIDTH = 100
HEIGHT = 75
NUM_CLASSES = len(dic)

num = 0

for i in range(0,NUM_CLASSES):
    dir = 'anime_images/%s/' % dic[i]  
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            try:
                img = Image.open(filename)
                img.close()
                num += 1
                if num > 1000:
                    os.remove(filename)
            except:
                os.remove(filename)
                print(filename)
        num = 0