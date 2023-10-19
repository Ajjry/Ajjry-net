import rawpy
import numpy as np
import os
import ISP
name=os.listdir("G:/TEST/z")
name.sort(key=lambda x: int(x[5:8]))

for i in range(2):
    #if '_00_0.1s' in name[i]:

        img = rawpy.imread(os.path.join("G:/TEST/z", name[i]))
        ISP.ISP(img,i)