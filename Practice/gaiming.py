import os
import numpy as np

name=os.listdir("F:/eyes/data_R_L/adult/mask_R_crop")
name.sort(key=lambda x: int(x[:-4]))
for i in range(3000):
    oldname=os.path.join("F:/eyes/data_R_L/adult/mask_R_crop", name[i])
    newname=os.path.join("F:/eyes/data_R_L/adult/mask_R_crop", str(i+2745)+'.jpg')
    os.rename(oldname, newname)