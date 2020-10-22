import os
from glob import glob
import numpy as np

txts = glob(os.path.join('./gw_labels', '**/*.txt'), recursive=True)
for txt in txts:
    upper = os.path.split(txt)[0]
    oldname = os.path.split(txt)[1]
    oldframe = oldname.split('_')[0]
    newframe = oldframe.zfill(3)
    newname = oldname.replace(oldframe, newframe)

    newpath = os.path.join(upper, newname)

    os.rename(txt, newpath)

