# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
import numpy as np

im =np.array(Image.open(r'C:\Users\陆建华\Pictures\IMG20170410154737(32pxX32px).jpg'))  # 打开图像，并转成灰度图像
print(shape(im))
data=im[:,:,0]
print(shape(data))
print(shape(data.reshape(1,1024)))