from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import Image
import re
import numpy
import os
from random import randint

f = open("rtc.txt", 'r')
max_right=0;
min_right=1025
max_top=0;
min_top=1025;
max_radius=0;
radius=125;
i = 0
bad = [30 , 72 , 91 , 99 , 104 , 110 , 126 , 130 , 186 , 211 , 236 , 271 , 267 , 270 , 290 , 312]
for line in f:
  # The first value in the line is the database ID
  # some values are duplicated so we have to use this as the key
  image_num = int(line.split()[0].replace("mdb", "").replace(".pgm", ""))
  if image_num in bad :
    continue
  image_name = line.split()[0] 
  a =    int(line.split()[1])
  b=    int(line.split()[2])
  c=125
  



  im = Image.open("./images/"+image_name)
  
  #print(im.size)
  left = a-c
  top =1024-b-c
  Right = a+c
  bottom = 1024-b+c
  #print(image_name,left,Right,top,bottom)
  image=im.crop((left, top, Right, bottom))
  ninety=image.rotate(90)
  oneeighty=image.rotate(180)
  twoseventy=image.rotate(270)
  image.save("./img/" + str(i) + ".pgm")
  i+=1
  ninety.save("./img/" + str(i)+ ".pgm")
  i+=1;
  oneeighty.save("./img/"+ str(i)+ ".pgm")
  i+=1
  twoseventy.save("./img/"+ str(i)+ ".pgm")
  i+=1
