import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

for im in glob.glob("./img/*.pgm"):
	img = cv2.imread(im,0)
	print(im)
	ret,thresh1 = cv2.threshold(img,160,255,cv2.THRESH_BINARY)
	#ret,thresh2 = cv2.threshold(img,160,255,cv2.THRESH_BINARY_INV)
	#ret,thresh3 = cv2.threshold(img,180,255,cv2.THRESH_TRUNC)
	#ret,thresh4 = cv2.threshold(img,180,255,cv2.THRESH_TOZERO)
	#ret,thresh5 = cv2.threshold(img,180,255,cv2.THRESH_TOZERO_INV)

	#titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
	#images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
	cv2.imwrite("./thresh_img/"+str(im),thresh1)
'''for i in xrange(6):
	    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
	    plt.title(titles[i])
	    plt.xticks([]),plt.yticks([])

	plt.show()
'''
