from __future__ import division
import cv2
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load images
def image_make(file_name):
	folder='uploads'

	a = cv2.imread(os.path.join(folder,file_name))

	dataset = pd.read_csv("garlic_varieties_dataset.csv")
	cols = dataset.columns.values
	X = dataset[cols[1:14]] #starts with 0
	Y = dataset[['varieties']]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 25)
	Y_train.describe()
	Y_test.describe()

	# Create the classifier
	decision_tree_classifier = DecisionTreeClassifier(random_state = 0)

	# Train the classifier on the training set
	decision_tree_classifier.fit(X_train, Y_train)

	clone = a.copy()
	clone1 = a.copy()

	## Convert to Grayscale, HSV
	gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(clone, cv2.COLOR_BGR2HSV) #stats for color based features

	#increase the contrast (.1)
	alpha = 1.1
	beta = 0
	con = cv2.convertScaleAbs(clone1, alpha=alpha, beta=beta)

	#Watershed Algorithm
	blur = cv2.GaussianBlur(gray, (55,55),0) 
	ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	#noise removal
	kernel = np.ones((5,5),np.uint8) # (5, top) (3, bot,sides) = 5
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
	closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 2)
	#sure background area
	sure_bg = cv2.dilate(opening, kernel, iterations=3)

	#Get contour points		
	contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contour_list = []
	for contour in contours:
		area = cv2.contourArea(contour)
		if (area >= 100000 and area <= 900000): # (4-9, top) (1-9, bot,sides)
			imgfinal = None
			contour_list.append(contour)
			lst_bgr = []
			lst_hsv = []
			lst_gray = []
			lst_w = []

			for i in range(len(contour_list)):
				cimg = np.zeros((clone.shape[0],clone.shape[1]), np.uint8)
				cv2.drawContours(cimg, contour_list, i, color=255, thickness=-1)
				#Final Segmentation Result
				mask_rgb1 = cv2.cvtColor(cimg, cv2.COLOR_GRAY2RGB)
				maskBgr = cv2.bitwise_and(clone, mask_rgb1) # BGR
				maskCon = cv2.bitwise_and(con, mask_rgb1) # Contrast


				#Getting ready for Feature Extraction
				## Append all Image pixels
				indices = np.where(cimg == 255)
				height = indices[0]
				width = indices[1]

				## Grayscale ARRAY
				lst_gray.append(gray[height,width])
				arr_gray = lst_gray[i]
				garlic = len(arr_gray) # Get Garlic Length HERE
				## RGB ARRAY
				lst_bgr.append(a[height,width])
				arr_bgr = lst_bgr[i]
				## HSV ARRAY
				lst_hsv.append(hsv[height,width])
				arr_hsv = lst_hsv[i]

				# Color Based Features
				# Color Mean and Standard Deviation of BGR, HSV
				## RGB
				B, G, R = arr_bgr[:,0], arr_bgr[:,1], arr_bgr[:,2]
				mean_b = np.mean(B).flatten()
				mean_g = np.mean(G).flatten()
				mean_r = np.mean(R).flatten()
				std_b = np.std(B).flatten()
				std_g = np.std(G).flatten()
				std_r = np.std(R).flatten()
				## HSV
				H, S, V = arr_hsv[:,0], arr_hsv[:,1], arr_hsv[:,2]
				mean_h = np.mean(H).flatten()
				mean_s = np.mean(S).flatten()
				mean_v = np.mean(V).flatten()
				std_h = np.std(H).flatten()
				std_s = np.std(S).flatten()
				std_v = np.std(V).flatten()
				# Raw Feature Vector
				## RGB
				raw_b = B.flatten()
				raw_g = G.flatten()
				raw_r = R.flatten()
				## HSV
				raw_h = H.flatten()
				raw_s = S.flatten()
				raw_v = V.flatten()

				## Ratio of BGR, Contrast
				# Detects the Shade Pixels Here
				bgr1 = [240,255,255]
				bgr2 = [245,245,245]
				bgr3 = [240,255,255]
				bgr4 = [255,255,240]
				bgr5 = [255,255,255]
				bgr6 = [250,250,255]
				bgr7 = [255,248,240]
				thresh = 0
				
				#1
				minBGR1 = np.array([bgr1[0] - thresh, bgr1[1] - thresh, bgr1[2] - thresh])
				maxBGR1 = np.array([bgr1[0] + thresh, bgr1[1] + thresh, bgr1[2] + thresh])
				 
				mask_Bgr1 = cv2.inRange(maskBgr, minBGR1, maxBGR1)
				mask_Con1 = cv2.inRange(maskCon, minBGR1, maxBGR1)

				# Get Shade Pixels from Garlic Object
				index_Bgr1 = np.where(mask_Bgr1 == 255)
				index_Con1 = np.where(mask_Con1 == 255)

				h_Bgr1 = index_Bgr1[0]
				w_Bgr1 = index_Bgr1[1]
				h_Con1 = index_Con1[0]
				w_Con1 = index_Con1[1]

				lst_Bgr1 = []
				lst_Con1 = []

				lst_Bgr1.append(a[h_Bgr1,w_Bgr1])
				arr_Bgr1 = lst_Bgr1[i]
				shade_Bgr1 = len(arr_Bgr1)
				ratio_Bgr1 = shade_Bgr1 / garlic

				lst_Con1.append(a[h_Con1,w_Con1])
				arr_Con1 = lst_Con1[i]
				shade_Con1 = len(arr_Con1)
				ratio_Con1 = shade_Con1 / garlic

				#2
				minBGR2 = np.array([bgr2[0] - thresh, bgr2[1] - thresh, bgr2[2] - thresh])
				maxBGR2 = np.array([bgr2[0] + thresh, bgr2[1] + thresh, bgr2[2] + thresh])
				 
				mask_Bgr2 = cv2.inRange(maskBgr, minBGR2, maxBGR2)
				mask_Con2 = cv2.inRange(maskCon, minBGR2, maxBGR2)

				# Get Shade Pixels from Garlic Object
				index_Bgr2 = np.where(mask_Bgr2 == 255)
				index_Con2 = np.where(mask_Con2 == 255)

				h_Bgr2 = index_Bgr2[0]
				w_Bgr2 = index_Bgr2[1]
				h_Con2 = index_Con2[0]
				w_Con2 = index_Con2[1]

				lst_Bgr2 = []
				lst_Con2 = []

				lst_Bgr2.append(a[h_Bgr2,w_Bgr2])
				arr_Bgr2 = lst_Bgr2[i]
				shade_Bgr2 = len(arr_Bgr2)
				ratio_Bgr2 = shade_Bgr2 / garlic

				lst_Con2.append(a[h_Con2,w_Con2])
				arr_Con2 = lst_Con2[i]
				shade_Con2 = len(arr_Con2)
				ratio_Con2 = shade_Con2 / garlic

				#3
				minBGR3 = np.array([bgr3[0] - thresh, bgr3[1] - thresh, bgr3[2] - thresh])
				maxBGR3 = np.array([bgr3[0] + thresh, bgr3[1] + thresh, bgr3[2] + thresh])
				 
				mask_Bgr3 = cv2.inRange(maskBgr, minBGR3, maxBGR3)
				mask_Con3 = cv2.inRange(maskCon, minBGR3, maxBGR3)

				# Get Shade Pixels from Garlic Object
				index_Bgr3 = np.where(mask_Bgr3 == 255)
				index_Con3 = np.where(mask_Con3 == 255)

				h_Bgr3 = index_Bgr3[0]
				w_Bgr3 = index_Bgr3[1]
				h_Con3 = index_Con3[0]
				w_Con3 = index_Con3[1]

				lst_Bgr3 = []
				lst_Con3 = []

				lst_Bgr3.append(a[h_Bgr3,w_Bgr3])
				arr_Bgr3 = lst_Bgr3[i]
				shade_Bgr3 = len(arr_Bgr3)
				ratio_Bgr3 = shade_Bgr3 / garlic

				lst_Con3.append(a[h_Con3,w_Con3])
				arr_Con3 = lst_Con3[i]
				shade_Con3 = len(arr_Con3)
				ratio_Con3 = shade_Con3 / garlic

				#4
				minBGR4 = np.array([bgr4[0] - thresh, bgr4[1] - thresh, bgr4[2] - thresh])
				maxBGR4 = np.array([bgr4[0] + thresh, bgr4[1] + thresh, bgr4[2] + thresh])
				 
				mask_Bgr4 = cv2.inRange(maskBgr, minBGR4, maxBGR4)
				mask_Con4 = cv2.inRange(maskCon, minBGR4, maxBGR4)

				# Get Shade Pixels from Garlic Object
				index_Bgr4 = np.where(mask_Bgr4 == 255)
				index_Con4 = np.where(mask_Con4 == 255)

				h_Bgr4 = index_Bgr4[0]
				w_Bgr4 = index_Bgr4[1]
				h_Con4 = index_Con4[0]
				w_Con4 = index_Con4[1]

				lst_Bgr4 = []
				lst_Con4 = []

				lst_Bgr4.append(a[h_Bgr4,w_Bgr4])
				arr_Bgr4 = lst_Bgr4[i]
				shade_Bgr4 = len(arr_Bgr4)
				ratio_Bgr4 = shade_Bgr4 / garlic

				lst_Con4.append(a[h_Con4,w_Con4])
				arr_Con4 = lst_Con4[i]
				shade_Con4 = len(arr_Con4)
				ratio_Con4 = shade_Con4 / garlic

				#5
				minBGR5 = np.array([bgr5[0] - thresh, bgr5[1] - thresh, bgr5[2] - thresh])
				maxBGR5 = np.array([bgr5[0] + thresh, bgr5[1] + thresh, bgr5[2] + thresh])
				 
				mask_Bgr5 = cv2.inRange(maskBgr, minBGR5, maxBGR5)
				mask_Con5 = cv2.inRange(maskCon, minBGR5, maxBGR5)

				# Get Shade Pixels from Garlic Object
				index_Bgr5 = np.where(mask_Bgr5 == 255)
				index_Con5 = np.where(mask_Con5 == 255)

				h_Bgr5 = index_Bgr5[0]
				w_Bgr5 = index_Bgr5[1]
				h_Con5 = index_Con5[0]
				w_Con5 = index_Con5[1]

				lst_Bgr5 = []
				lst_Con5 = []

				lst_Bgr5.append(a[h_Bgr5,w_Bgr5])
				arr_Bgr5 = lst_Bgr5[i]
				shade_Bgr5 = len(arr_Bgr5)
				ratio_Bgr5 = shade_Bgr5 / garlic

				lst_Con5.append(a[h_Con5,w_Con5])
				arr_Con5 = lst_Con5[i]
				shade_Con5 = len(arr_Con5)
				ratio_Con5 = shade_Con5 / garlic

				#6
				minBGR6 = np.array([bgr6[0] - thresh, bgr6[1] - thresh, bgr6[2] - thresh])
				maxBGR6 = np.array([bgr6[0] + thresh, bgr6[1] + thresh, bgr6[2] + thresh])
				 
				mask_Bgr6 = cv2.inRange(maskBgr, minBGR6, maxBGR6)
				mask_Con6 = cv2.inRange(maskCon, minBGR6, maxBGR6)

				# Get Shade Pixels from Garlic Object
				index_Bgr6 = np.where(mask_Bgr6 == 255)
				index_Con6 = np.where(mask_Con6 == 255)

				h_Bgr6 = index_Bgr6[0]
				w_Bgr6 = index_Bgr6[1]
				h_Con6 = index_Con6[0]
				w_Con6 = index_Con6[1]

				lst_Bgr6 = []
				lst_Con6 = []

				lst_Bgr6.append(a[h_Bgr6,w_Bgr6])
				arr_Bgr6 = lst_Bgr6[i]
				shade_Bgr6 = len(arr_Bgr6)
				ratio_Bgr6 = shade_Bgr6 / garlic

				lst_Con6.append(a[h_Con6,w_Con6])
				arr_Con6 = lst_Con6[i]
				shade_Con6 = len(arr_Con6)
				ratio_Con6 = shade_Con6 / garlic

				#7
				minBGR7 = np.array([bgr7[0] - thresh, bgr7[1] - thresh, bgr7[2] - thresh])
				maxBGR7 = np.array([bgr7[0] + thresh, bgr7[1] + thresh, bgr7[2] + thresh])
				 
				mask_Bgr7 = cv2.inRange(maskBgr, minBGR7, maxBGR7)
				mask_Con7 = cv2.inRange(maskCon, minBGR7, maxBGR7)

				# Get Shade Pixels from Garlic Object
				index_Bgr7 = np.where(mask_Bgr7 == 255)
				index_Con7 = np.where(mask_Con7 == 255)

				h_Bgr7 = index_Bgr7[0]
				w_Bgr7 = index_Bgr7[1]
				h_Con7 = index_Con7[0]
				w_Con7 = index_Con7[1]

				lst_Bgr7 = []
				lst_Con7 = []

				lst_Bgr7.append(a[h_Bgr7,w_Bgr7])
				arr_Bgr7 = lst_Bgr7[i]
				shade_Bgr7 = len(arr_Bgr7)
				ratio_Bgr7 = shade_Bgr7 / garlic

				lst_Con7.append(a[h_Con7,w_Con7])
				arr_Con7 = lst_Con7[i]
				shade_Con7 = len(arr_Con7)
				ratio_Con7 = shade_Con7 / garlic

				# Shape Based Features
				cnt = contour_list[0]

				# Shape Based Features
				## CONTOUR
				# 1. Countour Area, Countour Perimeter, Compactness, Roundness
				contour_area = cv2.contourArea(cnt)
				contour_perimeter = cv2.arcLength(cnt,True)
				compactness = ((contour_perimeter)**2) / ((4 * 3.14159) * contour_area) # elliptical
				roundness = ((4 * 3.14159) * contour_area) / ((contour_perimeter)**2) # circle

				## CONVEX HULL
				# 1. Convex Hull Area, Solidity
				hull = cv2.convexHull(cnt)
				lines = np.hstack([hull,np.roll(hull,-1,axis=0)])
				x,y = lines[:,0], lines[:,1]
				x1, y1 = x[:,0], x[:,1]
				x2, y2 = y[:,0], y[:,1]
				convex_area = 0.5 * abs(sum(x1*y2-x2*y1))
				solidity = contour_area / convex_area
				# 2. Convex Hull Perimeter, Convexity, Roughness
				convex_perimeter = cv2.arcLength(hull,True)
				convexity = convex_perimeter / contour_perimeter
				roughness = contour_perimeter / convex_perimeter

				#Extracted Features
				# print ratio_Bgr1
				# print ratio_Bgr2
				# print ratio_Bgr3
				# print ratio_Bgr4
				# print convex_perimeter
				# print convexity
				# print mean_r
				# print mean_s
				# print roughness
				# print std_s
				# print ratio_Con5
				# print ratio_Con6
				# print ratio_Con7

				global_feature = np.hstack([ratio_Bgr1, ratio_Bgr2, ratio_Bgr3, ratio_Bgr4, convex_perimeter, convexity, mean_r,mean_s,roughness,std_s,ratio_Con5,ratio_Con6,ratio_Con7])
				# print global_feature

				class_names = ['batanes','ilocos_pink','ilocos_white','mexican', 'mmsu_gem', 'tanbolters', 'vfta']


				# predict label of test image
				prediction = decision_tree_classifier.predict(global_feature.reshape(1,-1))[0]
				# print prediction

				boxes = []
				for cnt in contour_list:
					(x, y, w, h) = cv2.boundingRect(cnt)
					boxes.append([x,y, x+w,y+h])
					boxes = np.asarray(boxes)
					# need an extra "min/max" for contours outside the frame
					left = np.min(boxes[:,0])
					top = np.min(boxes[:,1])
					right = np.max(boxes[:,2])
					bottom = np.max(boxes[:,3])
					cv2.rectangle(a, (left,top), (right,bottom), (255, 0, 0), 2)

				# show predicted label on image
				cv2.putText(a, class_names[prediction], (left+5, top), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,255,255), 3)

				# display the output image
				imgfinal = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
				# plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
				# plt.show()


	cv2.imwrite(os.path.join(folder,file_name),imgfinal)

#	cv2.waitKey(0)
#	cv2.destroyAllWindows()