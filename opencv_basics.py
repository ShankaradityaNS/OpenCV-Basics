import cv2
import numpy as np

# For reading img
# img = cv2.imread("Pics/dragonballz.png")
# cv2.imshow("OutPut", img)
# cv2.waitKey(0)


# For reading video
# cap = cv2.VideoCapture("Videos/CommunityS02E04BasicRocketScience.mp4")
# while True:
#     success, img= cap.read()
#     cv2.imshow("Video Output", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# For accessing web-cam
# cap = cv2.VideoCapture(0)
# cap.set(3,640)  #id no 3 for width
# cap.set(4,480)  #id no 4 for height
# cap.set(10,100) #id no 10 for brightness
# while True:
#     success, img= cap.read()
#     cv2.imshow("Video Output", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# # Converting Img to Grey, Blur, Edge Detection, Dialation, Errosion
# img = cv2.imread("Pics/dragonballz.png")
#
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #cv2.imshow("Grey Img", img_grey)
#
# img_blur = cv2.GaussianBlur(img_grey, (9, 9), 0)
# #cv2.imshow("Blur Img", img_blur)
#
# img_edge = cv2.Canny(img, 200, 150)
# cv2.imshow("Edge Detection", img_edge)
#
# img_dialation = cv2.dilate(img_edge, np.ones((5,5), np.uint8), iterations=1)
# cv2.imshow("Dialation IMG", img_dialation)
#
# img_errosion = cv2.erode(img_dialation, np.ones((5,5), np.uint8),iterations=1)
# cv2.imshow("Erroded Img", img_errosion)
#
# cv2.waitKey(0)


# #Resizing and Cropping Img
# img = cv2.imread("Pics/dragonballz.png")
#
# print(img.shape)    #height, width, no of channels(BGR)
#
# img_resize = cv2.resize(img,(500,500))
# print(img_resize.shape)
# cv2.imshow("Resized Img", img_resize)
#
# img_crop = img[100:300,100:500]     #height,width
# cv2.imshow("Cropped Img", img_crop)
#
# cv2.waitKey(0)


# Adding colors, line, rectangle,circle and text
# img = np.ones((512, 512, 3), np.uint8)  # np.zeros((512,512,3),np.uint8)
# print(img.shape)
# cv2.imshow("Img", img)
# cv2.waitKey(0)
#
# img[:] = 255, 0, 0  # B,G,R values
# cv2.imshow("Colored Img", img)  # Full screen with 1 color
# cv2.waitKey(0)
#
# img[100:200, 250:350] = 0, 0, 255  # B,G,R values for height and width cropped sections
# cv2.imshow("Cropped Colored Img", img)  # Cropped section
# cv2.waitKey(0)
#
# cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)  # img,starting pt, ending pt,color,thickness
# cv2.imshow("Line Img", img)
# cv2.waitKey(0)
#
# cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)  # img,starting pt, ending pt,color,thickness
# cv2.imshow("Line Img2", img)
# cv2.waitKey(0)
#
# cv2.rectangle(img, (50, 100), (250, 350), (100, 100, 100), 8)  # img,starting pt, ending pt,color,thickness
# cv2.imshow("Rectangle", img)
# cv2.waitKey(0)
#
# cv2.rectangle(img, (50, 100), (250, 350), (100, 100, 100),
#               cv2.FILLED)  # img,starting pt, ending pt,color,filling entire area
# cv2.imshow("Rectangle 2", img)
# cv2.waitKey(0)
#
# cv2.circle(img, (400, 50), 30, (100, 50, 150), 3)  # img,center,radius,color,thickness
# cv2.imshow("Cirlce", img)
# cv2.waitKey(0)
#
# cv2.circle(img, (400, 50), 30, (100, 0, 250), cv2.FILLED)  # img,center,radius,color,filling entire area
# cv2.imshow("Cirlce 2", img)
# cv2.waitKey(0)
#
# cv2.putText(img,"Hello World",(200,400),cv2.FONT_HERSHEY_DUPLEX,1.75,(225,225,225),4)      #img,text,origin,font,fontsize,color,thickness
# cv2.imshow("Text", img)
# cv2.waitKey(0)

# # Extracting part of img and stacking imgs
# img = cv2.imread("Pics/cards.png")
# print(img.shape)
# width, height = 500, 500  # Size of output img
# pt1 = np.float32([[318, 451], [563, 349], [707, 703], [464, 799]])  # pts of img to be extracted
# pt2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])  # corresponding loc of above pts
# matrix = cv2.getPerspectiveTransform(pt1, pt2)
# img_output = cv2.warpPerspective(img, matrix, (width, height))  # extracting the img
# cv2.imshow("Output", img_output)
# cv2.waitKey(0)
#
# pt1 = np.float32([[1216, 478], [1478, 441], [1522, 819], [1258, 852]])  # pts of img to be extracted
# pt2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])  # corresponding loc of above pts
# matrix = cv2.getPerspectiveTransform(pt1, pt2)
# img_output = cv2.warpPerspective(img, matrix, (width, height))  # extracting the img
# cv2.imshow("Output2", img_output)
# cv2.waitKey(0)
#
# img_horizontal = np.hstack((img_output, img_output))    #Dimensions(no of channels, width,height) of given two i/p imgs has to be the same
# cv2.imshow("Horrizontal", img_horizontal)
# cv2.waitKey(0)
#
# img_vertical = np.vstack((img_output, img_output))      #Dimensions of given two i/p imgs has to be the same
# cv2.imshow("Vertical", img_vertical)
# cv2.waitKey(0)


# # Color Detection
# def empty(a):
#     pass
#
#
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)
# cv2.createTrackbar("Hue Min", "TrackBars", 19, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 91, 179, empty)
# cv2.createTrackbar("Saturation Min", "TrackBars", 49, 255, empty)
# cv2.createTrackbar("Saturation Max", "TrackBars", 108, 255, empty)
# cv2.createTrackbar("Value Min", "TrackBars", 91, 255, empty)
# cv2.createTrackbar("Value Max", "TrackBars", 152, 255, empty)
#
# while True:
#     img = cv2.imread('Pics/dragonballz.png')
#     img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # HSV - Hue Saturation Value it gives a numerical values to the colors in img ranging from 0-360
#     hue_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     hue_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     sat_min = cv2.getTrackbarPos("Saturation Min", "TrackBars")
#     sat_max = cv2.getTrackbarPos("Saturation Max", "TrackBars")
#     val_min = cv2.getTrackbarPos("Value Min", "TrackBars")
#     val_max = cv2.getTrackbarPos("Value Max", "TrackBars")
#     print(hue_min, hue_max, sat_min, sat_max, val_min, val_max)
#     lower = np.array([hue_min, sat_min, val_min])
#     upper = np.array([hue_max, sat_max, val_max])
#     mask = cv2.inRange(img, lower, upper)
#     img_result = cv2.bitwise_and(img, img, mask=mask)
#     cv2.imshow("Original Img", img)
#     cv2.imshow("HSV IMG", img_HSV)
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Result Img", img_result)
#     cv2.waitKey(1)


# # Shapes and Contours
# def getContours(img):
#     contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area != 0:
#             print(area)
#             cv2.drawContours(img_contour, cnt, -1, (255, 0, 0), 3)
#             perimeter = cv2.arcLength(cnt, True)
#             print(perimeter)
#             corner_pts = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
#             print(len(corner_pts))
#             print(corner_pts)
#             objcor = len(corner_pts)
#             x, y, width, height = cv2.boundingRect(corner_pts)
#             if objcor == 3:
#                 objtype = "Triangle"
#             elif objcor == 4 and width == height:
#                 objtype = "Square"
#             elif objcor == 4 and width != height:
#                 objtype = "Rectangle"
#             elif objcor>4:
#                 objtype = "Circles"
#             else:
#                 objtype = "None"
#             cv2.rectangle(img_contour, (x, y), (x + width, y + height), (0, 255, 0), 2)
#             cv2.putText(img_contour, objtype, (int(x + (width / 2) - 10), int(y + (height / 2) - 10)),
#                         cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
# 
# 
# img = cv2.imread('Pics/shapes3.png')
# img_contour = img.copy()
# print(img.shape)
# img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_blur = cv2.GaussianBlur(img_grey, (7, 7), 1)
# img_edge = cv2.Canny(img, 50, 50)
# getContours(img_edge)
# cv2.imshow("Original Img", img)
# cv2.imshow("Grey Img", img_grey)
# cv2.imshow("Blur Img", img_blur)
# cv2.imshow("Edges", img_edge)
# cv2.imshow("Contours Img", img_contour)
# cv2.waitKey(0)


# # Face Detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# img = cv2.imread('Pics/people.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
# for (x, y, width, height) in faces:
#     cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 1)
#
# cv2.imshow("People Img", img)
# cv2.waitKey(0)

