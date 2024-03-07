import cv2
import numpy as np


# Below function will read video imgs
# cap = cv2.VideoCapture('bottle_cap_v1.mp4')

#def change_lower_red_S(value):
#    global lower_red_S  # inform function to assign value to global variable instead of local variable
#    lower_red_S = value
#
#
#def change_high_red_S(value):
#    global high_red_S  # inform function to assign value to global variable instead of local variable
#    high_red_S = value
#
#
#def change_lower_red_V(value):
#    global lower_red_V  # inform function to assign value to global variable instead of local variable
#    lower_red_V = value
#
#
#def change_high_red_V(value):
#    global high_red_V  # inform function to assign value to global variable instead of local variable
#    high_red_V = value
#
#
#lower_red_S = 30
#
#high_red_S = 255
#
#lower_red_V = 30
#
#high_red_V = 255

cap = cv2.VideoCapture(0)

cv2.namedWindow('image')
#cv2.createTrackbar('lower_red_S', 'image', lower_red_S, 255, change_lower_red_S)
#cv2.createTrackbar('high_red_S', 'image', high_red_S, 255, change_high_red_S)
#cv2.createTrackbar('lower_red_V', 'image', lower_red_V, 255, change_lower_red_V)
#cv2.createTrackbar('high_red_V', 'image', high_red_V, 255, change_high_red_V)

while True:
    read_ok, img = cap.read()
    img_bcp = img.copy()

    img = cv2.resize(img, (640, 480))
    # Make a copy to draw contour outline
    input_image_cpy = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    min_contour_area = 2000

    # define range of red color in HSV
    lower_red = np.array([0, 100, 90])  # np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])  # np.array([10, 255, 255])

    # define range of green color in HSV
    lower_green = np.array([40, 100, 50])  # np.array([40, 20, 50])
    upper_green = np.array([90, 255, 255])  # np.array([90, 255, 255])

    # define range of blue color in HSV
    lower_blue = np.array([100, 180, 50])  # np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])  # np.array([130, 255, 255])

    # create a mask for red color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # create a mask for green color
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # create a mask for blue color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # find contours in the red mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the green mask
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the blue mask
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop through the red contours and draw a rectangle around them
    for cnt in contours_red:
        contour_area = cv2.contourArea(cnt)
        if contour_area > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, 'Red', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # loop through the green contours and draw a rectangle around them
    for cnt in contours_green:
        contour_area = cv2.contourArea(cnt)
        if contour_area > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Green', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # loop through the blue contours and draw a rectangle around them
    for cnt in contours_blue:
        contour_area = cv2.contourArea(cnt)
        if contour_area > min_contour_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, 'Blue', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Color Recognition Output', img)

    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
