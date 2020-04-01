import numpy as np
import cv2

def TemplateMatching(input_img, template_img, threshold):
    #transform img and template to grayscale
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)

    #resize input and template image for optimization
    resized_input_img = cv2.resize(input_img, (int(input_img.shape[1] * 0.5), int(input_img.shape[0] * 0.5)), interpolation = cv2.INTER_NEAREST)
    resized_template_img = cv2.resize(template_img, (int(template_img.shape[1] * 0.5), int(template_img.shape[0] * 0.5)), interpolation = cv2.INTER_NEAREST)

    #get shapes and create matching mape with correct size
    input_height, input_width = resized_input_img.shape
    template_height, template_width = resized_template_img.shape
    matching_map = np.ones((input_height - template_height + 1, input_width - template_width + 1))
    matching_height, matching_width = matching_map.shape

    #template matching calculations
    for i in range (0, matching_height):
        for j in range(0, matching_width):
            matching_map[i,j] = ((resized_template_img[:,:] - resized_input_img[i:i+template_height, j:j+template_width])**2).sum(axis=(0,1))
   
    matching_map = matching_map/matching_map.max()
    detected = False
    if(matching_map.min()/matching_map.max() < threshold):
        detected = True

    #resize matching map to match original size it should have just for imgshow purpose
    matching_map = cv2.resize(matching_map, (int(matching_map.shape[1] * 2), int(matching_map.shape[0] * 2)), interpolation = cv2.INTER_NEAREST)

    return detected, matching_map

input_route = input("Enter input image: ")
template_route = input("Enter template image: ")
threshold = float(input("Enter threshold: "))

input_img = cv2.imread(input_route, -1)
template_img = cv2.imread(template_route, -1)

detected, matching_map = TemplateMatching(input_img, template_img, threshold)

#find spots where we should place a rectangle
loc = np.where(matching_map <= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(input_img, pt, (pt[0] + template_img.shape[1], pt[1] + template_img.shape[0]), (0,255,0), 1)

#set a font txt_img depending if target was found or not
font = cv2.FONT_HERSHEY_SIMPLEX
if(detected):
    txt_img = np.zeros((40,245,3), np.uint8)
    cv2.putText(txt_img, 'TARGET FOUND', 
    (5,30), font, 1, (0,255,0), 2)
else:
    txt_img = np.zeros((40,325,3), np.uint8)
    cv2.putText(txt_img, 'TARGET NOT FOUND', 
    (5,30), font, 1, (0, 0, 255), 2)

#show all imgs
cv2.imshow("Target", template_img)
cv2.imshow("Matching Map", matching_map)
cv2.imshow("Input Image", input_img)
cv2.imshow("Result", txt_img)

k = cv2.waitKey(0)

if k==27:
    cv2.destroyAllWindows()