import numpy as np
import cv2

# img = cv2.imread('image.png',-1)
# cv2.imshow("MonkaW",img)

# k = cv2.waitKey(0)

# if k==27:
#     cv2.destroyAllWindows()
# elif k == ord('s'):
#     cv2.imwrite('image_grayscale',img)
#     cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,24.0,(640,480))

while(True):
    ret, frame = cap.read()

    if ret==True:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.flip(frame,1)
        out.write(frame)
        cv2.imshow('Pepega Cam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
