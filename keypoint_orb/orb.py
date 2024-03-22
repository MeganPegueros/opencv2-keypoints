import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../img3.jpg', cv2.IMREAD_GRAYSCALE)

# Inicializar el detector ORB
orb = cv2.ORB_create()

# Detectar y describir los keypoints
keypoints, descriptors = orb.detectAndCompute(img, None)

# Dibujar los keypoints en la imagen
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)

cv2.imshow('ORB', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("orb.jpeg", img_with_keypoints)