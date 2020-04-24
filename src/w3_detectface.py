import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time


def show1(imagen):
    plt.imshow(imagen, cmap="gray")
    plt.axis("off")
    plt.show()


def show2(image1, image2):
    plt.figure(1, figsize=(15, 20))
    plt.subplot(211)
    plt.imshow(image1, cmap="gray")
    plt.axis("off")

    plt.subplot(212)
    plt.imshow(image2, cmap="gray")
    plt.axis("off")
    plt.show()


# photo = cv.imread('letras.jpg')
# # show1(photo)
#
# photo_gris = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
# # show2(photo_gris, photo)
cascade_path = 'haarcascade_frontalface_default.xml'
classifier = cv.CascadeClassifier(cascade_path)
# faces_detected = classifier.detectMultiScale(photo_gris,
#                                              scaleFactor=1.163145,
#                                              minNeighbors=1,
#                                              flags=cv.CASCADE_SCALE_IMAGE,
#                                              minSize=(60, 60),
#                                              maxSize=(100, 100))
#
#
# for(x, y, l, a) in faces_detected:
#     photo = cv.rectangle(photo, (x, y), (x+l, y+a), (212, 172, 13), 2)
#     region_gaussian = cv.GaussianBlur(photo[y:y+l, x:x+a], (11, 11), 13)
#     photo[y:y + l, x:x + a] = region_gaussian
#
#
# show1(photo)
#     region_mean = cv.medianBlur(photo[y:y+l, x:x+a], 15)
#     photo[y:y + l, x:x + a] = region_mean

# photo2_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
# faces_detected2 = classifier.detectMultiScale(photo2_gray, scaleFactor=1.163145, minNeighbors=1, minSize=(60, 60),
#                                               maxSize=(100, 100))
#
# for(x, y, l, a) in faces_detected:
#     photo = cv.rectangle(photo, (x, y), (x+l, y+a), (212, 172, 13), 2)
#     # region_gaussian = cv.GaussianBlur(photo[y:y+l, x:x+a], (23, 23), 13)
#     # photo[y:y + l, x:x + a] = region_gaussian
#
# show1(photo)
# time.sleep(2)

# Second part
photo2 = cv.imread('letras.jpg')
photo2_gray = cv.cvtColor(photo2, cv.COLOR_BGR2GRAY)
faces_detected2 = classifier.detectMultiScale(photo2_gray,
                                              scaleFactor=1.163145,
                                              minNeighbors=1,
                                              minSize=(60, 60),
                                              maxSize=(100, 100))


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


for(x, y, l, a) in faces_detected2:
    photo2 = cv.rectangle(photo2, (x, y), (x+l, y+a), (212, 172, 13), 2)
    region = photo2[y:y+l, x:x+a]
    region = rotate_image(region, 180)
    time.sleep(2)
    photo2[y:y + l, x:x + a] = region

show1(photo2)
