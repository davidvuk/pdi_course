import matplotlib.pyplot as plt
import numpy as np
from skimage import color
from skimage.filters import gaussian
from skimage.segmentation import slic
from math import hypot
import cv2 as cv


def show1(imagen):
    plt.imshow(imagen, cmap="gray")
    plt.axis("off")
    plt.show()


def show2(image1, image2):
    plt.figure(1, figsize=(15, 20))
    plt.subplot(121)
    plt.imshow(image1, cmap="gray")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(image2, cmap="gray")
    plt.axis("off")
    plt.show()


coins = plt.imread('monedas.jpg')
coins_gray = color.rgb2gray(coins)
show1(coins)

# Realiza 2 procesos de segmetación kmeans, despues del primero se realiza un
# un filtro gausiano para poder distorcionar las segmentaciones del fondo
# para que asi despues de la siguiente segmentacion kmeans poder tener definido
# las formas de la segmentacion de las monedas
seg = slic(coins, n_segments=18)
image = color.label2rgb(seg, coins, kind='avg')
image = gaussian(image, sigma=10, multichannel=True)
seg = slic(image, n_segments=13)
image = color.label2rgb(seg, coins, kind='avg')
show1(image)

# se divide la segmentacion rgb en los respectivos ganales
# para poder utilizar este en la busqueda de circunferencias en la imagen
# de esta manera obtenemos obtemos las circunferencias encontradas que son
# las monedas y obtenemos así sus respectivos centros y radios
b, g, r = cv.split(image)
image = b
circles = cv.HoughCircles(image, cv.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=10, param2=20,
                          minRadius=100,
                          maxRadius=200)
circles = np.uint16(np.around(circles))

# creamos una mascara de color negro en cada pixel
# que asu vez se utiliza para crear la mascara del fondo de las monedas
mask = np.ones(np.shape(coins_gray))

# Utilizamos la mascara negra para poder tener una mascara de las monedas y de
# esta manera la utilizamos para poder dejar fondo negro de la imagen
# de las monedas
for z in range(0, 3):
    x, y, r = circles[0, :][z]
    rows, cols = coins_gray.shape
    for i in range(cols):
        for j in range(rows):
            if hypot(i - x, j - y) < r:
                mask[j, i] = 0

coins_fondonegro = coins.copy()
coins_fondonegro[np.where(mask)] = 0

show2(coins, coins_fondonegro)
