import matplotlib.pyplot as plt


def histogram(image):
    plt.hist(image.ravel(), bins=256)
    plt.show()


def show_2(coins, image, axis):
    plt.subplot(211)
    plt.imshow(coins, cmap="gray")
    plt.axis(axis)

    plt.subplot(212)
    plt.imshow(image, cmap="gray")
    plt.axis(axis)
    plt.show()


def show_1(image, axis):
    plt.imshow(image, cmap="gray")
    plt.axis(axis)
    plt.show()