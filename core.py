import cv2
from functions import extract, find_locals
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.io import imread, imshow
from matplotlib.pyplot import imread
from skimage.morphology import disk

"""
    Una vez tengamos los frames del video debemos
    1. Recorrer cada uno y calcular su entropia
    2 Guardar los valores de la entropía en una lista
    3. Pasar a la función find_locals la lista para encontrar máximos y mínimos locales
    4. Guardar 
"""

# Now let us import the image we will be working with.

#shawls = imread('/home/javargas/Escritorio/images/Frame1.jpg', as_gray=True)
#plt.figure(num=None, figsize=(8,62))
#imshow(shawls);
img = cv2.imread('/home/javargas/Escritorio/images/Frame1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
entropy_image = entropy(img, disk(5))
print(entropy_image)
#entropy_image = entropy(shawls, disk(5))

#plt.show()
#print((find_locals([1,4,2,6,5,8,7], "maxima")))
#print((find_locals([1,4,2,6,5,8,7], "minima")))