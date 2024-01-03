import numpy as np
import matplotlib.pyplot as plt

# Erzeugen Tranformationsmatrix

def generate_dct_matrix(N):
    matrix = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                matrix[k, n] = np.sqrt(1/N)
            else:
                matrix[k, n] = np.sqrt(2/N) * np.cos((np.pi * (2*n + 1) * k) / (2 * N))
    return matrix

# Anwenden der DCT 

def apply_dct_transform(signal, matrix):    
    return np.dot(matrix, signal)

# Anwenden der inversen DCT

def apply_inverse_dct_transform(transformed_signal, matrix):
    return np.dot(matrix, transformed_signal)

# Matrix Grösse
N = 16
signal = np.random.rand(N)

# Generierung DCT Matrix und Anwenden der Transformation
dct_matrix = generate_dct_matrix(N)
dct_signal = apply_dct_transform(signal, dct_matrix)

# Durchführung der inversen Transformation
idct_matrix = generate_dct_matrix(N).T  
reconstructed_signal = apply_inverse_dct_transform(dct_signal, idct_matrix)

# Plotting der Signale
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(signal, label='Original Signal')
axes[0].set_title('Originales Signal')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True)

axes[1].plot(dct_signal, label='DCT Transformed Signal', color='red')
axes[1].set_title('DCT Transformiertes Signal')
axes[1].set_xlabel('Frequency Index')
axes[1].set_ylabel('Amplitude')
axes[1].grid(True)

axes[2].plot(reconstructed_signal, label='Reconstructed Signal', color='green')
axes[2].set_title('Rekonstruiertes Signal')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Amplitude')
axes[2].grid(True)

plt.tight_layout()
plt.show()

### 2D - DCT mit SCIPY

# Import der Bibliotheken
from scipy.fftpack import dct, idct
from PIL import Image

# Definition Methoden
def load_image(file_path):
    img = Image.open(file_path).convert('L')
    return np.array(img)

def apply_dct(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

def apply_idct(image):
    return idct(idct(image.T, norm='ortho').T, norm='ortho')

def normalize_dct(dct_image):
    return np.log(np.abs(dct_image) + 1)

#Laden Der Bilder 
photo1 = 'picture.jpg' 
photo2 = 'line.jpg'  

photo = load_image(photo1)
line_drawing = load_image(photo2)

# Anwenden DCT auf die Bilder
dct_photo = apply_dct(photo)
dct_line_drawing = apply_dct(line_drawing)

# Normalisieren der Bilder
normalized_dct_photo = normalize_dct(dct_photo)
normalized_dct_line_drawing = normalize_dct(dct_line_drawing)

# Anwenden der inversen DCT
idct_photo = apply_idct(dct_photo)
idct_line_drawing = apply_idct(dct_line_drawing)

# VVisualisierung der Bilder
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0, 0].imshow(photo, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(normalized_dct_photo, cmap='gray')
axes[0, 1].set_title('DCT')
axes[0, 2].imshow(idct_photo, cmap='gray')
axes[0, 2].set_title('IDCT')

axes[1, 0].imshow(line_drawing, cmap='gray')
axes[1, 0].set_title('Original ')
axes[1, 1].imshow(normalized_dct_line_drawing, cmap='gray')
axes[1, 1].set_title('DCT')
axes[1, 2].imshow(idct_line_drawing, cmap='gray')
axes[1, 2].set_title('IDCT')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show() 
