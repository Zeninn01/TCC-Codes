import cv2
from skimage import exposure, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from imageio import imread

# 1. Leitura da imagem usando OpenCV
imagem_colorida = cv2.imread('imagem_de_satelite.jpg')

# Verificação se a imagem foi carregada corretamente
if imagem_colorida is None:
    print("Erro ao carregar a imagem.")
    exit()

# 2. Conversão para tons de cinza
imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)

# 3. Aplicação da equalização de histograma (melhora o contraste)
imagem_equalizada = exposure.equalize_hist(imagem_cinza)

# 4. Aplicação do algoritmo de detecção de bordas de Canny
bordas_canny = cv2.Canny((imagem_equalizada * 255).astype(np.uint8), 100, 200)

# 5. Aplicação de filtro Gaussiano para suavização
imagem_suave = gaussian_filter(imagem_equalizada, sigma=2)

#normalização da imagem suavizada para o intervalo 0-255 e conversão para uint8
imagem_suave_8bit = (imagem_suave * 255).astype(np.uint8)

# 6. Aplicação do filtro Sobel para detecção de bordas
bordas_sobel = sobel(imagem_equalizada)

#normalização das bordas sobel para 0-255 e conversão para uint8
bordas_sobel_8bit = (bordas_sobel * 255 / bordas_sobel.max().astype(np.uint8))

# Salvar cada imagem processada
cv2.imwrite('imagem_cinza.jpg', imagem_cinza) #tons de cinza
cv2.imwrite('imagem_equalizada.jpg', (imagem_equalizada * 255).astype(np.uint8)) #equalização de histograma
cv2.imwrite('bordas_canny.jpg', bordas_canny) #bordas de canny
cv2.imwrite('imagem_suave.jpg', imagem_suave_8bit) #imagem suavizda
cv2.imwrite('bordas_sobel.jpg', bordas_sobel_8bit) #bordas de sobel

# 7. Exibição das imagens processadas com Matplotlib
plt.figure(figsize=(12, 8))

# Imagem original colorida
plt.subplot(2, 3, 1)
plt.title("Imagem Original")
plt.imshow(cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2RGB))  # OpenCV usa BGR, então convertemos para RGB
plt.axis('off')

# Imagem em tons de cinza
plt.subplot(2, 3, 2)
plt.title("Imagem Cinza")
plt.imshow(imagem_cinza, cmap='gray')
plt.axis('off')

# Imagem equalizada (histograma)
plt.subplot(2, 3, 3)
plt.title("Equalização de Histograma")
plt.imshow(imagem_equalizada, cmap='gray')
plt.axis('off')

# Bordas detectadas com Canny
plt.subplot(2, 3, 4)
plt.title("Bordas Canny")
plt.imshow(bordas_canny, cmap='gray')
plt.axis('off')

# Imagem suavizada com filtro Gaussiano
plt.subplot(2, 3, 5)
plt.title("Imagem Suavizada")
plt.imshow(imagem_suave, cmap='gray')
plt.axis('off')

# Bordas detectadas com Sobel
plt.subplot(2, 3, 6)
plt.title("Bordas Sobel")
plt.imshow(bordas_sobel, cmap='gray')
plt.axis('off')

# Exibir todas as imagens
plt.tight_layout()
plt.show()
