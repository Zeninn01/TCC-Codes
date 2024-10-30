import cv2
from skimage import exposure, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
import requests
from PIL import Image
from io import BytesIO

# Configurações da API do Sentinel Hub
CLIENT_ID = 'CLIENT_ID'  #informação sigilosa
CLIENT_SECRET = 'CLIENT_SECRET'  #informação sigilosa
INSTANCE_ID = 'INSTANCE_ID'  #informação sigilosa
COORDINATES = [-49.3759, -20.8111]  # Latitude e longitude de São José do Rio Preto

# Função para obter o token de autenticação
def get_access_token():
    response = requests.post(
        "https://services.sentinel-hub.com/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
    )
    return response.json().get("access_token")

# Função para fazer a solicitação da imagem de satélite
def download_satellite_image():
    headers = {
        "Authorization": f"Bearer {get_access_token()}"
    }
    payload = {
    "input": {
        "bounds": {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-49.3809, -20.8161],  # canto superior esquerdo
                    [-49.3709, -20.8161],  # canto superior direito
                    [-49.3709, -20.8061],  # canto inferior direito
                    [-49.3809, -20.8061],  # canto inferior esquerdo
                    [-49.3809, -20.8161]   # fechamento do polígono
                ]]
            }
        },
        "data": [{
            "type": "S2L2A",
            "dataFilter": {
                "timeRange": {
                    "from": "2023-01-01T00:00:00Z",
                    "to": "2023-01-31T23:59:59Z"
                }
            }
        }]
    },
    "output": {"width": 512, "height": 512},
    "evalscript": """
    // Script de renderização para imagem RGB
    function setup() {
        return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """
}   
    response = requests.post(
        f'https://services.sentinel-hub.com/api/v1/process', 
        headers=headers, 
        json=payload
    )

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        print("Erro ao buscar imagem de satélite:", response.json())
        return None

# Baixar a imagem de satélite
imagem_colorida = download_satellite_image()

# Verificação se a imagem foi carregada corretamente
if imagem_colorida is None:
    print("Erro ao carregar a imagem.")
    exit()

# Processo de pré-processamento e visualização das imagens
imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)
imagem_equalizada = exposure.equalize_hist(imagem_cinza)
bordas_canny = cv2.Canny((imagem_equalizada * 255).astype(np.uint8), 100, 200)
imagem_suave = gaussian_filter(imagem_equalizada, sigma=2)
imagem_suave_8bit = (imagem_suave * 255).astype(np.uint8)
bordas_sobel = sobel(imagem_equalizada)
bordas_sobel_8bit = (bordas_sobel * 255 / bordas_sobel.max()).astype(np.uint8)

# Salvar cada imagem processada
cv2.imwrite('imagem_cinza.jpg', imagem_cinza)
cv2.imwrite('imagem_equalizada.jpg', (imagem_equalizada * 255).astype(np.uint8))
cv2.imwrite('bordas_canny.jpg', bordas_canny)
cv2.imwrite('imagem_suave.jpg', imagem_suave_8bit)
cv2.imwrite('bordas_sobel.jpg', bordas_sobel_8bit)

# Exibição das imagens processadas com Matplotlib
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.title("Imagem Original")
plt.imshow(cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Imagem Cinza")
plt.imshow(imagem_cinza, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Equalização de Histograma")
plt.imshow(imagem_equalizada, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Bordas Canny")
plt.imshow(bordas_canny, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Imagem Suavizada")
plt.imshow(imagem_suave, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Bordas Sobel")
plt.imshow(bordas_sobel, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
