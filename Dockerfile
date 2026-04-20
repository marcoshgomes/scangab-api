FROM python:3.9-slim

# Evita que o Linux faça perguntas durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências essenciais para o OpenCV rodar em servidor
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instala as bibliotecas do Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando para rodar a API na porta padrão do Render (10000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]