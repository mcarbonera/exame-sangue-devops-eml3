# Imagem Slim
FROM python:3.12.3-slim

# Diretório do projeto
WORKDIR /usr/src/app

# Arquivo de gerenciamento de dependências
COPY requirements.txt ./

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# copiar arquivos do projeto
COPY *.py .
COPY model.pkl .

# Rodar aplicação flask
CMD [ "flask", "run", "--host=0.0.0.0" ]