# Sobre:

-Disciplina de DevOps do MBA em Machine Learning in Production da UFSCAR.

# Estrutura do projeto:

- /treinamento: nessa pasta foi criado um script salvar-modelo.py para repetir o processo do notebook e salvar o modelo em um arquivo "model.pkl"
- /treinamento/notebook-atividade: notebook original de uma atividade de aprendizado não supervisionado (Kmeans com método do cotovelo). Nele, foram feitas duas análises de datasets, distintos, um com dados de minas terrestres, outro com dados de exames de sangue.
- /api: API em Flask para carregar o modelo e fazer as previsões.

# Descrição da etapa de geração do modelo:

- O script /treinamento/salvar-modelo.py realiza novamente as etapas presentes no notebook, mas apenas para o dataset de exames de sangue, a fim de salvar o modelo. 
- Para rodar o script, deve-se apenas executar:

```
python3 treinamento/salvar-modelo.py
```

# Etapa de desenvolvimento da API:

- A API possui 1 endpoint para enviar dados de exame de sangue e obter a classificação:

```
  POST localhost:5000/classificar
```

- Os dados de entrada devem ter esse formato, que são os parâmetros de um exame de sangue:

```
{
  "Age": 32,
  "ALB": 43.2,
  "ALP": 52.0,
  "ALT": 30.6,
  "AST": 22.6,
  "BIL": 18.9,
  "CHE": 7.33,
  "CHOL": 4.74,
  "CREA": 80.0,
  "GGT": 33.8,
  "PROT": 75.7
}
```

- No caso de algum desses valores ser nulo, a mediana é considerada.

- O objeto é mapeado para um vetor, normalizado na etapa de pré-processamento para depois ser classificado.

- É importante dizer que o resultado dessa classificação não tem um significado específico e nem foi desprendida muita energia tentando compreendê-lo. Portanto, será apenas um número que foi resultado da aplicação do algoritmo Kmeans, sem valor qualitativo tendo em vista a ausência de uma análise aprofundada.

- Para rodar localmente, executar:

```
cd api
flask run
```

# Instruções para rodar:

- Build da imagem

```
cd api
docker build -t moduloeml1:atividade2 .
```

- Subir container

```
docker run -p 5000:5000 moduloeml1:atividade2
```