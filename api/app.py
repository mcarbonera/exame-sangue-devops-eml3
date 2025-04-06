from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

loadedModel = pickle.load(open('model.pkl', 'rb'))
model = loadedModel['model']
preprocessor = loadedModel['preprocessor']
medians = loadedModel['medians']

def get_input_data(request_data, param):
  return request_data[param] if param in request_data else medians[param]

def get_input(request_data):
  df = pd.DataFrame(columns=['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'])
  age = get_input_data(request_data, 'Age')
  alb = get_input_data(request_data, 'ALB')
  alp = get_input_data(request_data, 'ALP')
  alt = get_input_data(request_data, 'ALT')
  ast = get_input_data(request_data, 'AST')
  bil = get_input_data(request_data, 'BIL')
  che = get_input_data(request_data, 'CHE')
  chol = get_input_data(request_data, 'CHOL')
  crea = get_input_data(request_data, 'CREA')
  ggt = get_input_data(request_data, 'GGT')
  prot = get_input_data(request_data, 'PROT')
  df.loc[-1] = [age, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]
  return df

@app.route("/classificar", methods=["POST"])
def classificar():
  request_data = request.get_json()  
  df = get_input(request_data)
  
  infoNormalizada = preprocessor.transform(df)
  dfNormalizado = pd.DataFrame(infoNormalizada)
  resultado = model.predict(dfNormalizado)
  
  response = {'categoria': int(resultado[0])}
  return jsonify(response)