import os

from flask import Flask, request
from flask_cors import CORS
import pandas as pd
from EvaluadorCandidatos import EvaluadorCandidatos

app = Flask(__name__)
CORS(app)

evaluador = EvaluadorCandidatos("data\\candidatos_evaluados.json")
evaluador.entrenar_modelo()

@app.route("/evaluar", methods=['POST'])
def evaluar_candidatos():
    file = request.files['file']
    df = pd.read_excel(file)

    resultados = evaluador.evaluar(df)

    return resultados.to_json(orient='records', force_ascii=False)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)