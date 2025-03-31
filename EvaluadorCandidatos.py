import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier


class EvaluadorCandidatos:

    def __init__(self, archivo_candidatos):
        self.archivo_candidatos = archivo_candidatos
        self.data_frame = None
        self.modelo = None
        self.labelEncoderEducativo = LabelEncoder()
        self.labelEncoderApto = LabelEncoder()
        self.mlb = MultiLabelBinarizer()
        self.__cargar_archivo_candidatos()
        self.__normalizar_datos()

    def __cargar_archivo_candidatos(self):
        self.data_frame = pd.DataFrame(pd.read_json(self.archivo_candidatos, encoding='utf-8'))

    def __normalizar_datos(self):
        #paso a valores numéricos las columnas nivel educativo y apto
        self.data_frame['Nivel educativo'] = self.labelEncoderEducativo.fit_transform(self.data_frame['Nivel educativo'])
        self.data_frame['Apto'] = self.labelEncoderApto.fit_transform(self.data_frame['Apto'])

        #crea tabla de onehot encoding para habilidades
        self.data_frame['Habilidades'] = self.data_frame['Habilidades'].apply(lambda x: x.split(', '))
        habilidades_encoded = pd.DataFrame(self.mlb.fit_transform(self.data_frame['Habilidades']), columns=self.mlb.classes_)
        self.data_frame = pd.concat([self.data_frame, habilidades_encoded], axis=1).drop(columns=['Habilidades'])

    def entrenar_modelo(self):
        x = self.data_frame.drop(columns=['Nombre', 'Apellido', 'Edad', 'Apto'])
        y = self.data_frame['Apto']

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)

        self.modelo = DecisionTreeClassifier()
        self.modelo.fit(x_train, y_train)

    def evaluar(self, candidatos):
        candidatos = pd.DataFrame(candidatos)
        columnas_extras = candidatos[["Nombre", "Apellido", "Edad", "Habilidades"]].copy() #guardo columnas que no se usaron para entrenar

        candidatos["Nivel educativo"] = self.labelEncoderEducativo.transform(candidatos["Nivel educativo"])
        candidatos["Habilidades"] = candidatos["Habilidades"].apply(lambda x: x.split(", "))
        habilidades_encoded = pd.DataFrame(self.mlb.transform(candidatos["Habilidades"]), columns=self.mlb.classes_)
        candidatos = pd.concat([candidatos, habilidades_encoded], axis=1).drop(columns=["Nombre", "Apellido", "Edad", "Habilidades"])

        apto = self.modelo.predict(candidatos)
        apto_texto = self.labelEncoderApto.inverse_transform(apto)

        candidatos['Nivel educativo'] = self.labelEncoderEducativo.inverse_transform(candidatos['Nivel educativo'])
        candidatos['Apto'] = apto_texto
        candidatos = pd.concat([candidatos, columnas_extras], axis=1)
        candidatos = candidatos.drop(columns=self.mlb.classes_)

        return candidatos

evaluador = EvaluadorCandidatos('data/candidatos_evaluados.json')
evaluador.entrenar_modelo()
nuevos_candidatos = pd.DataFrame(pd.read_json('data/nuevos_candidatos.json', encoding='utf-8'))
print(evaluador.evaluar(nuevos_candidatos))
