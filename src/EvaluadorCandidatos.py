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
        self.labelEncoderIngles = LabelEncoder()
        self.labelEncoderDisponibilidad = LabelEncoder()
        self.mlb_habilidades = MultiLabelBinarizer()
        self.mlb_conocimientos = MultiLabelBinarizer()
        self.mlb_certificaciones = MultiLabelBinarizer()
        self.__cargar_archivo_candidatos()
        self.__normalizar_datos()

    def __cargar_archivo_candidatos(self):
        self.data_frame = pd.DataFrame(pd.read_json(self.archivo_candidatos, encoding='utf-8'))

    def __normalizar_datos(self):
        # paso a valores numéricos las columnas nivel educativo y apto
        self.data_frame['Nivel educativo'] = self.labelEncoderEducativo.fit_transform(
            self.data_frame['Nivel educativo'])
        self.data_frame['Apto'] = self.labelEncoderApto.fit_transform(self.data_frame['Apto'])
        self.data_frame['Nivel de inglés'] = self.labelEncoderIngles.fit_transform(self.data_frame['Nivel de inglés'])
        self.data_frame['Disponibilidad'] = self.labelEncoderDisponibilidad.fit_transform(
            self.data_frame['Disponibilidad'])

        # crea tabla de onehot encoding para habilidades, conocimientos adicionales y certificaciones
        self.data_frame['Habilidades'] = self.data_frame['Habilidades'].fillna('').apply(lambda x: x.split(', '))
        habilidades_encoded = pd.DataFrame(self.mlb_habilidades.fit_transform(self.data_frame['Habilidades']),
                                           columns=self.mlb_habilidades.classes_)

        self.data_frame['Conocimientos adicionales'] = self.data_frame['Conocimientos adicionales'].fillna('').apply(
            lambda x: x.split(', '))
        conocimientos_encoded = pd.DataFrame(
            self.mlb_conocimientos.fit_transform(self.data_frame['Conocimientos adicionales']),
            columns=self.mlb_conocimientos.classes_)

        self.data_frame['Certificaciones'] = self.data_frame['Certificaciones'].fillna('').apply(
            lambda x: x.split(', '))
        certificaciones_encoded = pd.DataFrame(
            self.mlb_certificaciones.fit_transform(self.data_frame['Certificaciones']),
            columns=self.mlb_certificaciones.classes_)

        #agrego al dataframe de entremaniento las tablas de onehot encoding y elimino las que tienen strings
        self.data_frame = pd.concat(
            [self.data_frame, habilidades_encoded, conocimientos_encoded, certificaciones_encoded], axis=1).drop(
            columns=['Habilidades', 'Conocimientos adicionales', 'Certificaciones'])

    def entrenar_modelo(self):
        #elimino datos no relevantes para el entrenamiento y entreno al modelo
        x = self.data_frame.drop(columns=['Nombre', 'Apellido', 'Edad', 'Apto'])
        y = self.data_frame['Apto']

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=0)

        self.modelo = DecisionTreeClassifier()
        self.modelo.fit(x_train, y_train)

    def evaluar(self, candidatos):
        candidatos = pd.DataFrame(candidatos)
        columnas_extras = candidatos[["Nombre", "Apellido", "Edad", "Habilidades", "Conocimientos adicionales",
                                      "Certificaciones"]].copy()  # guardo columnas que no se usaron para entrenar

        #paso a valores numéricos los campos nivel educativo, de inglés y disponibilidad para evaluar los candidatos
        candidatos["Nivel educativo"] = self.labelEncoderEducativo.transform(candidatos["Nivel educativo"])
        candidatos["Nivel de inglés"] = self.labelEncoderIngles.transform(candidatos["Nivel de inglés"])
        candidatos["Disponibilidad"] = self.labelEncoderDisponibilidad.transform(candidatos["Disponibilidad"])

        #paso a one hot encoding los campos de habilidades, conocimientos adicionales y certificaciones
        candidatos["Habilidades"] = candidatos['Habilidades'].fillna('').apply(lambda x: x.split(', '))
        habilidades_encoded = pd.DataFrame(self.mlb_habilidades.transform(candidatos["Habilidades"]),
                                           columns=self.mlb_habilidades.classes_)

        candidatos['Conocimientos adicionales'] = candidatos['Conocimientos adicionales'].fillna('').apply(lambda x: x.split(', '))
        conocimientos_encoded = pd.DataFrame(self.mlb_conocimientos.transform(candidatos['Conocimientos adicionales']),
                                             columns=self.mlb_conocimientos.classes_)

        candidatos['Certificaciones'] = candidatos['Certificaciones'].fillna('').apply(lambda x: x.split(', '))
        certificaciones_encoded = pd.DataFrame(self.mlb_certificaciones.transform(candidatos['Certificaciones']),
                                               columns=self.mlb_certificaciones.classes_)

        #agrego los valores numericos y tablas de onehot encoding y elimino las columnas que les corresponden
        candidatos = pd.concat([candidatos, habilidades_encoded, conocimientos_encoded, certificaciones_encoded],
                               axis=1).drop(
            columns=["Nombre", "Apellido", "Edad", "Habilidades", "Conocimientos adicionales", "Certificaciones"])

        #el modelo predice los nuevos candidatos
        apto = self.modelo.predict(candidatos)
        apto_texto = self.labelEncoderApto.inverse_transform(apto)

        #devuelvo los valores numéricos de las columnas a string
        candidatos["Apto"] = apto_texto
        candidatos['Nivel educativo'] = self.labelEncoderEducativo.inverse_transform(candidatos['Nivel educativo'])
        candidatos['Nivel de inglés'] = self.labelEncoderIngles.inverse_transform(candidatos['Nivel de inglés'])
        candidatos['Disponibilidad'] = self.labelEncoderDisponibilidad.inverse_transform(candidatos['Disponibilidad'])

        #vuelvo a agregar las columnas que no eran relevantes para predecir la aptitud de los candidatos
        #y elimino las tablas one hot encoding
        candidatos = pd.concat([candidatos, columnas_extras], axis=1)
        columnas_onehot = (list(self.mlb_habilidades.classes_) + list(self.mlb_conocimientos.classes_) + list(
            certificaciones_encoded.columns))
        candidatos = candidatos.drop(columns=columnas_onehot, errors='ignore')

        #ordeno como se verán en el front-end las columnas
        columnas_ordenadas = ["Nombre", "Apellido", "Edad", "Años de experiencia", "Nivel educativo", "Nivel de inglés",
                              "Disponibilidad",
                              "Pretensión salarial", "Último empleo (meses)", "Habilidades",
                              "Conocimientos adicionales",
                              "Certificaciones", "Apto"]

        candidatos = candidatos[columnas_ordenadas]

        return candidatos
