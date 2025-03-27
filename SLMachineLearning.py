import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier

candidatos = {
    'Nombre': ["Juan", "Pedro", "Maria", "Ana"],
    'Apellido': ["Perez", "Garcia", "Garcia", "Garcia"],
    'Años de experiencia': [1, 5, 3, 13],
    'Nivel educativo': ["secundario", "universitario", "universitario", "universitario"],
    'Habilidades': ["python, java", "c++, docker", "java, sql", "python, java, c++"],
    'Edad': [25, 30, 20, 25],
    'Apto': ["No", "Si", "No", "Si"]
}
data_frame = pd.DataFrame(candidatos)

labelEncoderEducativo = LabelEncoder()
labelEncoderApto = LabelEncoder()
data_frame['Nivel educativo'] = labelEncoderEducativo.fit_transform(data_frame['Nivel educativo'])
data_frame['Apto'] = labelEncoderApto.fit_transform(data_frame['Apto'])

data_frame['Habilidades'] = data_frame['Habilidades'].apply(lambda x: x.split(', '))
mlb = MultiLabelBinarizer()
habilidades_encoded = pd.DataFrame(mlb.fit_transform(data_frame['Habilidades']), columns=mlb.classes_)

data_frame = pd.concat([data_frame, habilidades_encoded], axis=1).drop(columns=['Habilidades'])

X = data_frame.drop(columns=['Nombre', 'Apellido', 'Edad', 'Apto'])
y = data_frame['Apto']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_pred_texto = labelEncoderApto.inverse_transform(y_pred)

print(y_pred_texto)
