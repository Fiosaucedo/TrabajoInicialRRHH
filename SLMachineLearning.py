import pandas as pd
from pandas import read_excel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Revisar en la docu de scikit learn LabelEncoder que sirve para transformar labels no numéricos en numéricos.
#ej le.transform(["tokyo", "tokyo", "paris"])
#   array([2, 2, 1]...)
# o sea, tokyo vale 2 y paris 1.

candidatos = { #Back
    'Nombre': ["Juan", "Pedro", "Maria", "Ana"],
    'Apellido': ["Perez", "Garcia", "Garcia", "Garcia"],
    'Años de experiencia': [1, 5, 3, 13],
    'Habilidades': [1, 1, 1.5, 3],
    'Edad': [25, 30, 20, 25],
    'Apto': ["No", "Si", "No", "Si"]
}
df = pd.DataFrame(candidatos)

X = df[['Años de experiencia', 'Nivel educativo', 'Habilidades']]
y = df['Apto']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)
print(y_pred)
