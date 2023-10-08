#jean paul ariza
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

 
#funcion para leer los datos 
def read_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    data.dropna(inplace=True)
    label_encoder = LabelEncoder()
    T_vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=5000)
    X = data['Email Text'].values
    y = data['Email Type'].values
    y = np.where(y == 'Phishing Email', 1, 0)
    y = label_encoder.fit_transform(y)
    return X, y

# funcion para imprimir datos
def print_report(y_val, y_pred, fold):
    print(f'Fold: {fold}')
    print(f'Accuracy Score: {accuracy_score(y_val, y_pred)}')
    print(f'Confusion Matrix: \n {confusion_matrix(y_val, y_pred)}')
    print(f'Classification Report: \n {classification_report(y_val, y_pred)}')

#Funcion para predecir un mensaje de phishing
def predict_phishing(email_text, model, vectorizer):
    # Realizar la transformación TF-IDF en el texto del correo electrónico
    email_text = [email_text]
    email_text = vectorizer.transform(email_text)
    
    # Realizar la predicción utilizando el modelo
    prediction = model.predict(email_text)
    
    # Devolver el resultado (1 para phishing, 0 para no phishing)
    return prediction[0]

#funcion para que el usuario ingrese un mensaje y valide si es phishing o no 
def main():
    # Cargar el modelo y el vectorizador entrenados
    model = XGBClassifier(n_estimators=800, learning_rate=0.1, max_depth=4, colsample_bytree=0.2, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), max_features=10000)
    vectorizer.fit(X_train)
    
    # Obtener el mensaje del usuario
    user_input = input("Ingrese el mensaje de correo electrónico: ")
    
    # Realizar la predicción
    prediction = predict_phishing(user_input, model, vectorizer)
    
    # Mostrar el resultado
    if prediction == 1:
        print("El mensaje es un correo de phishing.")
    else:
        print("El mensaje no es un correo de phishing.")

if __name__ == "__main__":
    main()



#se ingresa el dataset de correos con phising y no phising
data = pd.read_csv('C:/Jean Stuff/Programacion/redes neuronales/demo/proyecto/ia/Phishing_Email.csv')
print(data.head())
print(data['Email Type'].value_counts())

# se divide el dataset en conjuntos de entrenamiento y prueba
X, y = read_data('C:/Jean Stuff/Programacion/redes neuronales/demo/proyecto/ia/Phishing_Email.csv')
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
print(X.shape, y.shape)
#variable contador para la validación cruzada kfold
fold = 1

#entrenamieto e impresion de los entrenamientos
for train_index, val_index in kfold.split(X):

  X_train, X_val = X[train_index], X[val_index]
  y_train, y_val = y[train_index], y[val_index]

  vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), max_features=10000)
  vectorizer.fit(X_train)

  X_train = vectorizer.transform(X_train)
  X_val = vectorizer.transform(X_val)
  
  model = XGBClassifier(n_estimators=800, learning_rate=0.1, max_depth=4, colsample_bytree=0.2, n_jobs=-1, random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_val)

  print_report(y_val, y_pred, fold)
  fold += 1

