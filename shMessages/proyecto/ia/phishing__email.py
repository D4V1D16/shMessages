#jean paul
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from googletrans import Translator



##NLP en español e ingles
nlp_es = spacy.load("es_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")
##


# Función para preprocesar el texto en español e inglés
def preprocess_text(text, language):
    if language == 'es':
        doc = nlp_es(text)
    elif language == 'en':
        doc = nlp_en(text)
    else:
        raise ValueError("Idioma no compatible. Utilice 'es' o 'en'.")

    # Lematizar y eliminar palabras vacías
    processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop])

    return processed_text


# Función para leer los datos
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

# Función para visualizar los parametros del modelo
def print_report(y_val, y_pred, fold):
    print(f'Fold: {fold}')
    print(f'Accuracy Score: {accuracy_score(y_val, y_pred)}')
    print(f'Confusion Matrix: \n {confusion_matrix(y_val, y_pred)}')
    print(f'Classification Report: \n {classification_report(y_val, y_pred)}')

# Función para predecir si un mensaje es phishing o no
def predict_phishing(email_text, model, vectorizer):
    # Realizar la transformación TF-IDF en el texto del correo electrónico
    email_text = [email_text]
    email_text = vectorizer.transform(email_text)
    
    # Realizar la predicción utilizando el modelo
    prediction = model.predict(email_text)
    
    # Devolver el resultado (1 para phishing, 0 para no phishing)
    return prediction[0]

# Función para cargar o entrenar el modelo
def load_or_train_model():
    model = None
    X_train, y_train = [], []

    try:
        with open('X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = joblib.load(f)
        with open('model.pkl', 'rb') as f:
            model = joblib.load(f)
        
        print("\nDatos de entrenamiento y modelo cargados.\n")
    except FileNotFoundError:
        print("\nEntrenamiento necesario. Entrenando el modelo y guardando los datos de entrenamiento...\n")
        
        # Se ingresa el dataset de correos con phishing y no phishing
        data = pd.read_csv('./Phishing_Email.csv')
        print(data.head())
        print(data['Email Type'].value_counts())

        # Se divide el dataset en conjuntos de entrenamiento y prueba
        X, y = read_data('./Phishing_Email.csv')
        num_folds = 5
        kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
        print(X.shape, y.shape)


        # Variable contador para la validación cruzada kfold
        fold = 1

        # Entrenamiento e impresión de los entrenamientos
        for train_index, val_index in kfold.split(X,y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 3), max_features=10000)
            vectorizer.fit(X_train)

            X_train = vectorizer.transform(X_train)
            X_val = vectorizer.transform(X_val)
            
            # Calcula las ponderaciones de clase
            class_weights = len(y_train) / (2 * np.bincount(y_train))

            # Crea el modelo XGBoost con ponderación de clases
            model = XGBClassifier(n_estimators=800, learning_rate=0.1, max_depth=4, colsample_bytree=0.2,n_jobs=-1, random_state=42, scale_pos_weight=class_weights[1])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            print_report(y_val, y_pred, fold)

            # Llama a la función de visualización
            plot_results(y_val, y_pred, fold)
            
            fold += 1

        # Guardar los datos de entrenamiento y el modelo entrenado
        with open('X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open('y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open('vectorizer.pkl', 'wb') as f:
            joblib.dump(vectorizer, f)
        with open('model.pkl', 'wb') as f:
            joblib.dump(model, f)
    

        print("Modelo entrenado y datos de entrenamiento guardados.")
        
    return model, vectorizer

# Función para graficar los resultados
def plot_results(y_val, y_pred, fold):
    # Matriz de confusión
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicciones")
    plt.ylabel("Valores reales")
    plt.title(f'Matriz de Confusión - Fold {fold}')
    plt.show()

    # Gráfico de precisión y recall
    report = classification_report(y_val, y_pred, output_dict=True)
    accuracy = report['accuracy']
    precision = report['1']['precision']
    recall = report['1']['recall']

    plt.figure(figsize=(8, 6))
    plt.bar(['Precisión', 'Recall', 'Exactitud'], [precision, recall, accuracy], color=['blue', 'green', 'orange'])
    plt.ylabel("Puntuación")
    plt.title(f'Precisión, Recall y Exactitud - Fold {fold}')
    plt.show()


# Función principal
def main():
    model, vectorizer = load_or_train_model()
    translator = Translator()
    
    # Interacción con el usuario
    while True:
        user_input = input("Ingrese un mensaje de correo electrónico (o escriba 'salir' para salir): ")
        while not user_input:
            print("No has escrito nada. Ingrese un correo válido.")
            user_input = input("Ingrese un mensaje de correo electrónico (o escriba 'salir' para salir): ")
        
        if user_input.lower() == 'salir':
            break
        
        try:
            # Traduce el texto al inglés
            translated_input = translator.translate(user_input, src='es', dest='en')
            english_text = translated_input.text

            # Preprocesa el texto traducido en inglés
            processed_input = preprocess_text(english_text)
            result = predict_phishing(processed_input, model, vectorizer)
            if result == 1:
                print("\n El mensaje SI es un correo de phishing. \n")
            else:
                print("\n El mensaje NO es un correo de phishing. \n")
        except Exception as e:
            print(f"Error al traducir o procesar el texto: {str(e)}")

if __name__ == "__main__":
    main()
