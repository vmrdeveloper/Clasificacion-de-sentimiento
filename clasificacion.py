import numpy as np
import pandas as pd
import re
import seaborn as sn
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report





# Cargamos nuestro DATA SET y revisamos el contenido del mismo
opinions_df = pd.read_csv('Twitter_Data.csv')
opinions_df

# Preprocesamiento del texto, no se usara porque el data set que tenemos esta limpio
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r"[^a-z\s]", "", text)
    # Eliminar palabras vacías (stopwords)
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lematización
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)



# Convertir los sentimientos a valores numéricos
label_mapping = {"Negativo": 0, "Neutro": 1, "Positivo": 2}
opinions_df['label'] = opinions_df['category'].map(label_mapping)

opinions_df

# Vectorizamos utilizando TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(opinions_df['clean_text'])
y = opinions_df['label']

# Dividimos los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamos el modelo de Regresión Logística
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Aqui evaluamos el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"])

accuracy, report
