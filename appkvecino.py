import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# Cargar datos
clientes= pd.read_csv("creditos.csv")

# Preparar datos
buenos = clientes[clientes["cumplio"] == 1]
malos = clientes[clientes["cumplio"] == 0]

datos = clientes[["edad", "credito"]]
clase = clientes["cumplio"]

# Escalar datos
escalador = preprocessing.MinMaxScaler()
datos = escalador.fit_transform(datos)

# Entrenar modelo KNN
clasificador = KNeighborsClassifier(n_neighbors=3)
clasificador.fit(datos, clase)

# Interfaz en Streamlit
st.title("Predicción de Crédito con KNN")
st.write("Ingrese los datos del solicitante para predecir si cumplirá con el pago.")

# Entrada de datos
edad = st.number_input("Edad del solicitante", min_value=18, max_value=100, value=30)
monto = st.number_input("Monto del crédito", min_value=10000, max_value=1000000, step=1000, value=50000)

if st.button("Predecir"):
    solicitante = escalador.transform([[edad, monto]])
    clase_predicha = clasificador.predict(solicitante)[0]
    probabilidad = clasificador.predict_proba(solicitante)
    
    st.write(f"### Predicción: {'Sí pagará' if clase_predicha == 1 else 'No pagará'}")
    st.write(f"Probabilidad de cumplimiento: {probabilidad[0][1]:.2%}")

    # Visualización
    fig, ax = plt.subplots()
    ax.scatter(buenos["edad"], buenos["credito"], marker="*", s=150, color="skyblue", label="Sí pagó")
    ax.scatter(malos["edad"], malos["credito"], marker="*", s=150, color="red", label="No pagó")
    ax.scatter(edad, monto, marker="P", s=250, color="green", label="Solicitante")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Monto del crédito")
    ax.legend()
    st.pyplot(fig)
