# Archivo base para el despliegue del Agente en Streamlit
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

#En Streamlit vamos a agregar un titulo para la pagina web
st.title("Configuracion inicial")
#Agregamos un textbox en nuestra página web
st.write("Primera prueba de uso de streamlit y ambiente de MA2026")

#El slider de streamlit me permite ingresar un slider el parametro inversión

#Variables de nuestro modelo
gasto=st.slider("Seleccine nivel de gasto en publicicdad", 10,200,50)
variable_x = np.array([[10], [20], [30], [40],[50]])
variable_y = np.array([15,25,35,45,55])
modelo_lr = LinearRegression()

#Entrenamiento de nuestro modelo LR
modelo_lr.fit(variable_x,variable_y)

#En streamlit tenemos un botón que dice "Predecir" y al hacer click activara las lineas de código del if
if st.button("Predecir"):
 resultado = modelo_lr.predict([[gasto]])

#Streamlit muestra un mensaje de éxito en verde bonito usando success
 st.success(f"Las ventas proyectadas para una inversion de ${gasto} son: ${resultado[0]}")