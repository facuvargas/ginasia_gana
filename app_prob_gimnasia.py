# Copia aquí todo el código de tu aplicación Streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


gimnasia = pd.read_excel("ginasia.xlsx")
# Crear una función para contar la cantidad de goles a favor y en contra en cada fila
def contar_goles_por_fila(row):
    goles_favor = sum(1 for col in row.index if 'gol_favor' in col and not pd.isna(row[col]))
    goles_contra = sum(1 for col in row.index if 'en_con' in col and not pd.isna(row[col]))
    return goles_favor - goles_contra  # Calcular la Diferencia de Goles

# Aplicar la función a cada fila para contar los goles y calcular la Diferencia de Goles
gimnasia['Diferencia_Goles'] = gimnasia.apply(contar_goles_por_fila, axis=1)
# Crear la variable "Resultado del Partido" en base a la diferencia de goles
gimnasia['Resultado_Partido'] = gimnasia['Diferencia_Goles'].apply(lambda x: 'Victoria' if x > 0 else ('Empate' if x == 0 else 'Derrota'))
# Crear una nueva columna 'Resultado' que codifique los resultados como números (0 para Derrota, 1 para Empate, 2 para Victoria)
gimnasia['Resultado'] = gimnasia['Resultado_Partido'].apply(lambda x: 0 if x == 'Derrota' else (1 if x == 'Empate' else 2))


# Calcular la diferencia en días entre cada partido
gimnasia['Fecha'] = pd.to_datetime(gimnasia['Fecha'], format='%d-%b')
gimnasia['Dias_Descanso'] = gimnasia['Fecha'].diff().dt.days

# Asegurarse de que 'Dia_Semana' sea de tipo categórico
gimnasia['Dia_Semana'] = gimnasia['Fecha'].dt.day_name()
gimnasia['Dia_Semana'] = gimnasia['Dia_Semana'].astype('category')

# Codificar la variable categórica 'Dia_Semana' en variables ficticias
gimnasia = pd.get_dummies(gimnasia, columns=['Dia_Semana'], drop_first=True)
gimnasia['Minutos'] = np.random.randint(0, 91, len(gimnasia))

# Definir las variables predictoras X

# Transformar la variable categórica 'Condicion' en una variable binaria
gimnasia['Condicion'] = gimnasia['Condicion'].apply(lambda x: 1 if x == 'Local' else 0)
gimnasia['primer_gol'] = gimnasia['primer_gol'].apply(lambda x: 1 if x == 'Local' else 0)

X = gimnasia[['Diferencia_Goles', 'Condicion', 'descanso' ,'Dia_Semana_Saturday', 'Minutos']]


# La columna objetivo es 'Resultado'
y = gimnasia['Resultado']
# Crear un modelo de regresión logística multinomial (Softmax)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter = 10000)


# Ajustar el modelo (entrenamiento)
model.fit(X, y)

# Cargar tu modelo previamente entrenado
# Nota: Asegúrate de tener 'model' definido y entrenado como lo has hecho en tu código anterior.

# Título de la aplicación
st.title("Gana el lobito")

# Entrada para minutos
minutos = st.number_input("Minutos:", min_value=0, max_value=90)

# Entrada para diferencia de goles
diferencia_goles = st.number_input("Diferencia de Goles:")

# Botón para calcular probabilidades
if st.button("Calcular Probabilidades"):
    nueva_situacion = pd.DataFrame({
        'Diferencia_Goles': [diferencia_goles],
        'Condicion': [1],  # Puedes cambiar esto según la condición deseada (Local o Visitante)
        'descanso': [8],
        'Dia_Semana_Saturday': [1],
        'Minutos': [minutos]
    })

    probabilidades = model.predict_proba(nueva_situacion)

    st.write(f"Probabilidad de Derrota: {probabilidades[0][0]:.2%}")
    st.write(f"Probabilidad de Empate: {probabilidades[0][1]:.2%}")
    st.write(f"Probabilidad de Victoria: {probabilidades[0][2]:.2%}")
