# Copia aquí todo el código de tu aplicación Streamlit
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

## ABRIR EL DF ##########
gimnasia = pd.read_excel("ginasia.xlsx")



####Funcion que cuenta la cantidad de goles
def contar_goles_por_fila(row):
    goles_favor = sum(1 for col in row.index if 'gol_favor' in col and not pd.isna(row[col]))
    goles_contra = sum(1 for col in row.index if 'en_con' in col and not pd.isna(row[col]))
    return goles_favor - goles_contra  # Calcular la Diferencia de Goles

# Aplicar la función 
gimnasia['Diferencia_Goles'] = gimnasia.apply(contar_goles_por_fila, axis=1)

# Crear la variable "Resultado del Partido" en base a la diferencia de goles
gimnasia['Resultado_Partido'] = gimnasia['Diferencia_Goles'].apply(lambda x: 'Victoria' if x > 0 else ('Empate' if x == 0 else 'Derrota'))

# Crear una nueva columna 'Resultado' que codifique los resultados como números (0 para Derrota, 1 para Empate, 2 para Victoria)

gimnasia['Resultado'] = gimnasia['Resultado_Partido'].apply(lambda x: 0 if x == 'Derrota' else (1 if x == 'Empate' else 2))


# Calcular la diferencia en días entre cada partido
gimnasia['Fecha'] = pd.to_datetime(gimnasia['Fecha'], format='%d-%b')
gimnasia['Dias_Descanso'] = gimnasia['Fecha'].diff().dt.days

# 'Dia de la semana as factor
gimnasia['Dia_Semana'] = gimnasia['Fecha'].dt.day_name()
gimnasia['Dia_Semana'] = gimnasia['Dia_Semana'].astype('category')

# Codificar la variable categórica 'Dia_Semana' en variables ficticias
gimnasia = pd.get_dummies(gimnasia, columns=['Dia_Semana'], drop_first=True)
gimnasia['Minutos'] = np.random.randint(0, 99, len(gimnasia))

# Definir las variables predictoras X

# Jugamos de local o de visitante
gimnasia['Condicion'] = gimnasia['Condicion'].apply(lambda x: 1 if x == 'Local' else 0)


# Gener vector de variables 
X = gimnasia[['Diferencia_Goles', 'Condicion', 'descanso' ,'Dia_Semana_Saturday', 'Minutos']]

# 'Resultado': variable a estimar
y = gimnasia['Resultado']
# Modelo logit
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter = 10000)


# Ajustar el modelo (entrenamiento)
model.fit(X, y)


# Aplicación
st.title("Gana el lobito")



# Crear una lista de opciones para la condición
opciones_condicion = ['Local', 'Visitante']

# Agregar el widget de selección para la condición
condicion_elegida = st.radio("Selecciona la Condición:", opciones_condicion)

# Entrada para minutos
minutos = st.number_input("Minutos:", min_value=0, max_value=98)

# Entrada para diferencia de goles
diferencia_goles = st.number_input("Diferencia de Goles:", step=1)

# Botón para calcular probabilidades
if st.button("Calcular Probabilidades"):
    # Aquí puedes usar la variable 'condicion_elegida' en lugar de un valor fijo
    nueva_situacion = pd.DataFrame({
        'Diferencia_Goles': [diferencia_goles],
        'Condicion': [1 if condicion_elegida == 'Local' else 0],
        'descanso': [8],
        'Dia_Semana_Saturday': [1],
        'Minutos': [minutos]
    })

    probabilidades = model.predict_proba(nueva_situacion)

    st.write(f"Probabilidad de Derrota: {probabilidades[0][0]:.2%}")
    st.write(f"Probabilidad de Empate: {probabilidades[0][1]:.2%}")
    st.write(f"Probabilidad de Victoria: {probabilidades[0][2]:.2%}")
