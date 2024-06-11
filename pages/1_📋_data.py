import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap 
from sklearn.model_selection import train_test_split
from pickle import dump 

st.set_page_config(page_title='Datos', page_icon='📋')

df = pd.read_csv('diabetes.csv')

st.header('Datos de trabajo')
st.markdown("""
            Esto son los datos con los que se va a trabajar
            para el entrenamiento de los clasificadores.
            En la barra lateral, puedes elegir las métricas
            y las gráficas correspondientes al EDA (Análisis
            Exploratorio de Datos).
            """)

with st.sidebar:
    st.title('Opciones del modelo')
    st.markdown("""
                Apartado para modificar los datos de entrada y
                modelos de clasificación.
                """)

    st.title('Datos')
    st.markdown("""
                *Selecciona aquí las características que van a 
                utilizar los modelos. Una vez elegidas, haz click
                sobre el botón "Seleccionar" para guardar tu elección.*
                """)
    features = st.multiselect(label='Características de entrada',
                              options=filter(lambda col: col != 'Outcome', df.columns))
    features.append('Outcome')
    df = df.loc[:, features]

    if st.button('Seleccionar', type='primary'):
        dump(df, open('selected_data.pkl', 'wb'))

    st.header('Información sobre los datos')
    st.markdown("""
                En esta sección puedes seleccionar aquellos elementos
                de información sobre los datos del DataFrame que
                quieres que se muestre en la pantalla principal 
                """)

    data_elements = st.multiselect(label='Elementos de información',
                                   options=['Tabla', 
                                            'Histogramas', 
                                            'Diagramas de caja y bigote', 
                                            'Mapa de correlación'])

    show_table = 'Tabla' in data_elements
    show_distrib = 'Histogramas' in data_elements
    show_boxplot = 'Diagramas de caja y bigote' in data_elements
    show_corrmap = 'Mapa de correlación' in data_elements

if show_table:
    st.dataframe(df.loc[:, features])

if show_distrib:
    fig, ax = plt.subplots()
    df.hist(ax=ax)
    st.pyplot(fig)

if show_boxplot:
    box_fig, box_ax = plt.subplots()
    plt.tight_layout()
    df.boxplot(ax=box_ax)
    st.pyplot(box_fig)

if show_corrmap:
    hm_fig, hm_ax = plt.subplots()
    hm_ax.grid(False)
    heatmap(data=df.corr(), annot=True, fmt='.2f', ax=hm_ax)
    st.pyplot(hm_fig)