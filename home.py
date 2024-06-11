import streamlit as st

st.set_page_config(page_title='Inicio', page_icon='🏠')

st.title('Aplicación final IA')
st.header('Introducción a la Inteligencia Artificial')

st.markdown("""
            *¿Qué es la Inteligencia Artificial?* Podríamos decir que la IA es una **rama de la computación**
            que se encarga del estudio de **algoritmos de aprendizaje** que tienen por objetivo llegar a un
            una **predicción** en base a una entrada.

            Estos algoritmos de aprendizaje pueden tener **2 objetivos principales**: **:orange[regresión]** y **:green[clasificación]**.
            La **:orange[regresión]** consiste en obtener un **:orange[modelo matemático]** que permite hacer **:orange[predicciones]** acerca de unos
            datos en base a una entrada. La **:green[clasificación]** por el otro lado, consiste en **:green[identificar la clase]** a la
            que pertenece una entrada en concreto en base a las características de un conjunto de datos.
            """)

st.image('images/classregr.jpg')

st.markdown("""
            En lo que respecta a **la forma en la que estos modelos aprenden**, se encuentran **3 tipos de aprendizaje**
            esenciales: **:violet[supervisado]**, **:blue[por refuerzo]** y **:red[no supervisado]**. En el **:violet[aprendizaje supervisado]**,
            el modelo aprende mediante **:violet[conjuntos de datos etiquetados]**, esto es, conjuntos de datos con características diferenciadas
            de las que el modelo puede "aprender" para hacer una predicción. En el **:blue[aprendizaje por refuerzo]**, el modelo
            aprende gracias a un ***:blue[bucle de feedback retroalimentativo]***, es decir, el modelo aprende gracias al resultado
            que obtiene de sus predicciones. Este aprendizaje sigue siendo en cierta parte supervisado, puesto que debe
            haber un operador que pueda dar indicaciones al modelo sobre lo que está bien y lo que está mal en lo que ha
            predicho. Y por último, en el **:red[aprendizaje no supervisado]**, el modelo **:red[aprende]** las características de un conjunto
            de datos a través de su **:red[análsis]**. Por tanto, este tipo de modelos **:red[no requieren de ningún tipo de operador]** que los
            supervise o proporcione las características de los datos, los propios modelos son los encargados de extraerlas.
            """)

st.image('images/superunreinforced.png')

st.markdown("""
            Dentro de la IA encontramos dos ramas de desarrollo: el **:green[Aprendizaje Automático o Machine Learning]** y el 
            **:red[Aprendizaje Profundo o Deep Learning]**. Modelos propios del Machine Learning son aquellos con **:green[aprendizaje supervisado]** o 
            **:green[por refuerzo]**. Modelos propios del Deep Learning son **:red[aquellos no supervisados]**. Por lo que la diferencia es clara,
            los modelos de **:green[Machine Learning]** apreden con la **:green[supervisión de un tercero y datos con características ya identificadas]**
            y los modelos de **:red[Deep Learning]** aprenden **:red[sin supervisión alguna ni características previamente identificadas]**.
            """)

st.image('images/mlvsdeepl.png')

st.markdown("""
            Dentro de cada rama existen **distintos tipos** de algoritmos para realizar las tareas de predicción (*ya sean para 
            regresión o clasificación*). En este aplicación web se tratan algunos ejemplos de algoritmos de **Machine Learning**
            que son: **:orange[Regresión Lineal], :green[Regresión Logística], :green[Árbol de Decisión] y :green[Perceptrón (Red Neuronal simple)]**.
            """)