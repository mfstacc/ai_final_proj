import streamlit as st

st.set_page_config(page_title='Inicio', page_icon='🏠')

st.title('Aplicación final IA')
st.header('Introducción a la Inteligencia Artificial')

st.markdown("""
            *¿Qué es la Inteligencia Artificial?* Podríamos decir que la IA es una rama de la computación
            que se encarga del estudio de algoritmos de aprendizaje que tienen por objetivo llegar a un
            una predicción en base a una entrada.

            Estos algoritmos de aprendizaje pueden tener dos objetivos principales: regresión y clasificación.
            La regresión consiste en obtener un modelo matemático que permite hacer predicciones acerca de unos
            datos en base a una entrada. La clasificación por el otro lado, consiste en discernir identificar
            la clase a la que pertenece una entrada en concreto en base a las características de un conjunto de datos.

            En lo que respecta a la forma en la que estos modelos aprenden, se encuentran 3 tipos de aprendizaje
            esenciales: supervisado, por refuerzo y no supervisado. En el aprendizaje supervisado, el modelo aprende
            mediante conjuntos de datos etiquetados, esto es, conjuntos de datos con características diferenciadas
            de las que el modelo puede "aprender" para hacer una predicción. En el aprendizaje por refuerzo, el modelo
            aprende gracias a un "bucle de feedback retroalimentativo", es decir, el modelo aprende gracias al resultado
            que obtiene de sus predicciones. Este aprendizaje sigue siendo en cierta parte supervisado, puesto que debe
            haber un operador que pueda dar indicaciones al modelo sobre lo que está bien y lo que está mal en lo que ha
            predicho. Y por último, en el aprendizaje no supervisado, el modelo aprende las características de un conjunto
            de datos a través de su análsis. Por tanto, este tipo de modelos no requieren de ningún tipo de operador que los
            supervise o proporcione las características de los datos, los propios modelos son los encargados de extraerlas.

            Dentro de la IA encontramos dos ramas de desarrollo: el Aprendizaje Automático o Machine Learning y el Aprendizaje
            Profundo o Deep Learning. Modelos propios del Machine Learning son aquellos con aprendizaje supervisado o 
            por refuerzo. Modelos propios del Deep Learning son aquellos no supervisados. Por lo que la diferencia es clara,
            los modelos de Machine Learning apreden con la supervisión de un tercero y datos con características ya identificadas
            y los modelos de Deep Learning aprenden sin supervisión alguna ni características previamente identificadas.

            Dentro de cada rama existen distintos tipos de algoritmos para realizar las tareas de predicción (*ya sean para 
            regresión o clasificación*). En este aplicación web se tratan algunos ejemplos de algoritmos de Machine Learning
            que son: Regresión Lineal, Regresión Logística, Árbol de Decisión y Perceptrón (Red Neuronal simple).
            """)

st.markdown("""
            """)
