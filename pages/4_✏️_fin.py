import streamlit as st

st.set_page_config(page_title='Final', page_icon='✏️')

st.title('Final de la aplicación web')
st.markdown('***:violet[Gracias por visitar mi aplicación web] :smile:***')

st.markdown('<h1 style="text-align: center;">Otros de mis proyectos</h1>', unsafe_allow_html=True)

st.markdown('<h2 style="text-align: center;">Regresor de pólizas de seguros</h2>', unsafe_allow_html=True)
st.image('images/ensurancerl.png')
st.markdown("""
            Este página web desplegada en HuggingFace sirve de FrontEnd para realizar
            predicciones en cuanto al cargo de la póliza de un seguro en función de
            varios parámetros como la edad, número de hijos, altura, sexo, etc. 
            """)

st.link_button(label='Llévame allí', url='https://huggingface.co/spaces/M0xn/ensurancerl')

st.markdown('<h2 style="text-align: center;">Analizador de calificaciones</h2>', unsafe_allow_html=True)
st.image('images/gradesanalizer.png')
st.markdown("""
            Este página web desplegada en HuggingFace sirve de FrontEnd para mostrar un
            análisis estadístico de una variable bidimensional orientado al análisis de las
            calificaciones de dos materias correlacionadas en una misma clase.  
            """)

st.link_button(label='Llévame allí', url='https://huggingface.co/spaces/M0xn/grades-analizer')