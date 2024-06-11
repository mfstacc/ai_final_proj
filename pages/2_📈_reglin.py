import streamlit as st
from pickle import load
import numpy as np

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split

from seaborn import regplot
import matplotlib.pyplot as plt

st.set_page_config(page_title='Regresión Lineal', page_icon='📈')
st.title('Regresión Lineal')

df = load(open('selected_data.pkl', 'rb'))

coef = intercept = 0 

models = {
    'LinearRegression': LinearRegression,
    'Lasso': LassoCV,
    'Ridge': RidgeCV
}

with st.sidebar:
    selected_model = st.selectbox(label='Modelo',
                                  options=models.keys())

    x_var = st.selectbox(label='Variable X', 
                         options=filter(lambda col: col != 'Outcome', df.columns))
    y_var = st.selectbox(label='Variable Y',
                         options=filter(lambda col: col != 'Outcome', df.columns))

    selected_features = [x_var, y_var]

    three_dim_regression = st.checkbox(label='Utilizar otra variable')

    if three_dim_regression:
        z_var = st.selectbox(label='Variable Z',
                             options=filter(lambda col: col != 'Outcome', df.columns))

        selected_features.append(z_var)

    X = df.loc[:, selected_features]
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

    model = models[selected_model]()
    model.fit(X_train, y_train)

if three_dim_regression:
    st.markdown('## Hiperplano de regresión')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_1_coef = model.coef_[0]
    x_2_coef = model.coef_[1]
    intercept = model.intercept_

    xs = df.loc[:, x_var]
    ys = df.loc[:, y_var]
    zs = df.loc[:, z_var]

    ax.scatter(xs, ys, zs)
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)
    ax.set_zlabel(z_var)

    plt.tight_layout()
    st.pyplot(fig)

    st.latex(r'y=' + f'{x_1_coef:.3f}' + r'x_1' + ('+' if x_2_coef > 0 else '-') + f'{abs(x_2_coef):.3f}' + r'x_2' + ('+' if intercept > 0 else '-') + f'{abs(intercept):.2f}')
else:
    st.markdown('## Recta de regresión')

    coef = model.coef_[0]
    intercept = model.intercept_

    st.pyplot(regplot(data=df.iloc[:100], x=x_var, y=y_var).figure)
    st.latex(r'y=' + f'{coef:.3f}' + 'x' + ('+' if intercept > 0 else '-') + f'{abs(intercept):.2f}')