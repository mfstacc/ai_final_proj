import streamlit as st
from pickle import load
import numpy as np

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split

from seaborn import regplot
import plotly.express as px
import plotly.graph_objects as go

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

    x_1_coef = model.coef_[0]
    x_2_coef = model.coef_[1]
    intercept = model.intercept_

    xs = df.loc[:, x_var]
    ys = df.loc[:, y_var]
    zs = df.loc[:, z_var]

    x_range = np.arange(xs.min(), xs.max(), 0.2)
    y_range = np.arange(ys.min(), ys.max(), 0.2)
    xx, yy = np.meshgrid(x_range, y_range)

    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)
    fig = px.scatter_3d(xs, ys, zs)
    fig.add_traces(go.Surface())

    st.plotly_chart(fig)

    st.latex(r'y=' + f'{x_1_coef:.3f}' + r'x_1' + ('+' if x_2_coef > 0 else '-') + f'{abs(x_2_coef):.3f}' + r'x_2' + ('+' if intercept > 0 else '-') + f'{abs(intercept):.2f}')
else:
    st.markdown('## Recta de regresión')

    coef = model.coef_[0]
    intercept = model.intercept_

    st.pyplot(regplot(data=df.iloc[:100], x=x_var, y=y_var).figure)
    st.latex(r'y=' + f'{coef:.3f}' + 'x' + ('+' if intercept > 0 else '-') + f'{abs(intercept):.2f}')