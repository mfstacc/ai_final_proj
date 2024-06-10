import streamlit as st
from pickle import load

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split

from seaborn import regplot

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

    X = df.loc[:, df.columns != 'Outcome']
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)

    model = models[selected_model]()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    coef = model.coef_[0]
    intercept = model.intercept_.ravel()[0]

st.markdown('## Recta de regresión')

st.pyplot(regplot(data=df.iloc[:100], x=x_var, y=y_var).figure)
st.latex(r'y=' + str(round(coef, 2)) + 'x' + '+' + str(round(intercept, 2)))