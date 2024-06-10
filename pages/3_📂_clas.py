import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pickle import load

from seaborn import heatmap
from matplotlib.pyplot import plot, subplots 

st.set_page_config(page_title='Regresión Logística', page_icon='📂')
st.title('Modelos de clasificación')

df = load(open('selected_data.pkl', 'rb'))

models = {
    'Regresión Logística': LogisticRegression,
    'Árbol de Clasificación': DecisionTreeClassifier,
    'Perceptrón Multicapa': MLPClassifier
}

solver_penalty = {
    'lbfgs': ['l2', None],
    'liblinear': ['l1', 'l2'],
    'newton-cg': ['l2', None],
    'newton-cholesky': ['l2', None],
    'sag': ['l2', None],
    'saga': ['elasticnet', 'l1', 'l2', None]
}

get_possible_solvers = lambda penalty: [k for k, v in solver_penalty.items() if penalty in v]
classify = False

with st.sidebar:
    st.title('Opciones de modelos')

    selected_model = st.selectbox('Modelo', 
                                  options=models.keys())
    
    model = models[selected_model]

    match selected_model:
        case 'Regresión Logística':
            penalty = st.selectbox('Penalización', 
                                   options=['l2', 'l1', 'elasticnet', None])

            c = st.number_input('C', min_value=0.1, max_value=2.0)
            solver = st.selectbox('Solver',
                                  options=get_possible_solvers(penalty))

            model = models[selected_model](penalty=penalty,
                                           C=c,
                                           solver=solver,
                                           l1_ratio=0.5)

        case 'Árbol de Clasificación':
            max_depth = st.number_input('Profunidad máxima del árbol', 
                                        min_value=1, max_value=3)
            criterion = st.selectbox('Criterio de división',
                                     options=['gini', 'entropy'])

            model = models[selected_model](criterion=criterion,
                                           max_depth=max_depth)
        case 'Perceptrón Multicapa':
            hidden_layers = st.number_input('Número de capas ocultas',
                                            min_value=0, max_value=100,
                                            value=50)
            activation_fn = st.selectbox('Función de activación',
                                         options=['identity', 'logistic', 'tanh', 'relu'])
            solver = st.selectbox('Solver',
                                  options=['lbfgs', 'sgd', 'adam'])
            solver = 'adam'
            learning_rate = st.slider('Tasa de aprendizaje',
                                      min_value=0.001, max_value=1.000, value=0.05)
            epochs = st.slider('Número de épocas',
                               min_value=5, max_value=300, value=100)
            
            model = models[selected_model](hidden_layer_sizes=(hidden_layers,),
                                           activation=activation_fn,
                                           solver=solver,
                                           learning_rate_init=learning_rate,
                                           max_iter=epochs)
        
    if st.button('Clasificar', type='primary'):
        classify = True
        X = df.loc[:, df.columns != 'Outcome']
        y = df['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        conf_matrix_data = confusion_matrix(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_score) 
        precision, recall, _ = precision_recall_curve(y_test, y_score)

if classify:
    st.markdown('## Matriz de confusión')
    conf_matrix = heatmap(conf_matrix_data, annot=True, fmt='.0f')
    st.pyplot(conf_matrix.figure)

    st.markdown('## Curva ROC')
    fig, ax = subplots()
    ax.plot(fpr, tpr)
    st.pyplot(fig)


    st.markdown('## Curva Precision-Recall')
    fig_prc, ax_prc = subplots()
    ax_prc.plot(precision, recall)
    st.pyplot(fig_prc)

    accuracy = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test)
    auc = roc_auc_score(y_pred, y_score) 

    acc, f1_s, auc_s= st.columns(3)

    acc.metric('Precisión', f'{accuracy*100:.2f}%')
    f1_s.metric('F1-score', round(f1, 2))
    auc_s.metric('AUC:', round(auc, 2))