import pandas as pd
import streamlit as st
import altair as alt
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('punkt')
from nltk import tokenize
from tqdm import trange
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from collections import Counter
nltk.download('stopwords')
from streamlit_option_menu import option_menu
from joblib import load
import os
from PIL import Image
    
#Datos-------
def leer_archivo(nombre_archivo):
    try:
        with open(nombre_archivo, "r", encoding="utf-8") as file:
            lineas = file.readlines()
    except UnicodeDecodeError:
        with open(nombre_archivo, "r", encoding="latin1") as file:
            lineas = file.readlines()

    # Convertir a DataFrame y limpiar contenido
    df = pd.DataFrame(lineas, columns=["Contenido"])
    df["Contenido"] = df["Contenido"].str.strip()
    return df

try:
    train = leer_archivo("thai_nlp/train.txt")
    train_label = leer_archivo("thai_nlp/train_label.txt")
    test = leer_archivo("thai_nlp/test.txt")
    test_label = leer_archivo("thai_nlp/test_label.txt")
except FileNotFoundError as e:
    st.error(f"Error: {e}")
    st.stop()
    
def clean(review):
    
    review = review.lower()
    review = re.sub('[^a-z A-Z 0-9-]+', '', review)
    review = " ".join([word for word in review.split() if word not in stopwords.words('english')])
    
    return review

def corpus(text):
    text_list = text.split()
    return text_list

train["Label"] = train_label["Contenido"]
test["Label"] = test_label["Contenido"]

df = pd.concat([train, test], ignore_index=True)


#df['Comment'] = df['Contenido'].apply(clean)
#df['Comment_Lists'] = df['Contenido'].apply(corpus)

df['comment_length'] = df["Contenido"].astype(str).apply(len)
df['Word_count'] = df["Contenido"].apply(lambda x: len(str(x).split()))


#APP----------

custom_style = """
<style>
/* Cambiar el color de fondo de todo el dashboard */
[data-testid="stAppViewContainer"] {
    background-color: #E5EFF0;
}

/* Ajustar el fondo del contenido principal */
[data-testid="stApp"] {
    background-color: #E5EFF0;
    padding: 0;
    display: flex;
    justify-content: center;
    align-items: center;
}

.stAppViewBlockContainer {
        padding-top: 0rem;
            }

.sidebar .sidebar-content h2 {
        color: #ff6347; /* Cambia el color aquí */
        font-size: 20px; /* Ajusta el tamaño */
        font-weight: bold; /* Negrita */
    }

</style>
"""

# Aplicar el CSS combinado
st.markdown(custom_style, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",  # Título del menú
        options=["Dashboard", "Models"],  # Opciones del menú
        icons=["bar-chart", "robot"],  # Iconos (de https://icons.getbootstrap.com/)
        menu_icon="cast",  # Icono del menú principal
        default_index=0,  # Página por defecto
    )

st.sidebar.header("Filter Options")
options = ["Positive", "Negative", "Neutral", "Question"]
selection = st.sidebar.multiselect(
    "Select Sentiment", options, default=options
)


#Filtros-----
label_mapping = {
    "Positive": "pos",
    "Negative": "neg",
    "Neutral": "neu",
    "Question": "q",
}

if selected == "Dashboard":
    
    st.title("Sentiment Analysis")

    if selection:
        mapped_selection = [label_mapping[opt] for opt in selection]
        filtered_df = df[df["Label"].isin(mapped_selection)]
    else:
        filtered_df = df

    #most_common = Counter(corpus).most_common(10)
    #words = [word for word, count in most_common]
    #freq = [count for word, count in most_common]

    # Mostrar conteo
    #st.bar_chart(pd.DataFrame({"Words": words, "Frequency": freq}).set_index("Words"))

    #Métricas--------
    puntos = {
    'pos': 2,
    'neu': 1,
    'neg': 0,
    'q': 0 
    }
    filtered_df['Points'] = filtered_df['Label'].map(puntos)
    grade = round(filtered_df['Points'].sum() / (len(filtered_df) * 2), 2)
    
    col1, col2= st.columns(2)
    col1.metric("Number of Comments", len(filtered_df))
    col2.metric("Comments Score", grade)


    #Gráficos------
    # Crear columnas
    #col1, col2 = st.columns(2)

    # Gráfico de distribución de etiquetas (dona) en la primera columna
    label_counts = filtered_df["Label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    fig_pie = alt.Chart(label_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Label", type="nominal"),
            tooltip=["Label", "Count"]
        ).properties(
            title="Distribution of Sentiment Labels"
        )
    st.altair_chart(fig_pie, use_container_width=True)

    # Histograma de longitud de comentarios en la segunda columna
    fig_hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X("comment_length", bin=alt.Bin(maxbins=30), title="Comment Length (characters)"),
            alt.Y("count()", title="Frequency"),
            tooltip=["count()"]
        ).properties(
            title="Histogram of Comment Lengths"
        )
    st.altair_chart(fig_hist, use_container_width=True)

    # Histograma del número de palabras en la tercera columna

    fig_word_hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X("Word_count", bin=alt.Bin(maxbins=30), title="Number of Words"),
            alt.Y("count()", title="Frequency"),
            tooltip=["count()"]
        ).properties(
            title="Histogram of Number of Words"
        )
    st.altair_chart(fig_word_hist, use_container_width=True)



    def visualize(col):
        # Boxplot
        boxplot = alt.Chart(df).mark_boxplot().encode(
            x=alt.X('Label:N', title="Sentiment Label"),
            y=alt.Y(f'{col}:Q', title=col),
            color='Label:N'
        ).properties(
            title=f"Boxplot of {col} by Sentiment",
            width=400,
            height=300
        )
        
        # KDE plot (Density Plot)
        kdeplot = alt.Chart(df).transform_density(
            col, 
            groupby=['Label'],
            as_=[col, 'density']
        ).mark_area(opacity=0.5).encode(
            x=alt.X(f'{col}:Q', title=col),
            y=alt.Y('density:Q', title='Density'),
            color='Label:N'
        ).properties(
            title=f"KDE Plot of {col} by Sentiment",
            width=400,
            height=300
        )
        
        return boxplot, kdeplot

    features = ['comment_length', 'Word_count']

    for feature in features:
        
        boxplot, kdeplot = visualize(feature)

        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(boxplot, use_container_width=True)
        with col2:
            st.altair_chart(kdeplot, use_container_width=True)

else:
    st.title("Deep Learning Models")
    st.subheader('Resultados LSTM')

    if os.path.exists('history_lstm.joblib'):

        history_lstm = load('history_lstm.joblib')
        
        # Preparar los datos para el gráfico
        history_df = pd.DataFrame({
            'epoch': list(range(1, len(history_lstm['accuracy']) + 1)),
            'accuracy': history_lstm['accuracy'],
            'loss': history_lstm['loss']
        })

        history_melted = history_df.melt(id_vars='epoch', var_name='metric', value_name='value')
        
        acc_chart = alt.Chart(history_melted).mark_line(point=True).encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('metric:N', title='Metric'),
            tooltip=['epoch:Q', 'metric:N', 'value:Q']
        ).properties(
            title='Model Training Metrics',
            width=600,
            height=400
        )

        st.altair_chart(acc_chart, use_container_width=True)
    else:
        st.error("El archivo 'history_lstm.joblib' no existe. Por favor, entrena el modelo primero.")
    
    image_lstm = Image.open("confusion_matrix_lstm.png")
    image_resized_lstm = image_lstm.resize((500, 500))
    st.image(image_resized_lstm)
        
    st.subheader('Resultados RNN')
    
    if os.path.exists('history_RNN.joblib'):

        history_rnn = load('history_RNN.joblib')
        
        # Preparar los datos para el gráfico
        history_df_RNN = pd.DataFrame({
            'epoch': list(range(1, len(history_lstm['accuracy']) + 1)),
            'accuracy': history_lstm['accuracy'],
            'loss': history_lstm['loss']
        })

        history_melted = history_df_RNN.melt(id_vars='epoch', var_name='metric', value_name='value')
        
        acc_chart = alt.Chart(history_melted).mark_line(point=True).encode(
            x=alt.X('epoch:Q', title='Epoch'),
            y=alt.Y('value:Q', title='Value'),
            color=alt.Color('metric:N', title='Metric'),
            tooltip=['epoch:Q', 'metric:N', 'value:Q']
        ).properties(
            title='Model Training Metrics',
            width=600,
            height=400
        )

        st.altair_chart(acc_chart, use_container_width=True)
    
        
    else:
        st.error("El archivo 'history_lstm.joblib' no existe. Por favor, entrena el modelo primero.")
    
    image = Image.open("confusion_matrix_rnn.png")
    image_resized = image.resize((500, 500))
    st.image(image_resized)
        
    





