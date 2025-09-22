import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# Configuración inicial
# ------------------------------
st.set_page_config(
    page_title="🌱 Recomendador de Cultivos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Cargar datos
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")  # Cambia el nombre si tu archivo es distinto

data = load_data()

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("Información del dataset")
st.sidebar.markdown(f"**Filas:** {data.shape[0]}")
st.sidebar.markdown(f"**Columnas:** {data.shape[1]}")

# Filtros interactivos
st.sidebar.header("Filtros")
numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

filters = {}
for col in numeric_cols:
    min_val, max_val = data[col].min(), data[col].max()
    sel_range = st.sidebar.slider(
        f"{col}",
        float(min_val), float(max_val),
        (float(min_val), float(max_val))
    )
    filters[col] = sel_range

# Aplicar filtros
filtered_data = data.copy()
for col, (min_val, max_val) in filters.items():
    filtered_data = filtered_data[
        (filtered_data[col] >= min_val) & (filtered_data[col] <= max_val)
    ]

# ------------------------------
# Sección principal
# ------------------------------
st.title("🌱 Recomendador de Cultivos")
st.markdown("Explora cómo las condiciones de suelo y clima influyen en el tipo de cultivo ideal.")

st.subheader("Vista previa de los datos filtrados")
st.dataframe(filtered_data)

# ------------------------------
# Visualizaciones
# ------------------------------
st.subheader("Distribución de variables")
selected_var = st.selectbox("Selecciona una variable para graficar:", numeric_cols)

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(filtered_data[selected_var], kde=True, ax=ax)
st.pyplot(fig)

# Relación con el cultivo
if "label" in data.columns:  # Asegúrate de que tu dataset tenga esta columna
    st.subheader("Relación entre variable y tipo de cultivo")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x="label", y=selected_var, data=filtered_data, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.warning("El dataset no tiene columna 'label' para el tipo de cultivo.")
