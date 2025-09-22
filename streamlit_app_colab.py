%%bash
cat > streamlit_app_colab.py <<'PY'
# streamlit_app_colab.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Recomendador de Cultivos", layout="wide")

@st.cache_data
def load_data(path="datos.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

data = load_data()

if data.empty:
    st.title("❗ No se encontró 'datos.csv' en el directorio")
    st.write("Sube un `datos.csv` al entorno o ponlo en la raíz del repo antes de desplegar.")
    st.stop()

st.title("🌱 Panel interactivo — Recomendador de Cultivos")
st.markdown("Explora cómo las condiciones de suelo y clima se relacionan con el cultivo ideal.")

# Sidebar información
st.sidebar.header("Información del dataset")
st.sidebar.markdown(f"- Filas: **{data.shape[0]}**  \n- Columnas: **{data.shape[1]}**")
if 'label' in data.columns:
    st.sidebar.markdown(f"- Target detectado: **label** (valores: {data['label'].nunique()})")
else:
    st.sidebar.markdown("- **No se encontró la columna `label`**")

# Mostrar tabla y estadísticas
with st.expander("📊 Ver dataset (primeras filas)"):
    st.dataframe(data.head(50))

st.subheader("🔍 Estadísticas descriptivas")
st.write(data.describe(include='all').T)

# Comprobar columna target
if 'label' not in data.columns:
    st.error("El dataset debe contener la columna `label` con el tipo de cultivo. Añádela y vuelve a ejecutar.")
    st.stop()

# Seleccionar columnas numéricas para el modelo automático
features = [c for c in data.columns if c != 'label']
num_features = data[features].select_dtypes(include=[np.number]).columns.tolist()
non_num = [c for c in features if c not in num_features]
if non_num:
    st.warning(f"Columnas no numéricas (serán ignoradas para el modelo): {non_num}")

if len(num_features) == 0:
    st.error("No hay columnas numéricas para entrenar el modelo. Revisa el dataset.")
    st.stop()

# Preprocesado y pipeline
X = data[num_features].copy()
y = data['label'].copy()
le = LabelEncoder()
y_enc = le.fit_transform(y)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_features)
    ],
    remainder='drop'
)

model_pipeline = Pipeline([("preprocessor", preprocessor),
                           ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])

# Entrenar y evaluar
try:
    strat = y_enc if len(np.unique(y_enc))>1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=strat)
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
except Exception as e:
    st.error(f"Error al entrenar/evaluar el modelo: {e}")
    st.stop()

# Métricas
st.subheader("📈 Desempeño del modelo")
st.write(f"- Accuracy (test): **{acc:.3f}**")
st.text(classification_report(y_test, y_pred, target_names=le.classes_))

# Importancia de features
try:
    importances = model_pipeline.named_steps['clf'].feature_importances_
    fi = pd.DataFrame({"feature": num_features, "importance": importances}).sort_values("importance", ascending=False)
    st.subheader("🧾 Importancia de variables (top)")
    st.bar_chart(fi.set_index("feature")["importance"])
except Exception:
    st.info("No fue posible calcular importancias de features.")

# Exploración gráfica interactiva
st.subheader("🔎 Exploración por variable y cultivo")
col = st.selectbox("Selecciona variable", num_features)
fig, ax = plt.subplots(figsize=(8,4))
sns.boxplot(x="label", y=col, data=data, ax=ax)
ax.set_title(f"{col} por cultivo")
plt.xticks(rotation=45)
st.pyplot(fig)

# Panel de predicción interactiva
st.subheader("🌾 Recomendar cultivo según condiciones")
st.write("Ajusta las condiciones de suelo y clima. Los valores por defecto son la mediana del dataset.")

input_values = {}
cols = st.columns(3)
for i, feature in enumerate(num_features):
    minv = float(data[feature].min())
    maxv = float(data[feature].max())
    med = float(data[feature].median())
    with cols[i % 3]:
        if data[feature].dtype.kind in 'iu' and (med).is_integer():
            val = st.slider(feature, int(np.floor(minv)), int(np.ceil(maxv)), int(med))
        else:
            step = (maxv - minv) / 100 if (maxv - minv) != 0 else 0.1
            val = st.slider(feature, float(minv), float(maxv), float(med), step=step)
    input_values[feature] = val

if st.button("🔎 Recomendar cultivo"):
    input_df = pd.DataFrame([input_values], columns=num_features)
    pred_enc = model_pipeline.predict(input_df)
    pred_label = le.inverse_transform(pred_enc)[0]
    st.success(f"✅ Cultivo recomendado: **{pred_label}**")
    st.write("Valores ingresados:")
    st.table(input_df.T)

# Descargar modelo
if st.button("💾 Descargar modelo (.joblib)"):
    joblib.dump(model_pipeline, "modelo_recomendador.joblib")
    with open("modelo_recomendador.joblib", "rb") as f:
        st.download_button("Descargar modelo", data=f, file_name="modelo_recomendador.joblib", mime="application/octet-stream")
PY
