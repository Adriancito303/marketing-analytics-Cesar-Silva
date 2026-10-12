"""
Laboratorio Final — Starbucks Rewards
PUNTO DE PARTIDA para su propia app de Streamlit. Complete los `# TODO`.

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus
resultados del laboratorio — el Qini de su T-learner y la tabla de políticas —
con al menos un control interactivo.

Correr con:  streamlit run streamlit_starter.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]


@st.cache_data
def load_data():
    # Buscar el archivo CSV desde distintas ubicaciones del repositorio
    posibles_rutas = [
        Path(__file__).parent / "data" / "uplift_campaign.csv",
        Path(__file__).parent.parent / "data" / "uplift_campaign.csv",
        Path("data/uplift_campaign.csv"),
        Path("../data/uplift_campaign.csv"),
    ]

    for ruta in posibles_rutas:
        if ruta.exists():
            return pd.read_csv(ruta)

    st.error(
        "No se encontró data/uplift_campaign.csv. "
        "Verifica que el archivo esté incluido en el repositorio de GitHub."
    )
    st.stop()


@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed,
        stratify=df["treatment_group"]
    )

    # TODO: entrenen el T-learner (dos modelos, igual que en el notebook)
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]
    modelo_tratado = LogisticRegression(max_iter=1000).fit(
        train_t[FEATURES], train_t["responded_60d"]
    )
    modelo_control = LogisticRegression(max_iter=1000).fit(
        train_c[FEATURES], train_c["responded_60d"]
    )

    test = test.copy()
    X_test = test[FEATURES]
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test


def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]

    # Comparar la utilidad neta de los tratados con el margen del grupo control
    valor_tratado = sel.loc[
        sel["treatment_group"] == "treatment", "utilidad_neta_60d"
    ].mean()
    valor_control = sel.loc[
        sel["treatment_group"] == "control", "margin_60d"
    ].mean()

    # Evitar resultados inválidos si alguno de los grupos está vacío
    if pd.isna(valor_tratado) or pd.isna(valor_control):
        return 0.0, len(sel)

    return valor_tratado - valor_control, len(sel)


# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards")
st.caption(
    "Explora qué clientes podrían responder mejor a una campaña y estima "
    "el valor de contactar a los clientes con mayor uplift."
)

df = load_data()

# Validar las columnas necesarias antes de entrenar
columnas_necesarias = FEATURES + [
    "treatment_group",
    "responded_60d",
    "utilidad_neta_60d",
    "margin_60d"
]
faltantes = [col for col in columnas_necesarias if col not in df.columns]

if faltantes:
    st.error(f"Faltan estas columnas en el CSV: {', '.join(faltantes)}")
    st.stop()

# --- Parámetros ---
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider(
        "Porcentaje de clientes a contactar",
        min_value=5,
        max_value=100,
        value=30,
        step=5
    )

train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

# --- TODO: calculen y grafiquen la curva Qini de su T-learner ---
test_ordenado = test.sort_values(
    "uplift_estimado", ascending=False
).reset_index(drop=True)

test_ordenado["tratamiento_real"] = (
    test_ordenado["treatment_group"] == "treatment"
).astype(int)

test_ordenado["respuesta"] = test_ordenado["responded_60d"]

n = len(test_ordenado)
fracciones = np.arange(1, n + 1) / n

n_tratados = test_ordenado["tratamiento_real"].sum()
n_controles = n - n_tratados

tratados_acum = (
    test_ordenado["respuesta"] * test_ordenado["tratamiento_real"]
).cumsum()

controles_acum = (
    test_ordenado["respuesta"] * (1 - test_ordenado["tratamiento_real"])
).cumsum()

factor_control = n_tratados / n_controles if n_controles else 0

qini_modelo = tratados_acum - controles_acum * factor_control

respuesta_tratados = (
    test_ordenado["respuesta"] * test_ordenado["tratamiento_real"]
).sum()

respuesta_controles = (
    test_ordenado["respuesta"] * (1 - test_ordenado["tratamiento_real"])
).sum()

qini_aleatorio = (
    fracciones * respuesta_tratados
    - fracciones * respuesta_controles * factor_control
)

st.subheader("Curva Qini")
st.write(
    "Compara la capacidad del modelo para priorizar clientes que podrían "
    "responder a la campaña."
)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fracciones, qini_modelo, label="T-learner", color=GREEN)
ax.plot(
    fracciones, qini_aleatorio,
    linestyle="--", label="Referencia aleatoria", color=GRAY
)
ax.set_xlabel("Proporción acumulada de clientes contactados")
ax.set_ylabel("Qini acumulado")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

# --- TODO: muestren la tabla de políticas (valor por cliente y total) para el % elegido ---
st.subheader("Tabla de políticas")

cantidad = max(1, int(len(test_ordenado) * pct_contactar / 100))
seleccionados = test_ordenado.head(cantidad).index
seleccionados_mask = test_ordenado.index.isin(seleccionados)

valor_total, cantidad_seleccionada = valor_de_la_politica(
    test_ordenado, seleccionados_mask
)

valor_por_cliente = (
    valor_total / cantidad_seleccionada if cantidad_seleccionada else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Clientes seleccionados", f"{cantidad_seleccionada:,}")
col2.metric("Valor estimado por cliente", f"{valor_por_cliente:.3f}")
col3.metric("Diferencia media estimada", f"{valor_total:.3f}")

tabla_politica = pd.DataFrame({
    "Política": [
        "Contactar a todos",
        f"Contactar al {pct_contactar}% con mayor uplift"
    ],
    "Clientes": [
        len(test_ordenado),
        cantidad_seleccionada
    ],
    "Uplift medio estimado": [
        test_ordenado["uplift_estimado"].mean(),
        test_ordenado.head(cantidad)["uplift_estimado"].mean()
    ]
})

st.dataframe(tabla_politica, use_container_width=True)

st.caption(
    "Nota: el uplift estimado representa la diferencia de probabilidades "
    "predichas entre tratamiento y control; no garantiza el resultado real."
)

st.info(
    "Los resultados dependen de la calidad de los datos y del desempeño "
    "del modelo. Valida las métricas antes de tomar decisiones de negocio."
)