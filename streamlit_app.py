"""
============================================================
ML Predictor - Prediccion de Ventas Publicitarias en Tiempo Real
Regresion Lineal Multiple vs Arbol de Decision
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACION DE PAGINA
# ============================================================
st.set_page_config(
    page_title="Predictor de Ventas",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Predictor de Ventas Publicitarias")
st.markdown("Ajusta los sliders y obtén predicciones en tiempo real con **Regresión Lineal** y **Árbol de Decisión**.")

# ============================================================
# CARGA Y ENTRENAMIENTO DE MODELOS (cacheado)
# ============================================================
@st.cache_resource
def entrenar_modelos():
    """Carga datos y entrena ambos modelos una sola vez."""
    df = pd.read_csv("advertising.csv", index_col=0)
    
    X = df[['TV', 'Radio', 'Newspaper']].values
    y = df['Sales'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Regresion Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    
    # Arbol de Decision
    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    r2_dt = r2_score(y_test, y_pred_dt)
    
    return lr, dt, r2_lr, r2_dt

lr, dt, r2_lr, r2_dt = entrenar_modelos()

# ============================================================
# SLIDERS PARA PREDICCION
# ============================================================
st.subheader("🔧 Inversión Publicitaria ($)")

col1, col2, col3 = st.columns(3)
with col1:
    tv = st.slider("📺 TV", 0, 300, 150)
with col2:
    radio = st.slider("📻 Radio", 0, 50, 25)
with col3:
    newspaper = st.slider("📰 Newspaper", 0, 120, 30)

# ============================================================
# PREDICCION EN TIEMPO REAL
# ============================================================
X_pred = np.array([[tv, radio, newspaper]])
pred_lr = lr.predict(X_pred)[0]
pred_dt = dt.predict(X_pred)[0]

# ============================================================
# RESULTADOS
# ============================================================
st.markdown("---")
st.subheader("🎯 Resultados de la Predicción")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="background:#e8f4fd;padding:1.5rem;border-radius:10px;border-left:5px solid #1e3a5f;text-align:center;">
            <h3 style="color:#1e3a5f;margin:0;">📐 Regresión Lineal</h3>
            <p style="font-size:2rem;font-weight:bold;color:#1e3a5f;margin:0.5rem 0;">{pred_lr:.2f}</p>
            <p style="color:#666;">unidades de venta estimadas</p>
            <p style="font-size:0.8rem;color:#999;">R² = {r2_lr:.4f}</p>
        </div>
        """, unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="background:#e8fde8;padding:1.5rem;border-radius:10px;border-left:5px solid #2e7d32;text-align:center;">
            <h3 style="color:#2e7d32;margin:0;">🌳 Árbol de Decisión</h3>
            <p style="font-size:2rem;font-weight:bold;color:#2e7d32;margin:0.5rem 0;">{pred_dt:.2f}</p>
            <p style="color:#666;">unidades de venta estimadas</p>
            <p style="font-size:0.8rem;color:#999;">R² = {r2_dt:.4f}</p>
        </div>
        """, unsafe_allow_html=True
    )

# Ecuacion del modelo
st.markdown("---")
st.subheader("📊 Ecuación del Modelo de Regresión Lineal")
intercepto = lr.intercept_
coef_tv, coef_radio, coef_newspaper = lr.coef_
st.markdown(
    f"""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;font-family:monospace;font-size:1.1rem;text-align:center;">
        <b>Sales</b> = {intercepto:.4f} + ({coef_tv:.4f} × TV) + ({coef_radio:.4f} × Radio) + ({coef_newspaper:.4f} × Newspaper)
    </div>
    """, unsafe_allow_html=True
)

st.markdown("---")
st.markdown(
    f"""
    **Inversión ingresada:**
    - 📺 TV: ${tv}
    - 📻 Radio: ${radio}
    - 📰 Newspaper: ${newspaper}
    """
)