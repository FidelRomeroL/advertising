"""
============================================================
ML Predictor - Analisis de Campañas Publicitarias
Maestria en Economia - Inteligencia Artificial Aplicada
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================
# CONFIGURACION
# ============================================================
st.set_page_config(page_title="ML Predictor - Ventas", page_icon="📊", layout="wide")
st.title("📊 Análisis Predictivo de Ventas Publicitarias")
st.markdown("---")

# ============================================================
# CARGA Y ENTRENAMIENTO
# ============================================================
@st.cache_resource
def cargar_y_entrenar():
    df = pd.read_csv("advertising.csv")
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Regresion Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Arbol
    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    return df, X_train, X_test, y_train, y_test, lr, dt, y_pred_lr, y_pred_dt

df, X_train, X_test, y_train, y_test, lr, dt, y_pred_lr, y_pred_dt = cargar_y_entrenar()

# ============================================================
# SIDEBAR - NAVEGACION
# ============================================================
st.sidebar.title("📋 Navegación")
pagina = st.sidebar.radio("Ir a:", [
    "📊 Análisis Exploratorio",
    "📐 Regresión Lineal",
    "🌳 Árbol de Decisión",
    "⚖️ Comparación de Modelos",
    "🎯 Predictor Interactivo"
])

# ============================================================
# PAGINA 1: EDA
# ============================================================
if pagina == "📊 Análisis Exploratorio":
    st.subheader("📊 Análisis Exploratorio de Datos")
    st.markdown("Conozcamos la estructura y relaciones del dataset publicitario.")
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📦 Registros", df.shape[0])
    with col2: st.metric("📋 Variables", df.shape[1])
    with col3: st.metric("🎯 Variable Objetivo", "Sales (Ventas)")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Vista previa del dataset**")
        st.dataframe(df.head(10), width='stretch')
    
    with col2:
        st.markdown("**📈 Estadísticas descriptivas**")
        st.dataframe(df.describe().round(2), width='stretch')
    
    st.markdown("---")
    
    # Heatmap de correlación
    st.markdown("**🔥 Mapa de Calor - Correlaciones**")
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, cmap="RdYlBu", center=0, 
                square=True, linewidths=1, mask=mask, ax=ax, 
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title("Correlación entre Variables Publicitarias", fontsize=14, fontweight="bold")
    st.pyplot(fig)
    plt.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Distribución de Ventas**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['Sales'], bins=20, kde=True, color="#1e3a5f", ax=ax)
        ax.set_xlabel("Ventas (unidades)")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de la Variable Sales", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**📊 Relación: TV vs Sales**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.regplot(data=df, x='TV', y='Sales', scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax)
        ax.set_title("TV Publicidad vs Ventas", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Relación: Radio vs Sales**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.regplot(data=df, x='Radio', y='Sales', scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax)
        ax.set_title("Radio Publicidad vs Ventas", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close()
    with col2:
        st.markdown("**📊 Relación: Newspaper vs Sales**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.regplot(data=df, x='Newspaper', y='Sales', scatter_kws={'alpha':0.6}, line_kws={'color':'red'}, ax=ax)
        ax.set_title("Newspaper Publicidad vs Ventas", fontsize=12, fontweight="bold")
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.markdown("**📊 Boxplots para detección de valores atípicos**")
    col1, col2, col3, col4 = st.columns(4)
    vars_box = [('TV', '📺'), ('Radio', '📻'), ('Newspaper', '📰'), ('Sales', '🎯')]
    for idx, (var, emoji) in enumerate(vars_box):
        with [col1, col2, col3, col4][idx]:
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.boxplot(y=df[var], color="#3498db", ax=ax, width=0.4)
            ax.set_title(f"{emoji} {var}", fontweight="bold")
            st.pyplot(fig)
            plt.close()

# ============================================================
# PAGINA 2: REGRESION LINEAL
# ============================================================
elif pagina == "📐 Regresión Lineal":
    st.subheader("📐 Modelo de Regresión Lineal Múltiple")
    st.markdown("**Ecuación:** Relaciona las ventas con la inversión en TV, Radio y Newspaper.")
    
    intercepto = lr.intercept_
    coef_tv, coef_radio, coef_newspaper = lr.coef_
    
    st.markdown(
        f"""
        <div style="background:#e8f4fd;padding:1.5rem;border-radius:12px;text-align:center;border:2px solid #1e3a5f;margin:1rem 0;">
            <h3 style="color:#1e3a5f;">📐 Ecuación del Modelo</h3>
            <p style="font-size:1.3rem;font-family:monospace;font-weight:bold;">
                <b style="color:#1e3a5f;">Sales</b> = <span style="color:#e74c3c;">{intercepto:.4f}</span> 
                + (<span style="color:#2ecc71;">{coef_tv:.4f}</span> × TV) 
                + (<span style="color:#f39c12;">{coef_radio:.4f}</span> × Radio) 
                + (<span style="color:#9b59b6;">{coef_newspaper:.4f}</span> × Newspaper)
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📺 Coef. TV", f"{coef_tv:.4f}", delta="+", delta_color="normal")
        st.caption("Por cada $1 en TV, ventas aumentan {:.4f}".format(coef_tv))
    with col2:
        st.metric("📻 Coef. Radio", f"{coef_radio:.4f}", delta="+", delta_color="normal")
        st.caption("Por cada $1 en Radio, ventas aumentan {:.4f}".format(coef_radio))
    with col3:
        st.metric("📰 Coef. Newspaper", f"{coef_newspaper:.4f}")
        st.caption("Por cada $1 en Newspaper, ventas aumentan {:.4f}".format(coef_newspaper))
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    with col1: st.metric("🎯 R² (Precisión)", f"{r2_lr:.4f}", help="Qué tan bien el modelo explica las ventas. 1 = perfecto.")
    with col2: st.metric("📏 MAE", f"{mae_lr:.4f}", help="Error absoluto medio. Menor = mejor.")
    with col3: st.metric("📐 RMSE", f"{rmse_lr:.4f}", help="Raíz del error cuadrático medio.")
    
    st.markdown("---")
    
    # Gráfico de coeficientes
    st.markdown("**📊 Importancia de las Variables (Coeficientes)**")
    fig, ax = plt.subplots(figsize=(8, 4))
    coefs = [coef_tv, coef_radio, coef_newspaper]
    labels = ['TV', 'Radio', 'Newspaper']
    colors_bar = ['#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.barh(labels, coefs, color=colors_bar, edgecolor='black', height=0.6)
    for bar, val in zip(bars, coefs):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontweight='bold')
    ax.set_xlabel("Coeficiente")
    ax.set_title("Impacto de cada medio publicitario en las Ventas", fontweight="bold")
    ax.axvline(x=0, color='black', linewidth=0.5)
    st.pyplot(fig)
    plt.close()
    
    # Reales vs Predichos
    st.markdown("**📈 Valores Reales vs Predichos**")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred_lr, alpha=0.6, color='#1e3a5f', s=60, edgecolors='white', linewidth=0.5)
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    ax.set_xlabel("Valores Reales", fontsize=11)
    ax.set_ylabel("Valores Predichos", fontsize=11)
    ax.set_title(f"Regresión Lineal - Reales vs Predichos (R² = {r2_lr:.4f})", fontweight="bold")
    ax.legend()
    st.pyplot(fig)
    plt.close()
    
    with st.expander("📖 Interpretación del modelo"):
        st.markdown(f"""
        - **R² = {r2_lr:.4f}**: El modelo explica el **{r2_lr*100:.2f}%** de la variabilidad en las ventas.
        - **TV** es el medio con mayor impacto ({coef_tv:.4f} unidades de venta por cada $1 invertido).
        - **Radio** tiene un impacto de {coef_radio:.4f} unidades por $1.
        - **Newspaper** tiene el menor impacto ({coef_newspaper:.4f}).
        - **MAE = {mae_lr:.4f}**: En promedio, el modelo se equivoca por {mae_lr:.2f} unidades de venta.
        """)

# ============================================================
# PAGINA 3: ARBOL DE DECISION
# ============================================================
elif pagina == "🌳 Árbol de Decisión":
    st.subheader("🌳 Árbol de Decisión para Regresión")
    st.markdown("Modelo no lineal que segmenta los datos en grupos para hacer predicciones.")
    
    col1, col2, col3 = st.columns(3)
    r2_dt = r2_score(y_test, y_pred_dt)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    
    with col1: st.metric("🎯 R² (Precisión)", f"{r2_dt:.4f}", help="Qué tan bien el modelo explica las ventas.")
    with col2: st.metric("📏 MAE", f"{mae_dt:.4f}", help="Error absoluto medio.")
    with col3: st.metric("📐 RMSE", f"{rmse_dt:.4f}", help="Raíz del error cuadrático medio.")
    
    st.markdown("---")
    
    st.markdown("**⚙️ Hiperparámetros del Árbol**")
    col1, col2, col3 = st.columns(3)
    with col1: st.info(f"**Profundidad máxima:** {dt.get_params()['max_depth']}")
    with col2: st.info(f"**Muestras mínimas por división:** {dt.get_params()['min_samples_split']}")
    with col3: st.info(f"**Hojas:** {dt.get_n_leaves()}")
    
    st.markdown("---")
    
    # Importancia de variables
    st.markdown("**⭐ Importancia de las Variables**")
    importancia = dt.feature_importances_
    feat_names = ['TV', 'Radio', 'Newspaper']
    
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_imp = ['#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.barh(feat_names, importancia, color=colors_imp, edgecolor='black', height=0.6)
    for bar, val in zip(bars, importancia):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontweight='bold')
    ax.set_xlabel("Importancia Relativa")
    ax.set_title("Importancia de cada medio (Árbol de Decisión)", fontweight="bold")
    ax.set_xlim(0, 1.1)
    st.pyplot(fig)
    plt.close()
    
    st.markdown(f"""
    **Análisis:** 
    - **TV** aporta el **{importancia[0]*100:.1f}%** de la importancia predictiva.
    - **Radio** aporta el **{importancia[1]*100:.1f}%**.
    - **Newspaper** aporta el **{importancia[2]*100:.1f}%**.
    """)
    
    # Reales vs Predichos
    st.markdown("---")
    st.markdown("**📈 Valores Reales vs Predichos**")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred_dt, alpha=0.6, color='#2e7d32', s=60, edgecolors='white', linewidth=0.5)
    min_val = min(y_test.min(), y_pred_dt.min())
    max_val = max(y_test.max(), y_pred_dt.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    ax.set_xlabel("Valores Reales", fontsize=11)
    ax.set_ylabel("Valores Predichos", fontsize=11)
    ax.set_title(f"Árbol de Decisión - Reales vs Predichos (R² = {r2_dt:.4f})", fontweight="bold")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# ============================================================
# PAGINA 4: COMPARACION
# ============================================================
elif pagina == "⚖️ Comparación de Modelos":
    st.subheader("⚖️ Comparación: Regresión Lineal vs Árbol de Decisión")
    st.markdown("Análisis comparativo del rendimiento de ambos modelos.")
    
    r2_lr = r2_score(y_test, y_pred_lr)
    r2_dt = r2_score(y_test, y_pred_dt)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    
    # Tabla comparativa
    comparacion = pd.DataFrame({
        "Métrica": ["R² (Precisión)", "MAE (Error absoluto)", "RMSE (Error cuadrático)"],
        "📐 Regresión Lineal": [f"{r2_lr:.4f}", f"{mae_lr:.4f}", f"{rmse_lr:.4f}"],
        "🌳 Árbol de Decisión": [f"{r2_dt:.4f}", f"{mae_dt:.4f}", f"{rmse_dt:.4f}"],
        "🏆 Mejor Modelo": [
            "📐 Regresión Lineal" if r2_lr >= r2_dt else "🌳 Árbol",
            "🌳 Árbol" if mae_dt <= mae_lr else "📐 Regresión Lineal",
            "🌳 Árbol" if rmse_dt <= rmse_lr else "📐 Regresión Lineal"
        ]
    })
    
    st.dataframe(comparacion, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Grafico comparativo de barras
    st.markdown("**📊 Comparación Visual de Métricas**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(2)
        width = 0.35
        ax.bar(x[0], r2_lr, width, label='Regresión Lineal', color='#1e3a5f', edgecolor='black')
        ax.bar(x[0]+width, r2_dt, width, label='Árbol Decisión', color='#2e7d32', edgecolor='black')
        ax.set_xticks([x[0]+width/2])
        ax.set_xticklabels(['R²'])
        ax.set_ylabel("R² (mayor = mejor)")
        ax.set_title("Comparación de Precisión (R²)", fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 1)
        for i, v in enumerate([r2_lr, r2_dt]):
            ax.text(i*1.35 + 0.1, v + 0.01, f'{v:.4f}', fontweight='bold', ha='center')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x[0], mae_lr, width, label='Regresión Lineal', color='#1e3a5f', edgecolor='black')
        ax.bar(x[0]+width, mae_dt, width, label='Árbol Decisión', color='#2e7d32', edgecolor='black')
        ax.set_xticks([x[0]+width/2])
        ax.set_xticklabels(['MAE'])
        ax.set_ylabel("MAE (menor = mejor)")
        ax.set_title("Comparación de Error (MAE)", fontweight="bold")
        ax.legend()
        for i, v in enumerate([mae_lr, mae_dt]):
            ax.text(i*1.35 + 0.1, v + 0.01, f'{v:.4f}', fontweight='bold', ha='center')
        st.pyplot(fig)
        plt.close()
    
    # Reales vs Predichos - ambos modelos
    st.markdown("---")
    st.markdown("**📈 Reales vs Predichos - Ambos Modelos**")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred_lr, alpha=0.6, color='#1e3a5f', s=50, label='Regresión Lineal', edgecolors='white')
    ax.scatter(y_test, y_pred_dt, alpha=0.6, color='#2e7d32', s=50, label='Árbol Decisión', marker='s', edgecolors='white')
    min_val = min(y_test.min(), y_pred_lr.min(), y_pred_dt.min())
    max_val = max(y_test.max(), y_pred_lr.max(), y_pred_dt.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    ax.set_xlabel("Valores Reales", fontsize=12)
    ax.set_ylabel("Valores Predichos", fontsize=12)
    ax.set_title("Comparación: Reales vs Predichos", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    mejor_modelo = "📐 Regresión Lineal" if r2_lr >= r2_dt else "🌳 Árbol de Decisión"
    st.success(f"### 🏆 **Conclusión: El mejor modelo es {mejor_modelo}**")
    if r2_lr >= r2_dt:
        st.markdown(f"""
        - La **Regresión Lineal** tiene mejor R² ({r2_lr:.4f} vs {r2_dt:.4f})
        - Esto significa que las ventas tienen una relación **lineal** con la inversión publicitaria
        - Recomendación: Usar **Regresión Lineal** para predicciones de ventas
        """)
    else:
        st.markdown(f"""
        - El **Árbol de Decisión** tiene mejor R² ({r2_dt:.4f} vs {r2_lr:.4f})
        - Esto sugiere relaciones **no lineales** entre la inversión y las ventas
        - Recomendación: Usar **Árbol de Decisión** para predicciones más precisas
        """)

# ============================================================
# PAGINA 5: PREDICTOR
# ============================================================
elif pagina == "🎯 Predictor Interactivo":
    st.subheader("🎯 Simulador de Ventas en Tiempo Real")
    st.markdown("Ajusta la inversión en cada medio y observa cómo cambian las predicciones.")
    
    st.markdown("### 🔧 Inversión Publicitaria")
    col1, col2, col3 = st.columns(3)
    with col1:
        tv = st.slider("📺 **TV ($)**", 0, 300, 150, help="Inversión en publicidad de TV (0-300)")
    with col2:
        radio = st.slider("📻 **Radio ($)**", 0, 50, 25, help="Inversión en publicidad de Radio (0-50)")
    with col3:
        newspaper = st.slider("📰 **Newspaper ($)**", 0, 120, 30, help="Inversión en publicidad en periódicos (0-120)")
    
    X_pred = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    pred_lr = lr.predict(X_pred)[0]
    pred_dt = dt.predict(X_pred)[0]
    
    st.markdown("---")
    st.markdown("### 🎯 Resultados de la Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mejor = "Regresión Lineal" if r2_score(y_test, y_pred_lr) >= r2_score(y_test, y_pred_dt) else "Árbol"
        st.markdown(
            f"""
            <div style="background:#e8f4fd;padding:1.5rem;border-radius:12px;border-left:6px solid #1e3a5f;text-align:center;box-shadow:2px 2px 10px rgba(0,0,0,0.1);">
                <h3 style="color:#1e3a5f;margin:0;">📐 Regresión Lineal</h3>
                <p style="font-size:2.5rem;font-weight:bold;color:#1e3a5f;margin:0.5rem 0;">{pred_lr:.2f}</p>
                <p style="color:#555;font-size:1.1rem;">unidades de venta estimadas</p>
                <div style="background:rgba(30,58,95,0.1);padding:0.5rem;border-radius:8px;margin-top:0.5rem;">
                    <span style="font-size:0.9rem;">📊 R² = {r2_score(y_test, y_pred_lr):.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="background:#e8fde8;padding:1.5rem;border-radius:12px;border-left:6px solid #2e7d32;text-align:center;box-shadow:2px 2px 10px rgba(0,0,0,0.1);">
                <h3 style="color:#2e7d32;margin:0;">🌳 Árbol de Decisión</h3>
                <p style="font-size:2.5rem;font-weight:bold;color:#2e7d32;margin:0.5rem 0;">{pred_dt:.2f}</p>
                <p style="color:#555;font-size:1.1rem;">unidades de venta estimadas</p>
                <div style="background:rgba(46,125,50,0.1);padding:0.5rem;border-radius:8px;margin-top:0.5rem;">
                    <span style="font-size:0.9rem;">📊 R² = {r2_score(y_test, y_pred_dt):.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True
        )
    
    # Diferencia entre modelos
    diferencia = abs(pred_lr - pred_dt)
    st.markdown(f"""
    <div style="background:#f8f9fa;padding:1rem;border-radius:10px;text-align:center;margin:1rem 0;">
        <span style="font-size:1.1rem;">💡 <b>Diferencia entre modelos:</b> {diferencia:.2f} unidades</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Ecuación del Modelo")
    intercepto = lr.intercept_
    coef_tv, coef_radio, coef_newspaper = lr.coef_
    st.markdown(
        f"""
        <div style="background:#f8f9fa;padding:1.2rem;border-radius:10px;font-family:monospace;font-size:1.1rem;text-align:center;border:1px solid #ddd;">
            <b style="color:#1e3a5f;">Sales</b> = {intercepto:.4f} + 
            <span style="color:#2ecc71;">({coef_tv:.4f} × TV)</span> + 
            <span style="color:#f39c12;">({coef_radio:.4f} × Radio)</span> + 
            <span style="color:#9b59b6;">({coef_newspaper:.4f} × Newspaper)</span>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Resumen de inversión
    st.markdown("---")
    st.markdown("### 📋 Resumen de la Simulación")
    inversion_total = tv + radio + newspaper
    st.markdown(
        f"""
        <div style="background:#fff;padding:1.2rem;border-radius:10px;border:1px solid #ddd;">
            <table style="width:100%;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #ddd;">
                    <th style="text-align:left;padding:0.5rem;">Medio</th>
                    <th style="text-align:right;padding:0.5rem;">Inversión</th>
                    <th style="text-align:right;padding:0.5rem;">% del Total</th>
                </tr>
                <tr>
                    <td style="padding:0.3rem 0.5rem;">📺 TV</td>
                    <td style="text-align:right;padding:0.3rem 0.5rem;">${tv}</td>
                    <td style="text-align:right;padding:0.3rem 0.5rem;">{tv/inversion_total*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="padding:0.3rem 0.5rem;">📻 Radio</td>
                    <td style="text-align:right;padding:0.3rem 0.5rem;">${radio}</td>
                    <td style="text-align:right;padding:0.3rem 0.5rem;">{radio/inversion_total*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="padding:0.3rem 0.5rem;">📰 Newspaper</td>
                    <td style="text-align:right;padding:0.3rem 0.5rem;">${newspaper}</td>
                    <td style="text-align:right;padding:0.3rem 0.5rem;">{newspaper/inversion_total*100:.1f}%</td>
                </tr>
                <tr style="border-top:2px solid #1e3a5f;font-weight:bold;">
                    <td style="padding:0.5rem;">💰 Total Inversión</td>
                    <td style="text-align:right;padding:0.5rem;">${inversion_total}</td>
                    <td style="text-align:right;padding:0.5rem;">100%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("### 💡 Recomendación")
    if tv > 200:
        st.success("✅ **Alta inversión en TV**: Históricamente la TV tiene el mayor retorno por dólar invertido. Es una buena estrategia.")
    elif tv < 50:
        st.warning("⚠️ **Baja inversión en TV**: Considera aumentar la inversión en TV, ya que tiene el mayor impacto en ventas según los datos.")
    
    if radio > 40:
        st.success("✅ **Buena inversión en Radio**: La radio complementa bien a la TV para llegar a diferentes audiencias.")
    
    if newspaper > 80 and tv < 100:
        st.warning("⚠️ **Newspaper con baja TV**: Según los datos, la TV tiene mucho más impacto que el periódico. Considera reasignar presupuesto.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.9rem;'>"
    "Desarrollado para Machine Learning para Negocios | Maestría en Economía - IA Aplicada"
    "</p>", unsafe_allow_html=True
)