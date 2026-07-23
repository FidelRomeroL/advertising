"""
====================================================================
ML Predictor - App Web con Streamlit
Analisis predictivo: Regresion Lineal y Arboles de Decision
====================================================================
Maestria en Economia - Mencion en Inteligencia Artificial Aplicada
Machine Learning para Negocios
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import warnings
import base64
from io import BytesIO, StringIO

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACION DE LA PAGINA
# ============================================================
st.set_page_config(
    page_title="ML Predictor - Analisis Publicitario",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ESTILO CSS PERSONALIZADO
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #c8d6e5;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .card h3 {
        color: #1e3a5f;
        margin-top: 0;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid #2d6a9f;
    }
    .metric-card h3 {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1e3a5f;
        margin: 0.3rem 0;
    }
    .insight-box {
        background: #e8f4fd;
        border-left: 4px solid #2d6a9f;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        padding: 2rem 0 0 0;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1e3a5f, #2d6a9f);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2d6a9f, #1e3a5f);
        color: white;
    }
    div[data-testid="stSidebar"] {
        background: #f0f2f6;
        border-right: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

@st.cache_data
def cargar_datos(archivo):
    """Carga datos desde archivo CSV subido o usa el predeterminado."""
    if archivo is not None:
        df = pd.read_csv(archivo)
    else:
        # Usar datos precargados de advertising
        df = pd.read_csv("advertising.csv", index_col=0)
    return df

def ejecutar_regresion_lineal(X_train, X_test, y_train, y_test):
    """Entrena y evalua modelo de Regresion Lineal."""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return lr, y_pred, r2, mae, rmse

def ejecutar_arbol_decision(X_train, X_test, y_train, y_test, max_depth=5, min_samples_split=5):
    """Entrena y evalua modelo de Arbol de Decision."""
    dt = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return dt, y_pred, r2, mae, rmse

def crear_grafico_scatter(df, x_col='TV', y_col='Sales'):
    """Crea grafico de dispersion con linea de tendencia."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[x_col], df[y_col], alpha=0.6, edgecolors='k', linewidth=0.5, s=60)
    z = np.polyfit(df[x_col], df[y_col], 1)
    p = np.poly1d(z)
    ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8, linewidth=2)
    corr = df[x_col].corr(df[y_col])
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f'{x_col} vs {y_col} (r = {corr:.3f})', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def crear_grafico_real_vs_predicho(y_test, y_pred_lr, y_pred_dt, r2_lr, r2_dt):
    """Crea grafico de valores reales vs predichos para ambos modelos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Regresion Lineal
    axes[0].scatter(y_test, y_pred_lr, alpha=0.7, edgecolors='k', linewidth=0.5, color='steelblue', s=60)
    min_val = min(y_test.min(), y_pred_lr.min(), y_pred_dt.min())
    max_val = max(y_test.max(), y_pred_lr.max(), y_pred_dt.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0].set_xlabel('Valores Reales', fontsize=11)
    axes[0].set_ylabel('Valores Predichos', fontsize=11)
    axes[0].set_title(f'Regresion Lineal (R² = {r2_lr:.4f})', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Arbol de Decision
    axes[1].scatter(y_test, y_pred_dt, alpha=0.7, edgecolors='k', linewidth=0.5, color='seagreen', s=60)
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1].set_xlabel('Valores Reales', fontsize=11)
    axes[1].set_ylabel('Valores Predichos', fontsize=11)
    axes[1].set_title(f'Arbol de Decision (R² = {r2_dt:.4f})', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Comparacion: Valores Reales vs. Predichos', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def crear_grafico_comparacion_metricas(r2_lr, mae_lr, rmse_lr, r2_dt, mae_dt, rmse_dt):
    """Crea grafico de barras comparativo de metricas."""
    metricas = ['R²', 'MAE', 'RMSE']
    valores_lr = [r2_lr, mae_lr, rmse_lr]
    valores_dt = [r2_dt, mae_dt, rmse_dt]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metricas))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, valores_lr, width, label='Regresion Lineal', 
                   color='steelblue', edgecolor='k', alpha=0.9)
    bars2 = ax.bar(x + width/2, valores_dt, width, label='Arbol de Decision', 
                   color='seagreen', edgecolor='k', alpha=0.9)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Valor', fontsize=11)
    ax.set_title('Comparacion de Metricas: Regresion Lineal vs Arbol de Decision', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def crear_heatmap_correlacion(df):
    """Crea matriz de correlacion."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5,
                square=True, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Matriz de Correlacion', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig, corr

def descargar_dataframe(df, filename="datos_exportados.csv"):
    """Genera link de descarga para un DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Descargar CSV</a>'
    return href

# ============================================================
# INICIO DE LA APP - HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>📊 ML Predictor - Analisis de Campañas Publicitarias</h1>
    <p>Regresion Lineal Multiple vs Árbol de Decision | Machine Learning para Negocios</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR - CONTROLES
# ============================================================

st.sidebar.image("https://img.icons8.com/fluency/96/000000/bar-chart.png", width=80)
st.sidebar.title("⚙️ Controles")

# Carga de datos
st.sidebar.header("📁 Carga de Datos")
archivo_subido = st.sidebar.file_uploader(
    "Sube tu archivo CSV (opcional)",
    type=['csv'],
    help="Si no cargas archivo, se usara el dataset advertising.csv precargado"
)

usar_datos_ejemplo = st.sidebar.checkbox(
    "Usar datos de ejemplo (advertising.csv)", 
    value=True if archivo_subido is None else False
)

# Parametros del modelo
st.sidebar.header("🔧 Parametros del Modelo")
test_size = st.sidebar.slider(
    "Tamaño del conjunto de prueba (%)", 
    min_value=10, max_value=40, value=20, step=5,
    help="Proporcion de datos reservados para evaluacion"
)

max_depth = st.sidebar.slider(
    "Profundidad máxima (Arbol)", 
    min_value=2, max_value=15, value=5, step=1,
    help="Controla la complejidad del arbol. Valores mayores pueden causar sobreajuste"
)

min_samples_split = st.sidebar.slider(
    "Min. muestras por division", 
    min_value=2, max_value=20, value=5, step=1,
    help="Minimo de muestras requeridas para dividir un nodo"
)

random_state = st.sidebar.number_input("Semilla aleatoria", value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align:center; color:#666; font-size:0.8rem;">
    <b>Desarrollado para:</b><br>
    Maestria en Economia<br>
    Mencion en IA Aplicada<br>
    Machine Learning para Negocios
</div>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE DATOS
# ============================================================

with st.spinner('Cargando datos...'):
    if archivo_subido is not None:
        df = cargar_datos(archivo_subido)
        st.sidebar.success(f"✅ Archivo cargado: {archivo_subido.name}")
    elif usar_datos_ejemplo:
        df = cargar_datos(None)
        st.sidebar.info("📌 Usando dataset: advertising.csv")
    else:
        st.warning("⚠️ Debes cargar un archivo o seleccionar 'Usar datos de ejemplo'")
        st.stop()

# Verificar columnas
columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

if len(columnas_numericas) < 2:
    st.error(f"❌ Se requieren al menos 2 columnas numericas. Solo se encontraron: {len(columnas_numericas)}")
    st.stop()

# Seleccion de variable objetivo y predictoras
st.sidebar.header("🎯 Variables")
target_col = st.sidebar.selectbox(
    "Variable objetivo (Y)", 
    columnas_numericas,
    index=columnas_numericas.index('Sales') if 'Sales' in columnas_numericas else len(columnas_numericas)-1
)

feature_cols_default = [c for c in columnas_numericas if c != target_col]
feature_cols = st.sidebar.multiselect(
    "Variables predictoras (X)", 
    columnas_numericas,
    default=feature_cols_default,
    help="Selecciona las variables para predecir la variable objetivo"
)

if len(feature_cols) == 0:
    st.warning("⚠️ Selecciona al menos una variable predictora")
    st.stop()

# ============================================================
# TABS PRINCIPALES
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Datos", 
    "📈 EDA", 
    "📐 Regresion Lineal", 
    "🌳 Arbol de Decision", 
    "⚖️ Comparacion",
    "🔮 Predicciones"
])

# ============================================================
# TAB 1: DATOS
# ============================================================

with tab1:
    st.header("📋 Vista de Datos")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Observaciones", df.shape[0])
    col2.metric("Variables Numericas", len(columnas_numericas))
    col3.metric("Variable Objetivo", target_col)
    col4.metric("Predictores", len(feature_cols))
    
    st.markdown("---")
    
    # Mostrar datos
    st.subheader("Datos completos")
    st.dataframe(df, use_container_width=True)
    
    # Estadisticas descriptivas
    st.subheader("Estadisticas Descriptivas")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Valores nulos
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        st.success("✅ No se encontraron valores nulos en el dataset")
    else:
        st.warning(f"⚠️ Se encontraron {nulls.sum()} valores nulos")
        st.dataframe(nulls[nulls > 0].to_frame("Valores Nulos"), use_container_width=True)
    
    st.markdown(descargar_dataframe(df), unsafe_allow_html=True)

# ============================================================
# TAB 2: EDA
# ============================================================

with tab2:
    st.header("📈 Analisis Exploratorio de Datos")
    
    # Heatmap de correlacion
    st.subheader("Matriz de Correlacion")
    fig_heat, corr_matrix = crear_heatmap_correlacion(df)
    st.pyplot(fig_heat)
    
    # Interpretacion de correlaciones
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**📌 Interpretacion de Correlaciones:**")
    if target_col in corr_matrix.columns:
        correlaciones_target = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        for var, corr_val in correlaciones_target.items():
            intensidad = "fuerte" if abs(corr_val) > 0.7 else "moderada" if abs(corr_val) > 0.4 else "debíl"
            st.write(f"- **{var}**: r = {corr_val:.3f} (relacion {intensidad})")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Scatter plots
    st.subheader("Relacion entre Variables")
    cols_scatter = st.columns(min(3, len(feature_cols)))
    for i, col in enumerate(feature_cols[:3]):
        with cols_scatter[i]:
            fig = crear_grafico_scatter(df, col, target_col)
            st.pyplot(fig)
    
    if len(feature_cols) > 3:
        st.info(f"💡 Se muestran las primeras 3 variables predictoras de {len(feature_cols)} seleccionadas")
    
    # Boxplots
    st.markdown("---")
    st.subheader("Distribucion de Variables (Boxplots)")
    fig_box, axes_box = plt.subplots(1, len(columnas_numericas), figsize=(min(18, 4*len(columnas_numericas)), 5))
    if len(columnas_numericas) == 1:
        axes_box = [axes_box]
    for i, col in enumerate(columnas_numericas):
        axes_box[i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axes_box[i].set_title(col, fontsize=10, fontweight='bold')
        axes_box[i].grid(axis='y', alpha=0.3)
    plt.suptitle("Distribucion de Variables", y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_box)
    
    # Deteccion de outliers
    st.markdown("---")
    st.subheader("Deteccion de Valores Atipicos (IQR)")
    for col in columnas_numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lim_inf) | (df[col] > lim_sup)]
        if len(outliers) > 0:
            st.write(f"- **{col}**: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**💡 Insight:** El EDA permite identificar patrones iniciales, relaciones lineales entre variables, y posibles problemas como outliers o multicolinealidad antes de construir los modelos predictivos.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 3: REGRESION LINEAL
# ============================================================

with tab3:
    st.header("📐 Modelo de Regresion Lineal Multiple")
    
    # Preparacion de datos
    X = df[feature_cols].values
    y = df[target_col].values
    
    test_size_pct = test_size / 100
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_pct, random_state=random_state
    )
    
    with st.spinner('Entrenando modelo de Regresion Lineal...'):
        lr, y_pred_lr, r2_lr, mae_lr, rmse_lr = ejecutar_regresion_lineal(
            X_train, X_test, y_train, y_test
        )
    
    # Metricas
    st.subheader("Metricas de Desempeno")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""
    <div class="metric-card">
        <h3>R² (Coef. Determinacion)</h3>
        <div class="value">{r2_lr:.4f}</div>
        <p style="font-size:0.8rem; color:#666;">{r2_lr*100:.1f}% de varianza explicada</p>
    </div>
    """, unsafe_allow_html=True)
    col2.markdown(f"""
    <div class="metric-card">
        <h3>MAE</h3>
        <div class="value">{mae_lr:.4f}</div>
        <p style="font-size:0.8rem; color:#666;">Error absoluto medio</p>
    </div>
    """, unsafe_allow_html=True)
    col3.markdown(f"""
    <div class="metric-card">
        <h3>RMSE</h3>
        <div class="value">{rmse_lr:.4f}</div>
        <p style="font-size:0.8rem; color:#666;">Raiz del error cuadratico medio</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Ecuacion y coeficientes
    st.subheader("Ecuacion del Modelo")
    
    ecuacion = f"{target_col} = {lr.intercept_:.4f}"
    for i, col in enumerate(feature_cols):
        signo = "+" if lr.coef_[i] >= 0 else "-"
        ecuacion += f" {signo} ({abs(lr.coef_[i]):.4f} × {col})"
    
    st.markdown(f"""
    <div class="card">
        <h3>Ecuacion de Regresion</h3>
        <p style="font-size:1.1rem; font-family: monospace;">
            <b>{ecuacion}</b>
        </p>
        <p style="margin-top:0.5rem;"><b>Intercepto (β₀):</b> {lr.intercept_:.4f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabla de coeficientes
    st.subheader("Coeficientes por Variable")
    coef_data = {
        'Variable': feature_cols,
        'Coeficiente': [f"{c:.4f}" for c in lr.coef_],
        'Interpretacion (por $1,000)': [
            f"Por cada unidad adicional en {feature_cols[i]}, {target_col} cambia en {lr.coef_[i]:.4f} unidades"
            for i in range(len(feature_cols))
        ]
    }
    st.dataframe(pd.DataFrame(coef_data), use_container_width=True, hide_index=True)
    
    # Interpretacion de negocio
    st.markdown("---")
    st.subheader("Interpretacion en Contexto de Negocio")
    
    coef_df = pd.DataFrame({'Variable': feature_cols, 'Coeficiente': lr.coef_}).sort_values('Coeficiente', ascending=False)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**📊 Jerarquia de Impacto (por coefiente):**")
    for i, row in coef_df.iterrows():
        icono = "🥇" if i == coef_df.index[0] else "🥈" if i == coef_df.index[1] else "🥉"
        st.write(f"{icono} **{row['Variable']}**: Coef = {row['Coeficiente']:.4f}")
    
    st.markdown("---")
    st.markdown(f"""
    **💡 Interpretacion:** 
    - La variable con mayor coefiente es **{coef_df.iloc[0]['Variable']}** ({coef_df.iloc[0]['Coeficiente']:.4f}).
    - Esto significa que por cada unidad adicional en {coef_df.iloc[0]['Variable']}, 
      {target_col} aumenta en promedio {coef_df.iloc[0]['Coeficiente']:.4f} unidades, 
      manteniendo las demas variables constantes.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Grafico real vs predicho
    st.markdown("---")
    st.subheader("Valores Reales vs. Predichos")
    fig_lr, ax_lr = plt.subplots(figsize=(8, 6))
    ax_lr.scatter(y_test, y_pred_lr, alpha=0.7, edgecolors='k', linewidth=0.5, color='steelblue', s=60)
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    ax_lr.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax_lr.set_xlabel('Valores Reales', fontsize=12)
    ax_lr.set_ylabel('Valores Predichos', fontsize=12)
    ax_lr.set_title(f'Regresion Lineal - Reales vs Predichos\nR² = {r2_lr:.4f}', fontsize=12, fontweight='bold')
    ax_lr.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_lr)

# ============================================================
# TAB 4: ARBOL DE DECISION
# ============================================================

with tab4:
    st.header("🌳 Arbol de Decision para Regresion")
    
    st.markdown(f"""
    <div class="card">
        <h3>Hiperparametros del Modelo</h3>
        <p><b>Profundidad máxima (max_depth):</b> {max_depth}</p>
        <p><b>Min. muestras por division (min_samples_split):</b> {min_samples_split}</p>
        <p><b>Semilla aleatoria:</b> {random_state}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner('Entrenando Arbol de Decision...'):
        dt, y_pred_dt, r2_dt, mae_dt, rmse_dt = ejecutar_arbol_decision(
            X_train, X_test, y_train, y_test, 
            max_depth=max_depth, 
            min_samples_split=min_samples_split
        )
    
    # Metricas
    st.subheader("Metricas de Desempeno")
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""
    <div class="metric-card">
        <h3>R² (Coef. Determinacion)</h3>
        <div class="value">{r2_dt:.4f}</div>
        <p style="font-size:0.8rem; color:#666;">{r2_dt*100:.1f}% de varianza explicada</p>
    </div>
    """, unsafe_allow_html=True)
    col2.markdown(f"""
    <div class="metric-card">
        <h3>MAE</h3>
        <div class="value">{mae_dt:.4f}</div>
        <p style="font-size:0.8rem; color:#666;">Error absoluto medio</p>
    </div>
    """, unsafe_allow_html=True)
    col3.markdown(f"""
    <div class="metric-card">
        <h3>RMSE</h3>
        <div class="value">{rmse_dt:.4f}</div>
        <p style="font-size:0.8rem; color:#666;">Raiz del error cuadratico medio</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Importancia de variables
    st.subheader("Importancia de Variables")
    
    importances = dt.feature_importances_
    imp_df = pd.DataFrame({
        'Variable': feature_cols,
        'Importancia': importances,
        'Porcentaje': [f"{imp*100:.1f}%" for imp in importances]
    }).sort_values('Importancia', ascending=False)
    
    col_imp1, col_imp2 = st.columns([1, 1])
    
    with col_imp1:
        st.dataframe(imp_df, use_container_width=True, hide_index=True)
    
    with col_imp2:
        # Grafico de barras de importancia
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        colors_imp = ['steelblue'] + ['lightblue'] * (len(imp_df) - 1)
        ax_imp.barh(imp_df['Variable'], imp_df['Importancia'], color=colors_imp, edgecolor='k')
        ax_imp.set_xlabel('Importancia', fontsize=11)
        ax_imp.set_title('Importancia de Variables', fontsize=12, fontweight='bold')
        ax_imp.invert_yaxis()
        ax_imp.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_imp)
    
    # Interpretacion
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(f"""
    **🌳 Interpretacion del Arbol:**
    - La variable mas importante es **{imp_df.iloc[0]['Variable']}** con {imp_df.iloc[0]['Porcentaje']} de importancia.
    - Esto significa que {imp_df.iloc[0]['Variable']} es la variable que mas contribuye a las decisiones de particion del arbol.
    - El Arbol de Decision captura relaciones **no lineales** que la Regresion Lineal podria no detectar.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizacion del arbol
    st.subheader("Visualizacion del Arbol")
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_cols, filled=True, rounded=True, 
              fontsize=10, precision=2, ax=ax_tree)
    ax_tree.set_title(f"Arbol de Decision (max_depth={max_depth}, min_samples_split={min_samples_split})", 
                      fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_tree)
    
    # Grafico real vs predicho
    st.markdown("---")
    st.subheader("Valores Reales vs. Predichos")
    fig_dt_plot, ax_dt = plt.subplots(figsize=(8, 6))
    ax_dt.scatter(y_test, y_pred_dt, alpha=0.7, edgecolors='k', linewidth=0.5, color='seagreen', s=60)
    min_val = min(y_test.min(), y_pred_dt.min())
    max_val = max(y_test.max(), y_pred_dt.max())
    ax_dt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax_dt.set_xlabel('Valores Reales', fontsize=12)
    ax_dt.set_ylabel('Valores Predichos', fontsize=12)
    ax_dt.set_title(f'Arbol de Decision - Reales vs Predichos\nR² = {r2_dt:.4f}', fontsize=12, fontweight='bold')
    ax_dt.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_dt_plot)

# ============================================================
# TAB 5: COMPARACION
# ============================================================

with tab5:
    st.header("⚖️ Comparacion de Modelos")
    
    # Tabla comparativa
    st.subheader("Tabla Comparativa de Metricas")
    
    comparacion_df = pd.DataFrame({
        'Metrica': ['R²', 'MAE', 'RMSE'],
        'Regresion Lineal': [f"{r2_lr:.4f}", f"{mae_lr:.4f}", f"{rmse_lr:.4f}"],
        'Arbol de Decision': [f"{r2_dt:.4f}", f"{mae_dt:.4f}", f"{rmse_dt:.4f}"],
        'Diferencia': [
            f"{r2_dt - r2_lr:+.4f}",
            f"{mae_lr - mae_dt:+.4f}" if mae_lr >= mae_dt else f"{mae_dt - mae_lr:+.4f}",
            f"{rmse_lr - rmse_dt:+.4f}" if rmse_lr >= rmse_dt else f"{rmse_dt - rmse_lr:+.4f}"
        ]
    })
    
    st.dataframe(comparacion_df, use_container_width=True, hide_index=True)
    
    # Determinar mejor modelo
    mejor_r2 = max(r2_lr, r2_dt)
    mejor_modelo = "Regresion Lineal" if r2_lr >= r2_dt else "Arbol de Decision"
    
    st.markdown(f"""
    <div class="card">
        <h3>🏆 Modelo Recomendado: {mejor_modelo}</h3>
        <p>Basado en el mayor R² ({mejor_r2:.4f})</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Grafico comparativo
    st.subheader("Grafico Comparativo de Metricas")
    fig_comp = crear_grafico_comparacion_metricas(r2_lr, mae_lr, rmse_lr, r2_dt, mae_dt, rmse_dt)
    st.pyplot(fig_comp)
    
    # Grafico real vs predicho (ambos)
    st.subheader("Valores Reales vs Predichos - Ambos Modelos")
    fig_real_pred = crear_grafico_real_vs_predicho(y_test, y_pred_lr, y_pred_dt, r2_lr, r2_dt)
    st.pyplot(fig_real_pred)
    
    # Conclusion
    st.markdown("---")
    st.subheader("Conclusiones")
    
    st.markdown(f"""
    <div class="insight-box">
        <h4>📌 Analisis Comparativo</h4>
        <ul>
            <li><b>Regresion Lineal:</b> R² = {r2_lr:.4f}, MAE = {mae_lr:.4f}, RMSE = {rmse_lr:.4f}</li>
            <li><b>Arbol de Decision:</b> R² = {r2_dt:.4f}, MAE = {mae_dt:.4f}, RMSE = {rmse_dt:.4f}</li>
        </ul>
        <p><b>Recomendacion:</b> El modelo {'<b>Regresion Lineal</b>' if r2_lr >= r2_dt else '<b>Arbol de Decision</b>'} 
        ofrece mejor desempeño predictivo (mayor R²). Sin embargo, la Regresion Lineal 
        tiene la ventaja de ser directamente interpretable, lo que facilita la toma de 
        decisiones de negocio.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 6: PREDICCIONES
# ============================================================

with tab6:
    st.header("🔮 Realizar Predicciones")
    
    st.markdown("""
    <div class="card">
        <h3>Ingresa los valores para las variables predictoras</h3>
        <p>Ajusta los sliders para cada variable y obten la prediccion de ventas 
        utilizando ambos modelos entrenados.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sliders para cada variable predictora
    st.subheader("Valores de Entrada")
    
    cols_sliders = st.columns(min(2, len(feature_cols)))
    valores_prediccion = {}
    
    for i, col in enumerate(feature_cols):
        with cols_sliders[i % 2]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            valores_prediccion[col] = st.slider(
                f"{col}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=round((max_val - min_val) / 100, 1),
                help=f"Rango: {min_val:.1f} a {max_val:.1f}"
            )
    
    st.markdown("---")
    
    # Crear array de entrada con los valores actuales de los sliders
    X_pred = np.array([[valores_prediccion[col] for col in feature_cols]])
    
    # Predicciones en tiempo real (sin boton)
    pred_lr = lr.predict(X_pred)[0]
    pred_dt = dt.predict(X_pred)[0]
    
    # --- Mostrar resultados en tiempo real ---
    st.subheader("Resultados de la Prediccion (Tiempo Real)")
    
    # Layout: 3 columnas - LR, DT, Diferencia
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: steelblue;">
            <h3>📐 Regresion Lineal</h3>
            <div class="value" style="color: steelblue;">{pred_lr:.2f}</div>
            <p style="font-size:0.9rem; color:#666;">{target_col} estimado</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: seagreen;">
            <h3>🌳 Arbol de Decision</h3>
            <div class="value" style="color: seagreen;">{pred_dt:.2f}</div>
            <p style="font-size:0.9rem; color:#666;">{target_col} estimado</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mejor_pred = "Regresion Lineal" if r2_lr >= r2_dt else "Arbol de Decision"
        st.markdown(f"""
        <div class="metric-card" style="border-top-color: {'steelblue' if r2_lr >= r2_dt else 'seagreen'};">
            <h3>⚡ Diferencia</h3>
            <div class="value" style="color: {'steelblue' if r2_lr >= r2_dt else 'seagreen'};">
                {abs(pred_lr - pred_dt):.2f}
            </div>
            <p style="font-size:0.9rem; color:#666;">entre ambos modelos</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Mostrar valores usados
    st.markdown("---")
    st.markdown("**Valores actuales de las variables:**")
    valores_df = pd.DataFrame([valores_prediccion])
    
    # Formatear columnas
    for col in valores_df.columns:
        valores_df[col] = valores_df[col].apply(lambda x: f"{x:.1f}")
    
    col_val1, col_val2, col_val3 = st.columns(3)
    for i, col in enumerate(feature_cols):
        with [col_val1, col_val2, col_val3][i % 3]:
            st.metric(label=col, value=f"${valores_prediccion[col]:.1f}")
    
    # Grafico de barras comparativo de las predicciones
    st.markdown("---")
    st.subheader("Comparacion Visual de Predicciones")
    
    fig_pred, ax_pred = plt.subplots(figsize=(6, 3.5))
    modelos = ['Regresion Lineal', 'Arbol de Decision']
    valores_pred = [pred_lr, pred_dt]
    colores = ['steelblue', 'seagreen']
    
    bars = ax_pred.bar(modelos, valores_pred, color=colores, edgecolor='k', width=0.5)
    for bar, val in zip(bars, valores_pred):
        ax_pred.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax_pred.set_ylabel(target_col, fontsize=11)
    ax_pred.set_title(f'Prediccion de {target_col} segun cada modelo', fontsize=12, fontweight='bold')
    ax_pred.grid(axis='y', alpha=0.3)
    ax_pred.set_ylim(0, max(valores_pred) * 1.2)
    plt.tight_layout()
    st.pyplot(fig_pred)
    
    # Interpretacion de la prediccion en tiempo real
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    
    # Escenario interpretativo
    if pred_lr > 20:
        escenario = "alto"
        consejo = "Excelente rendimiento de campana. Considere mantener o incrementar la inversion."
    elif pred_lr > 12:
        escenario = "moderado"
        consejo = "Rendimiento aceptable. Evalue reasignar presupuesto de Newspaper a TV o Radio."
    else:
        escenario = "bajo"
        consejo = "Rendimiento bajo. Revise la estrategia de medios y considere aumentar la inversion en TV y Radio."
    
    st.markdown(f"""
    **💡 Interpretacion en Tiempo Real:**
    - Con los valores ingresados (TV=${valores_prediccion.get('TV',0):.0f}, Radio=${valores_prediccion.get('Radio',0):.0f}, Newspaper=${valores_prediccion.get('Newspaper',0):.0f}):
        - 📐 **Regresion Lineal** predice **{pred_lr:.2f}** unidades de {target_col}.
        - 🌳 **Arbol de Decision** predice **{pred_dt:.2f}** unidades de {target_col}.
    - **Escenario:** Nivel de ventas **{escenario}**.
    - **Consejo:** {consejo}
    - La diferencia de {abs(pred_lr - pred_dt):.2f} unidades entre ambos modelos refleja las distintas formas en que cada uno captura la relacion entre las variables.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div style="text-align:center; color:#888; font-size:0.8rem; padding:1rem;">
        <p>💡 <b>Tip:</b> Ajusta los sliders para simular diferentes escenarios de inversion 
        publicitaria. Las predicciones se actualizan <b>automaticamente</b> en tiempo real.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <p><b>ML Predictor</b> | Desarrollado con Python, Streamlit, Scikit-learn | 
    Maestria en Economia con Mencion en IA Aplicada - Machine Learning para Negocios</p>
    <p>© 2026 - Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)