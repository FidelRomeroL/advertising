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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

st.set_page_config(page_title="ML Predictor - Ventas", page_icon="📊", layout="wide")
st.title("📊 Análisis Predictivo de Ventas Publicitarias")
st.markdown("---")

# ============================================================
# CARGA DE DATOS Y ENTRENAMIENTO
# ============================================================
@st.cache_resource
def cargar_y_entrenar():
    df = pd.read_csv("advertising.csv")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    dt = DecisionTreeRegressor(max_depth=5, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    
    return df, X_train, X_test, y_train, y_test, lr, dt, y_pred_lr, y_pred_dt


df, X_train, X_test, y_train, y_test, lr, dt, y_pred_lr, y_pred_dt = cargar_y_entrenar()

intercept = lr.intercept_
b_tv, b_radio, b_news = lr.coef_

# ============================================================
# FUNCIONES AUXILIARES - ÁRBOL DE DECISIÓN
# ============================================================
def get_tree_path(model, features, feature_names):
    """Recorre el árbol y devuelve los IDs de los nodos visitados."""
    tree = model.tree_
    node = 0
    path_ids = [node]
    while tree.children_left[node] != tree.children_right[node]:
        idx = tree.feature[node]
        th = tree.threshold[node]
        fval = features[idx]
        if fval <= th:
            node = tree.children_left[node]
        else:
            node = tree.children_right[node]
        path_ids.append(node)
    return path_ids

def get_leaf_value(model, features):
    """Obtiene el valor de la hoja destino para las features dadas."""
    tree = model.tree_
    node = 0
    while tree.children_left[node] != tree.children_right[node]:
        idx = tree.feature[node]
        th = tree.threshold[node]
        fval = features[idx]
        node = tree.children_left[node] if fval <= th else tree.children_right[node]
    return tree.value[node][0][0], tree.n_node_samples[node]

def draw_tree_full(model, features, feature_names, figsize=(18, 10)):
    """Dibuja el árbol de decisión completo con información en el título."""
    import matplotlib
    matplotlib.use('Agg')
    plt.ioff()
    
    try:
        leaf_val, n_samp_path = get_leaf_value(model, features)
        
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(model, feature_names=feature_names, filled=True,
                  rounded=True, precision=2, fontsize=8, ax=ax)
        
        ax.set_title(
            f"🌳 Árbol de Decisión Completo\n"
            f"TV={features[0]:.0f} | Radio={features[1]:.0f} | NP={features[2]:.0f} → Predicción: {leaf_val:.2f}",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=figsize)
        try:
            plot_tree(model, feature_names=feature_names, filled=True,
                      rounded=True, precision=2, fontsize=8, ax=ax)
        except:
            pass
        leaf_val, _ = get_leaf_value(model, features)
        ax.set_title(f"🌳 Árbol de Decisión | Predicción: {leaf_val:.2f}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

def draw_path_diagram(model, features, feature_names, figsize=(12, 6)):
    """
    Dibuja un DIAGRAMA DE FLUJO mostrando solo el CAMINO que sigue la predicción actual.
    Cada nodo del camino se muestra como un rectángulo con la decisión,
    conectado por flechas de colores.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.ioff()
    
    try:
        path_ids = get_tree_path(model, features, feature_names)
        tree = model.tree_
        leaf_val, _ = get_leaf_value(model, features)
        n_pasos = len(path_ids)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Altura de cada paso (distribuir uniformemente de arriba a abajo)
        step_height = 0.85 / max(n_pasos, 1)
        start_y = 0.90
        
        box_width = 0.6
        box_height = step_height * 0.7
        
        for i, nid in enumerate(path_ids):
            y_pos = start_y - i * step_height
            x_center = 0.5
            
            is_leaf = (tree.children_left[nid] == tree.children_right[nid])
            
            if not is_leaf and i < n_pasos - 1:
                idx = tree.feature[nid]
                th = tree.threshold[nid]
                fname = feature_names[idx]
                fval = features[idx]
                go_left = fval <= th
                
                # Texto del nodo de decisión
                node_text = f"¿{fname} ≤ {th:.2f}?"
                decision_text = f"{fname} = {fval:.1f}  ->  {'SI (menor/igual)' if go_left else 'NO (mayor)'}"
                
                # Color del nodo: azul para decisión
                edge_color = '#2980b9' if i == 0 else '#d35400'
                
            else:
                # Nodo hoja
                val = tree.value[nid][0][0]
                node_text = "PREDICCION (Hoja)"
                decision_text = f"Ventas = {val:.2f} unidades"
                edge_color = '#27ae60'
            
            # Dibujar rectángulo con FancyBboxPatch
            rect = FancyBboxPatch(
                (x_center - box_width/2, y_pos - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.05",
                facecolor='#fef9e7' if not is_leaf else '#e8f8f5',
                edgecolor=edge_color,
                linewidth=3 if not is_leaf else 4,
                zorder=10
            )
            ax.add_patch(rect)
            
            # Texto del nodo
            ax.text(x_center, y_pos + box_height*0.15, node_text,
                   ha='center', va='center', fontsize=11, fontweight='bold', zorder=11)
            ax.text(x_center, y_pos - box_height*0.15, decision_text,
                   ha='center', va='center', fontsize=9, fontstyle='italic', zorder=11,
                   color='#2c3e50')
            
            # Etiqueta de paso
            ax.text(x_center - box_width/2 - 0.04, y_pos, f'{i+1}',
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=edge_color, edgecolor='white', linewidth=2),
                   color='white', zorder=12)
        
        # Flechas entre pasos
        for i in range(n_pasos - 1):
            y_from = start_y - i * step_height - box_height/2
            y_to = start_y - (i + 1) * step_height + box_height/2
            
            arrow = FancyArrowPatch(
                (0.5, y_from), (0.5, y_to),
                arrowstyle='->,head_length=0.8,head_width=0.08',
                color='#e74c3c', linewidth=3, zorder=9
            )
            ax.add_patch(arrow)
            
            # Dirección
            nid = path_ids[i]
            idx = tree.feature[nid]
            th = tree.threshold[nid]
            fname = feature_names[idx]
            fval = features[idx]
            go_left = fval <= th
            
            ax.text(0.5 + box_width/2 + 0.02, (y_from + y_to)/2,
                   f'{fname} ≤ {th}' if go_left else f'{fname} > {th}',
                   ha='left', va='center', fontsize=8, color='#c0392b',
                   fontstyle='italic', zorder=9)
        
        ax.set_title(
            f"🔴 RUTA DE DECISIONES (Camino seguido → Predicción: {leaf_val:.2f})",
            fontsize=13, fontweight="bold", color='#2c3e50', pad=10
        )
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        # Fallback: gráfico simple con texto
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"🌳 Camino: {len(path_ids)} pasos → Predicción: {leaf_val:.2f}",
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig

# ============================================================
# SIDEBAR - NAVEGACION
# ============================================================
try:
    st.sidebar.image("fidel.jpeg", width=150, caption="")
except:
    st.sidebar.markdown("📷 *Foto no disponible*")
st.sidebar.markdown("**Econ. Fidel Romero León**", unsafe_allow_html=False)
st.sidebar.markdown("---")
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
    st.subheader("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Exploración visual y estadística del dataset de publicidad.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📦 Registros", df.shape[0])
    with col2: st.metric("📋 Variables predictoras", 3)
    with col3: st.metric("🎯 Variable objetivo", "Sales")
    with col4: st.metric("💰 Inversión total", f"${df['TV'].sum()+df['Radio'].sum()+df['Newspaper'].sum():,.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Primeros 10 registros del dataset**")
        st.dataframe(df.head(10), width='stretch')
    with col2:
        st.markdown("**📈 Estadísticas descriptivas**")
        st.dataframe(df.describe().round(2), width='stretch')
    
    st.markdown("---")
    
    st.markdown("**🔥 Matriz de Correlación**")
    fig, ax = plt.subplots(figsize=(8, 5))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, cmap="RdYlBu", center=0,
                square=True, linewidths=0.8, mask=mask, ax=ax,
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title("Correlación entre variables", fontsize=14, fontweight="bold")
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    st.markdown("**📊 Relación de cada medio con las ventas**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df['TV'], df['Sales'], alpha=0.6, c='#2ecc71', edgecolors='white', s=60)
        z = np.polyfit(df['TV'], df['Sales'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['TV'].min(), df['TV'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2)
        ax.set_xlabel("Inversión en TV ($)")
        ax.set_ylabel("Ventas (unidades)")
        ax.set_title("TV vs Ventas", fontweight="bold", fontsize=13)
        st.pyplot(fig)
        plt.close()
        st.caption(f"Correlación TV-Sales: **{df['TV'].corr(df['Sales']):.4f}**")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df['Radio'], df['Sales'], alpha=0.6, c='#f39c12', edgecolors='white', s=60)
        z = np.polyfit(df['Radio'], df['Sales'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['Radio'].min(), df['Radio'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2)
        ax.set_xlabel("Inversión en Radio ($)")
        ax.set_ylabel("Ventas (unidades)")
        ax.set_title("Radio vs Ventas", fontweight="bold", fontsize=13)
        st.pyplot(fig)
        plt.close()
        st.caption(f"Correlación Radio-Sales: **{df['Radio'].corr(df['Sales']):.4f}**")
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df['Newspaper'], df['Sales'], alpha=0.6, c='#9b59b6', edgecolors='white', s=60)
        z = np.polyfit(df['Newspaper'], df['Sales'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['Newspaper'].min(), df['Newspaper'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2)
        ax.set_xlabel("Inversión en Newspaper ($)")
        ax.set_ylabel("Ventas (unidades)")
        ax.set_title("Newspaper vs Ventas", fontweight="bold", fontsize=13)
        st.pyplot(fig)
        plt.close()
        st.caption(f"Correlación Newspaper-Sales: **{df['Newspaper'].corr(df['Sales']):.4f}**")
    
    with col2:
        st.markdown("**📊 Distribución de Ventas**")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['Sales'], bins=20, kde=True, color="#1e3a5f", ax=ax)
        ax.axvline(df['Sales'].mean(), color='red', linestyle='--', linewidth=2, label=f"Media={df['Sales'].mean():.2f}")
        ax.axvline(df['Sales'].median(), color='green', linestyle='--', linewidth=2, label=f"Mediana={df['Sales'].median():.2f}")
        ax.set_xlabel("Ventas (unidades)")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de Sales", fontsize=12, fontweight="bold")
        ax.legend()
        st.pyplot(fig)
        plt.close()

# ============================================================
# PAGINA 2: REGRESION LINEAL
# ============================================================
elif pagina == "📐 Regresión Lineal":
    st.subheader("📐 Modelo de Regresión Lineal Múltiple")
    st.markdown("---")
    
    st.markdown("### 📝 Ecuación del modelo estimado")
    
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    
    st.code(f"Sales = {intercept:.4f} + ({b_tv:.4f} × TV) + ({b_radio:.4f} × Radio) + ({b_news:.4f} × Newspaper)", language="text")
    
    st.info(f"**Ejemplo:** TV=$25, Radio=$0, Newspaper=$0 → Sales = {intercept:.4f} + ({b_tv:.4f}×25) = **{intercept + b_tv*25:.2f}** unidades")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📺 β₁ TV", f"{b_tv:.4f}", delta=f"+{b_tv:.4f} por $1")
        st.caption(f"Cada $1 adicional en TV aumenta ventas en {b_tv:.4f} unidades")
    with col2:
        st.metric("📻 β₂ Radio", f"{b_radio:.4f}", delta=f"+{b_radio:.4f} por $1")
        st.caption(f"Cada $1 adicional en Radio aumenta ventas en {b_radio:.4f} unidades")
    with col3:
        st.metric("📰 β₃ Newspaper", f"{b_news:.4f}", delta=f"+{b_news:.4f} por $1")
        st.caption(f"Cada $1 adicional en Newspaper aumenta ventas en {b_news:.4f} unidades")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 R² (Precisión)", f"{r2_lr:.4f}", help=f"Explica el {r2_lr*100:.2f}% de la variabilidad")
        st.caption(f"El modelo explica el **{r2_lr*100:.2f}%** de las ventas")
    with col2:
        st.metric("📏 MAE (Error absoluto)", f"{mae_lr:.4f}", help=f"Error promedio de ±{mae_lr:.4f} unidades")
        st.caption(f"Error promedio: **±{mae_lr:.4f}** unidades")
    with col3:
        st.metric("📐 RMSE", f"{rmse_lr:.4f}", help="Raíz del error cuadrático medio")
        st.caption(f"RMSE: **{rmse_lr:.4f}**")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Impacto de cada variable (Coeficientes)**")
        fig, ax = plt.subplots(figsize=(7, 4))
        coefs = [b_tv, b_radio, b_news]
        labels = ['TV', 'Radio', 'Newspaper']
        colors = ['#2ecc71', '#f39c12', '#9b59b6']
        bars = ax.barh(labels, coefs, color=colors, edgecolor='black', height=0.6)
        for bar, val in zip(bars, coefs):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel("Coeficiente (impacto en ventas por $1)")
        ax.set_title("Impacto de cada medio publicitario", fontweight="bold")
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**📈 Valores Reales vs Predichos**")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred_lr, alpha=0.6, color='#1e3a5f', s=60, edgecolors='white', linewidth=0.5)
        min_val = min(y_test.min(), y_pred_lr.min())
        max_val = max(y_test.max(), y_pred_lr.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
        ax.set_xlabel("Valores reales")
        ax.set_ylabel("Valores predichos")
        ax.set_title(f"Reales vs Predichos (R² = {r2_lr:.4f})", fontweight="bold")
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    with st.expander("📖 Interpretación completa del modelo"):
        st.markdown(f"""
### Interpretación de la Regresión Lineal

**Ecuación estimada:**
```
Sales = {intercept:.4f} + {b_tv:.4f}·TV + {b_radio:.4f}·Radio + {b_news:.4f}·Newspaper
```

**1. Intercepto (β₀ = {intercept:.4f}):**
- Si no se invierte en ningún medio (TV=0, Radio=0, Newspaper=0), se esperan **{intercept:.2f}** unidades de venta.

**2. Coeficiente de TV (β₁ = {b_tv:.4f}):**
- Por cada $1 adicional en TV, las ventas aumentan **{b_tv:.4f}** unidades (manteniendo lo demás constante).

**3. Coeficiente de Radio (β₂ = {b_radio:.4f}):**
- Por cada $1 adicional en Radio, las ventas aumentan **{b_radio:.4f}** unidades.

**4. Coeficiente de Newspaper (β₃ = {b_news:.4f}):**
- Por cada $1 adicional en Newspaper, las ventas aumentan **{b_news:.4f}** unidades.

**5. Calidad del modelo:**
- **R² = {r2_lr:.4f}**: El modelo explica el **{r2_lr*100:.2f}%** de la variabilidad.
- **MAE = {mae_lr:.4f}**: Error promedio de ±{mae_lr:.2f} unidades.
""")

# ============================================================
# PAGINA 3: ARBOL DE DECISION
# ============================================================
elif pagina == "🌳 Árbol de Decisión":
    st.subheader("🌳 Árbol de Decisión para Regresión")
    st.markdown("Modelo no lineal que aprende reglas de decisión a partir de los datos.")
    
    r2_dt = r2_score(y_test, y_pred_dt)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🎯 R² (Precisión)", f"{r2_dt:.4f}", help=f"Explica el {r2_dt*100:.2f}% de la varianza")
    with col2: st.metric("📏 MAE", f"{mae_dt:.4f}", help=f"Error promedio de ±{mae_dt:.4f} unidades")
    with col3: st.metric("📐 RMSE", f"{rmse_dt:.4f}")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1: st.info(f"**🌳 Profundidad máxima:** {dt.get_params()['max_depth']}")
    with col2: st.info(f"**✂️ Mín. muestras por división:** {dt.get_params()['min_samples_split']}")
    with col3: st.info(f"**🍃 Número de hojas:** {dt.get_n_leaves()}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⭐ Importancia de las Variables**")
        importancia = dt.feature_importances_
        feat_names = ['TV', 'Radio', 'Newspaper']
        
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#2ecc71', '#f39c12', '#9b59b6']
        bars = ax.barh(feat_names, importancia, color=colors, edgecolor='black', height=0.6)
        for bar, val in zip(bars, importancia):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center', fontweight='bold')
        ax.set_xlabel("Importancia Relativa")
        ax.set_title("Importancia de cada medio", fontweight="bold")
        ax.set_xlim(0, 1.1)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**📈 Valores Reales vs Predichos**")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred_dt, alpha=0.6, color='#2e7d32', s=60, edgecolors='white', linewidth=0.5)
        min_val = min(y_test.min(), y_pred_dt.min())
        max_val = max(y_test.max(), y_pred_dt.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
        ax.set_xlabel("Valores reales")
        ax.set_ylabel("Valores predichos")
        ax.set_title(f"Reales vs Predichos (R² = {r2_dt:.4f})", fontweight="bold")
        ax.legend()
        st.pyplot(fig)
        plt.close()
    
    st.markdown(f"""
**Análisis de importancia:**
- **📺 TV:** {importancia[0]*100:.1f}% de importancia predictiva
- **📻 Radio:** {importancia[1]*100:.1f}% de importancia predictiva
- **📰 Newspaper:** {importancia[2]*100:.1f}% de importancia predictiva
""")
    
    with st.expander("🌳 ¿Cómo funciona el Árbol de Decisión? Explicación completa"):
        st.markdown("""
## 🌳 ¿Cómo funciona el Árbol de Decisión?

### Concepto general

A diferencia de la regresión lineal (que usa una fórmula matemática global), el **Árbol de Decisión** funciona como un **sistema de preguntas y respuestas secuenciales**.

Imagina que eres un vendedor y quieres predecir cuánto venderás. En lugar de usar una fórmula, haces preguntas como:

1. **¿Inviertes mucho en TV?** (Sí/No)
2. **¿Inviertes en Radio?** (Sí/No)
3. **¿Inviertes en Periódico?** (Sí/No)

Cada respuesta te lleva a una nueva pregunta, hasta llegar a una **hoja** que te da la predicción final.

---

### 🔬 El árbol entrenado (con nuestros datos)

```
                        ┌─────────────────────┐
                        │  ¿TV ≤ 108.60?        │
                        │  (¿Inversión en TV    │
                        │   es baja o alta?)    │
                        └──────────┬──────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │  SÍ (TV baja)                       │  NO (TV alta)
                ▼                                     ▼
     ┌─────────────────────┐               ┌─────────────────────┐
     │  ¿TV ≤ 32.75?       │               │  ¿Radio ≤ 26.85?    │
     └──────────┬──────────┘               └──────────┬──────────┘
                │                                     │
     ┌──────────┴──────────┐                ┌─────────┴─────────┐
     │ SÍ                  │ NO             │ SÍ                │ NO
     ▼                     ▼                ▼                   ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    ┌─────────────┐
  │¿Newspaper   │   │¿Radio       │   │¿Radio       │    │¿TV ≤ 204.55?│
  │ ≤ 9.00?     │   │ ≤ 42.30?    │   │ ≤ 10.05?    │    └──────┬──────┘
  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
         │                 │                 │             ┌─────┴─────┐
    ┌────┴────┐       ┌────┴────┐       ┌────┴────┐    ┌────┴────┐ ┌────┴────┐
    │SÍ: 3.20│       │SÍ: 7.60│       │SÍ: 11.20│    │ TV ≤    │ │ Radio   │
    │        │       │NO: 9.65│       │NO: 10.30│    │150.60?  │ │ ≤ 34.90?│
    │NO: 5.60│       │        │       │         │    └────┬────┘ └────┬────┘
    │ o 7.07 │       │Radio   │       │Radio    │         │           │
    │ o 6.90 │       │>13.30? │       │>7.20?   │    ┌────┴────┐ ┌────┴────┐
    │ o 8.88 │       └────┬───┘       └────┬────┘    │14.80   │ │18.60   │
    └────────┘            │                 │         │17.19   │ │19.93   │
                    ┌─────┴─────┐     ┌─────┴─────┐   │18.30   │ │22.51   │
                    │14.55      │     │12.63      │   │20.50   │ │25.26   │
                    │(Radio alta)│     │14.33      │   └────────┘ └────────┘
                    └───────────┘     │15.49      │
                                       │17.14      │
                                       └───────────┘
```

---

### 📋 Las 22 reglas de decisión (hojas del árbol)

El árbol aprendió **22 grupos distintos** de clientes. Para cada grupo, asigna un valor de ventas:

| # | Condiciones (ruta en el árbol) | Ventas |
|---|-------------------------------|--------|
| 1 | TV ≤ 32.75, Newspaper ≤ 9 | **3.20** |
| 2 | TV ≤ 17.95, Newspaper > 9, Radio ≤ 23.70 | **5.60** |
| 3 | TV 17.95-32.75, Newspaper > 9, Radio ≤ 23.70 | **7.07** |
| 4 | TV ≤ 12.95, Newspaper > 9, Radio > 23.70 | **6.90** |
| 5 | TV 12.95-32.75, Newspaper > 9, Radio > 23.70 | **8.88** |
| 6 | TV 32.75-48.90, Radio ≤ 13.30 | **7.60** |
| 7 | TV 32.75-48.90, Radio 13.30-42.30 | **10.56** |
| 8 | TV 32.75-74.85, Radio 13.30-42.30 | **10.56** |
| 9 | TV 74.85-108.60, Radio 13.30-42.30 | **12.41** |
| 10 | TV 32.75-108.60, Radio > 42.30 | **14.55** |
| ... | ... | ... |
| 22 | TV > 242.45, Radio > 34.90 | **25.26** |

---

### 🧮 Ejemplo paso a paso: TV=25, Radio=0, Newspaper=0

Sigamos el camino del árbol para este caso:

```
Paso 1: ¿TV ≤ 108.60? → 25 ≤ 108.60 → SÍ (bajo)
Paso 2: ¿TV ≤ 32.75? → 25 ≤ 32.75 → SÍ (muy bajo)
Paso 3: ¿Newspaper ≤ 9.00? → 0 ≤ 9 → SÍ
Resultado: 3.20 unidades
```

**Predicción: 3.20 unidades** ✅

---

### 🧮 Ejemplo paso a paso: TV=230, Radio=40, Newspaper=50

```
Paso 1: ¿TV ≤ 108.60? → NO (alto)  
Paso 2: ¿Radio ≤ 26.85? → NO (alto)
Paso 3: ¿TV ≤ 204.55? → NO (muy alto)
Paso 4: ¿Radio ≤ 34.90? → NO (alto)
Paso 5: ¿TV ≤ 242.45? → 230 ≤ 242.45 → SÍ
Resultado: 22.51 unidades
```

**Predicción: 22.51 unidades** ✅

---

### 🆚 Diferencia clave con Regresión Lineal

| Aspecto | Regresión Lineal | Árbol de Decisión |
|---------|-----------------|-------------------|
| **Fórmula** | Una ecuación global: `Sales = 2.98 + 0.045·TV + 0.189·Radio + 0.003·Newspaper` | 22 reglas locales (si-entonces) |
| **Relaciones** | Lineales (una línea recta) | No lineales (segmentos) |
| **Interpretación** | Coeficientes (efecto marginal) | Reglas de decisión (categorías) |
| **TV = 25** | 2.98 + 0.045×25 = **4.10** | Va a la hoja → **3.20** |
| **TV = 230, Radio=40** | 2.98 + 0.045×230 + 0.189×40 + 0.003×50 = **18.79** | Va a la hoja → **22.51** |
| **¿Por qué diferente?** | Asume que la relación es siempre lineal | Detecta patrones no lineales |

La Regresión Lineal da **4.10** para TV=25 porque la línea recta pasa por ese punto. El Árbol da **3.20** porque en los datos reales, las ventas con inversión muy baja en TV son menores de lo que la línea recta sugiere.

---

### 📊 Ventajas y desventajas

**✅ Ventajas:**
- Captura relaciones **no lineales** automáticamente
- No necesita escalar los datos
- Fácil de visualizar y explicar a no técnicos
- Identifica qué variables son más importantes

**❌ Desventajas:**
- **Sobreajuste** (overfitting): puede memorizar el ruido
- Pequeños cambios en los datos pueden cambiar todo el árbol
- Menos preciso para extrapolar (valores fuera del rango de entrenamiento)

---

### 🏁 Conclusión

El Árbol de Decisión es como un **vendedor experto** que ha visto 22 tipos diferentes de escenarios publicitarios. Cuando llega un nuevo caso, hace preguntas secuenciales (¿cuánto invertiste en TV? ¿en Radio? ¿en Periódico?) hasta encontrar el grupo más parecido y dar su predicción basada en **casos similares anteriores**.
""")

# ============================================================
# PAGINA 4: COMPARACION
# ============================================================
elif pagina == "⚖️ Comparación de Modelos":
    st.subheader("⚖️ Comparación: Regresión Lineal vs Árbol de Decisión")
    
    r2_lr = r2_score(y_test, y_pred_lr)
    r2_dt = r2_score(y_test, y_pred_dt)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mae_dt = mean_absolute_error(y_test, y_pred_dt)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    
    st.markdown("### 📋 Tabla comparativa de métricas")
    comparacion = pd.DataFrame({
        "Métrica": ["R² (Precisión)", "MAE (Error absoluto)", "RMSE (Error cuadrático)"],
        "📐 Regresión Lineal": [f"{r2_lr:.4f}", f"{mae_lr:.4f}", f"{rmse_lr:.4f}"],
        "🌳 Árbol de Decisión": [f"{r2_dt:.4f}", f"{mae_dt:.4f}", f"{rmse_dt:.4f}"],
        "🏆 Mejor": [
            "📐 RL" if r2_lr >= r2_dt else "🌳 DT",
            "🌳 DT" if mae_dt <= mae_lr else "📐 RL",
            "🌳 DT" if rmse_dt <= rmse_lr else "📐 RL"
        ]
    })
    st.dataframe(comparacion, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Precisión (R²)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Regresión Lineal', 'Árbol Decisión'], [r2_lr, r2_dt],
               color=['#1e3a5f', '#2e7d32'], edgecolor='black', width=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("R² (mayor = mejor)")
        for i, v in enumerate([r2_lr, r2_dt]):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**📊 Error (MAE)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Regresión Lineal', 'Árbol Decisión'], [mae_lr, mae_dt],
               color=['#1e3a5f', '#2e7d32'], edgecolor='black', width=0.5)
        ax.set_ylabel("MAE (menor = mejor)")
        for i, v in enumerate([mae_lr, mae_dt]):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    st.markdown("**📈 Comparación: Reales vs Predichos (ambos modelos)**")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred_lr, alpha=0.7, color='#1e3a5f', s=50, label='Regresión Lineal', edgecolors='white')
    ax.scatter(y_test, y_pred_dt, alpha=0.7, color='#2e7d32', s=50, label='Árbol Decisión', marker='s', edgecolors='white')
    min_val = min(y_test.min(), y_pred_lr.min(), y_pred_dt.min())
    max_val = max(y_test.max(), y_pred_lr.max(), y_pred_dt.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
    ax.set_xlabel("Valores reales", fontsize=12)
    ax.set_ylabel("Valores predichos", fontsize=12)
    ax.set_title("Comparación de modelos", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    mejor_modelo = "📐 Regresión Lineal" if r2_lr >= r2_dt else "🌳 Árbol de Decisión"
    st.success(f"### 🏆 Conclusión: El mejor modelo es **{mejor_modelo}**")
    
    if r2_lr >= r2_dt:
        st.markdown(f"""
- **R²:** RL = {r2_lr:.4f} vs DT = {r2_dt:.4f}
- La **Regresión Lineal** explica mejor la relación entre inversión y ventas.
- **Recomendación:** Usar Regresión Lineal para presupuestos y planeación.
""")
    else:
        st.markdown(f"""
- **R²:** DT = {r2_dt:.4f} vs RL = {r2_lr:.4f}
- El **Árbol de Decisión** captura mejor las relaciones no lineales.
- **Recomendación:** Usar Árbol de Decisión.
""")

# ============================================================
# PAGINA 5: PREDICTOR INTERACTIVO
# ============================================================
elif pagina == "🎯 Predictor Interactivo":
    st.subheader("🎯 Simulador de Ventas en Tiempo Real")
    st.markdown("Ajusta los sliders para simular diferentes escenarios de inversión publicitaria.")
    
    st.markdown("### 🔧 Inversión en cada medio")
    col1, col2, col3 = st.columns(3)
    with col1:
        tv = st.slider("📺 **TV ($)**", 0, 300, 150)
        st.caption(f"Rango: $0 - ${df['TV'].max():.0f}")
    with col2:
        radio = st.slider("📻 **Radio ($)**", 0, 50, 25)
        st.caption(f"Rango: $0 - ${df['Radio'].max():.0f}")
    with col3:
        newspaper = st.slider("📰 **Newspaper ($)**", 0, 120, 30)
        st.caption(f"Rango: $0 - ${df['Newspaper'].max():.0f}")
    
    X_pred = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
    pred_lr = lr.predict(X_pred)[0]
    pred_dt = dt.predict(X_pred)[0]
    
    st.markdown("---")
    st.markdown("### 🎯 Resultados de la predicción")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown("---")
            st.markdown("#### 📐 Regresión Lineal")
            st.markdown(f"## **{pred_lr:.2f}** unidades")
            st.markdown(f"R² = {r2_score(y_test, y_pred_lr):.4f}")
    
    with col2:
        with st.container():
            st.markdown("---")
            st.markdown("#### 🌳 Árbol de Decisión")
            st.markdown(f"## **{pred_dt:.2f}** unidades")
            st.markdown(f"R² = {r2_score(y_test, y_pred_dt):.4f}")
    
    st.markdown("---")
    st.markdown("### 🔍 Verificación paso a paso - Regresión Lineal")
    
    st.markdown(f"""
**Fórmula:** Sales = β₀ + β₁·TV + β₂·Radio + β₃·Newspaper

**Sustitución:** 
Sales = {intercept:.4f} + ({b_tv:.4f} × {tv}) + ({b_radio:.4f} × {radio}) + ({b_news:.4f} × {newspaper})

**Cálculo:**
Sales = {intercept:.4f} + ({b_tv*tv:.4f}) + ({b_radio*radio:.4f}) + ({b_news*newspaper:.4f})

**Resultado:** **{pred_lr:.4f} ≈ {pred_lr:.2f}** unidades
""")
    
    st.success(f"✅ Verificación: La predicción del modelo ({pred_lr:.4f}) coincide con el cálculo manual ✅")
    
    st.markdown("---")
    st.markdown("### 🌳 Árbol de Decisión - Camino resaltado")
    st.markdown("""
    El siguiente gráfico muestra el **árbol de decisión completo**. Los nodos y flechas en **rojo 🔴** 
    marcan el camino exacto que siguió el árbol para llegar a la predicción con **tus valores actuales**:
    """)
    
    st.info(f"**Tus valores:** 📺 TV=${tv} | 📻 Radio=${radio} | 📰 Newspaper=${newspaper}")
    
    # Mostrar el árbol completo y el diagrama del camino
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌳 Árbol completo**")
        fig_full = draw_tree_full(dt, [tv, radio, newspaper], ['TV', 'Radio', 'Newspaper'], figsize=(12, 8))
        st.pyplot(fig_full)
        plt.close()
    with col2:
        st.markdown("**🔴 Diagrama del camino seguido**")
        fig_path = draw_path_diagram(dt, [tv, radio, newspaper], ['TV', 'Radio', 'Newspaper'], figsize=(12, 8))
        st.pyplot(fig_path)
        plt.close()
    
    st.markdown("---")
    st.markdown("### 📋 Pasos del camino seguido")
    st.markdown("El recorrido paso a paso que el árbol de decisión siguió con tus datos:")
    
    tree_path = get_tree_path(dt, [tv, radio, newspaper], ['TV', 'Radio', 'Newspaper'])
    tree = dt.tree_
    
    for i in range(len(tree_path) - 1):
        nid = tree_path[i]
        next_id = tree_path[i + 1]
        idx = tree.feature[nid]
        th = tree.threshold[nid]
        fname = ['TV', 'Radio', 'Newspaper'][idx]
        fval = [tv, radio, newspaper][idx]
        go_left = fval <= th
        direction = "izquierda" if go_left else "derecha"
        decision_str = f"{fval:.1f} ≤ {th:.1f}" if go_left else f"{fval:.1f} > {th:.1f}"
        n_samples = tree.n_node_samples[nid]
        
        c1, c2, c3, c4 = st.columns([3, 1.5, 1, 0.5])
        with c1:
            st.markdown(f"**Paso {i+1}:** ¿**{fname} ≤ {th:.2f}**?")
        with c2:
            ans_color = "#27ae60" if go_left else "#e74c3c"
            st.markdown(f"`{decision_str}` → **{'Sí ✅' if go_left else 'No ❌'}**")
        with c3:
            st.markdown(f"👥 {n_samples} casos")
        with c4:
            if i < len(tree_path) - 2:
                arrow_dir = "⬅️" if go_left else "➡️"
                st.markdown(arrow_dir)
            else:
                leaf_val = tree.value[next_id][0][0]
                st.markdown("🏁")
    
    # Último nodo (hoja)
    leaf_id = tree_path[-1]
    leaf_val, n_samp = tree.value[leaf_id][0][0], tree.n_node_samples[leaf_id]
    st.success(f"""
    ### 🍃 Hoja alcanzada (Predicción final)
    - **Valor predicho:** {leaf_val:.4f} ≈ **{leaf_val:.2f}** unidades
    - **Muestras en esta hoja:** {n_samp} (de {len(df)} registros totales)
    - **Porcentaje de datos:** {n_samp/len(df)*100:.1f}% del dataset
    """)
    
    # Mostrar contenido de la hoja acorde a los datos ingresados
    st.markdown("### 📊 Contenido de la hoja según tus datos")
    
    # Buscar todas las muestras de entrenamiento que caen en esta hoja
    tree_for_predict = dt.tree_
    def get_leaf_for_sample(sample):
        node = 0
        while tree_for_predict.children_left[node] != tree_for_predict.children_right[node]:
            idx = tree_for_predict.feature[node]
            th = tree_for_predict.threshold[node]
            node = tree_for_predict.children_left[node] if sample[idx] <= th else tree_for_predict.children_right[node]
        return node
    
    # Encontrar las muestras en training que caen en la misma hoja
    X_train_array = X_train.values
    y_train_array = y_train.values
    samples_in_leaf = []
    for i in range(len(X_train_array)):
        if get_leaf_for_sample(X_train_array[i]) == leaf_id:
            samples_in_leaf.append((X_train_array[i], y_train_array[i]))
    
    if samples_in_leaf:
        leaf_df = pd.DataFrame([s[0] for s in samples_in_leaf[:10]], columns=['TV', 'Radio', 'Newspaper'])
        leaf_df['Sales'] = [s[1] for s in samples_in_leaf[:10]]
        st.markdown(f"**{len(samples_in_leaf)} muestras de entrenamiento** caen en esta misma hoja. Mostrando las primeras 10:")
        st.dataframe(leaf_df.style.format({
            'TV': '${:.1f}', 'Radio': '${:.1f}', 'Newspaper': '${:.1f}', 'Sales': '{:.2f}'
        }), width='stretch')
        
        # Mostrar estadísticas de la hoja
        leaf_sales = [s[1] for s in samples_in_leaf]
        st.markdown(f"""
        **Estadísticas de la hoja:**
        - **Ventas mínimas:** {min(leaf_sales):.2f}
        - **Ventas máximas:** {max(leaf_sales):.2f}
        - **Ventas promedio:** {np.mean(leaf_sales):.2f} ← **Este es el valor que predice el árbol**
        - **TV promedio en hoja:** ${np.mean([s[0][0] for s in samples_in_leaf]):.1f}
        - **Radio promedio en hoja:** ${np.mean([s[0][1] for s in samples_in_leaf]):.1f}
        - **Newspaper promedio en hoja:** ${np.mean([s[0][2] for s in samples_in_leaf]):.1f}
        """)
    else:
        st.info("No hay muestras exactas en esta hoja (el árbol generaliza).")
    
    inversion_total = tv + radio + newspaper
    st.metric("💰 Inversión total", f"${inversion_total}")
    
    st.markdown("---")
    st.markdown("### 💡 Recomendaciones")
    
    recomendaciones = []
    if tv > 200:
        recomendaciones.append("✅ **Alta inversión en TV:** Excelente estrategia. TV tiene mayor retorno por dólar.")
    elif tv < 50:
        recomendaciones.append("💡 **Baja inversión en TV:** Considera aumentarla. TV tiene el mayor impacto en ventas.")
    
    if radio > 40:
        recomendaciones.append("✅ **Buena inversión en Radio:** Complementa bien a la TV.")
    
    if newspaper > 80 and tv < 100:
        recomendaciones.append("⚠️ **Newspaper alto vs TV bajo:** La TV tiene 10x más impacto que el periódico. Reasigna presupuesto.")
    
    if tv > 150 and radio > 30:
        recomendaciones.append("⭐ **Campaña balanceada:** TV + Radio es la combinación más efectiva.")
    
    if not recomendaciones:
        recomendaciones.append("💡 **Escenario moderado:** Distribución equilibrada de la inversión.")
    
    for rec in recomendaciones:
        st.markdown(rec)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Desarrollado para **Machine Learning para Negocios** | Maestría en Economía - Inteligencia Artificial Aplicada")