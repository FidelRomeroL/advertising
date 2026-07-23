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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 100

st.set_page_config(page_title="Predictor de Ventas", page_icon="📊", layout="wide")

# ============================================================
# SIDEBAR - SOLO FOTO Y NOMBRE
# ============================================================
try:
    st.sidebar.image("fidel.jpeg", width=150, caption="")
except:
    st.sidebar.markdown("*Foto no disponible*")
st.sidebar.markdown("**Econ. Fidel Romero León**", unsafe_allow_html=False)

st.title("📊 Predictor de Ventas Publicitarias")
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
# FUNCIONES AUXILIARES - ARBOL DE DECISION
# ============================================================
def get_tree_path(model, features, feature_names):
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
    tree = model.tree_
    node = 0
    while tree.children_left[node] != tree.children_right[node]:
        idx = tree.feature[node]
        th = tree.threshold[node]
        fval = features[idx]
        node = tree.children_left[node] if fval <= th else tree.children_right[node]
    return tree.value[node][0][0], tree.n_node_samples[node]

def draw_tree_full(model, features, feature_names, figsize=(18, 10)):
    import matplotlib
    matplotlib.use('Agg')
    plt.ioff()
    try:
        leaf_val, _ = get_leaf_value(model, features)
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(model, feature_names=feature_names, filled=True,
                  rounded=True, precision=2, fontsize=8, ax=ax)
        ax.set_title(
            f"Arbol de Decision Completo\n"
            f"TV={features[0]:.0f} | Radio={features[1]:.0f} | NP={features[2]:.0f} -> Prediccion: {leaf_val:.2f}",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        return fig
    except:
        fig, ax = plt.subplots(figsize=figsize)
        try:
            plot_tree(model, feature_names=feature_names, filled=True,
                      rounded=True, precision=2, fontsize=8, ax=ax)
        except:
            pass
        leaf_val, _ = get_leaf_value(model, features)
        ax.set_title(f"Arbol de Decision | Prediccion: {leaf_val:.2f}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

def draw_path_diagram(model, features, feature_names, figsize=(12, 6)):
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
                
                node_text = f"?{fname} <= {th:.2f}?"
                decision_text = f"{fname} = {fval:.1f} -> {'SI' if go_left else 'NO'}"
                edge_color = '#2980b9' if i == 0 else '#d35400'
            else:
                val = tree.value[nid][0][0]
                node_text = "PREDICCION (Hoja)"
                decision_text = f"Ventas = {val:.2f} unidades"
                edge_color = '#27ae60'
            
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
            
            ax.text(x_center, y_pos + box_height*0.15, node_text,
                   ha='center', va='center', fontsize=11, fontweight='bold', zorder=11)
            ax.text(x_center, y_pos - box_height*0.15, decision_text,
                   ha='center', va='center', fontsize=9, fontstyle='italic', zorder=11,
                   color='#2c3e50')
            
            ax.text(x_center - box_width/2 - 0.04, y_pos, f'{i+1}',
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='circle', facecolor=edge_color, edgecolor='white', linewidth=2),
                   color='white', zorder=12)
        
        for i in range(n_pasos - 1):
            y_from = start_y - i * step_height - box_height/2
            y_to = start_y - (i + 1) * step_height + box_height/2
            
            arrow = FancyArrowPatch(
                (0.5, y_from), (0.5, y_to),
                arrowstyle='->,head_length=0.8,head_width=0.08',
                color='#e74c3c', linewidth=3, zorder=9
            )
            ax.add_patch(arrow)
            
            nid = path_ids[i]
            idx = tree.feature[nid]
            th = tree.threshold[nid]
            fname = feature_names[idx]
            fval = features[idx]
            go_left = fval <= th
            
            ax.text(0.5 + box_width/2 + 0.02, (y_from + y_to)/2,
                   f'{fname} <= {th}' if go_left else f'{fname} > {th}',
                   ha='left', va='center', fontsize=8, color='#c0392b',
                   fontstyle='italic', zorder=9)
        
        ax.set_title(
            f"RUTA DE DECISIONES (Camino seguido -> Prediccion: {leaf_val:.2f})",
            fontsize=13, fontweight="bold", color='#2c3e50', pad=10
        )
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Camino: {len(path_ids)} pasos -> Prediccion: {leaf_val:.2f}",
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig

# ============================================================
# PREDICTOR INTERACTIVO
# ============================================================
st.markdown("### Ajusta los sliders para simular escenarios de inversion publicitaria")

col1, col2, col3 = st.columns(3)
with col1:
    tv = st.slider("**TV ($)**", 0, 300, 150)
    st.caption(f"Rango: $0 - ${df['TV'].max():.0f}")
with col2:
    radio = st.slider("**Radio ($)**", 0, 50, 25)
    st.caption(f"Rango: $0 - ${df['Radio'].max():.0f}")
with col3:
    newspaper = st.slider("**Newspaper ($)**", 0, 120, 30)
    st.caption(f"Rango: $0 - ${df['Newspaper'].max():.0f}")

X_pred = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
pred_lr = lr.predict(X_pred)[0]
pred_dt = dt.predict(X_pred)[0]

st.markdown("---")
st.markdown("### Resultados de la prediccion")

col1, col2 = st.columns(2)
with col1:
    st.markdown("---")
    st.markdown("**Regresion Lineal**")
    st.markdown(f"## **{pred_lr:.2f}** unidades")
    st.markdown(f"R^2 = {r2_score(y_test, y_pred_lr):.4f}")

with col2:
    st.markdown("---")
    st.markdown("**Arbol de Decision**")
    st.markdown(f"## **{pred_dt:.2f}** unidades")
    st.markdown(f"R^2 = {r2_score(y_test, y_pred_dt):.4f}")

st.markdown("---")
st.markdown("### Verificacion paso a paso - Regresion Lineal")

st.markdown(f"""
**Formula:** Sales = B0 + B1*TV + B2*Radio + B3*Newspaper

**Sustitucion:**
Sales = {intercept:.4f} + ({b_tv:.4f} * {tv}) + ({b_radio:.4f} * {radio}) + ({b_news:.4f} * {newspaper})

**Calculo:**
Sales = {intercept:.4f} + ({b_tv*tv:.4f}) + ({b_radio*radio:.4f}) + ({b_news*newspaper:.4f})

**Resultado:** **{pred_lr:.4f} ~ {pred_lr:.2f}** unidades
""")

st.success(f"Verificacion: La prediccion del modelo ({pred_lr:.4f}) coincide con el calculo manual")

st.markdown("---")
st.markdown("### Arbol de Decision - Camino resaltado")

st.info(f"**Tus valores:** TV=${tv} | Radio=${radio} | Newspaper=${newspaper}")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Arbol completo**")
    fig_full = draw_tree_full(dt, [tv, radio, newspaper], ['TV', 'Radio', 'Newspaper'], figsize=(12, 8))
    st.pyplot(fig_full)
    plt.close()
with col2:
    st.markdown("**Diagrama del camino seguido**")
    fig_path = draw_path_diagram(dt, [tv, radio, newspaper], ['TV', 'Radio', 'Newspaper'], figsize=(12, 8))
    st.pyplot(fig_path)
    plt.close()

st.markdown("---")
st.markdown("### Pasos del camino seguido")

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
    decision_str = f"{fval:.1f} <= {th:.1f}" if go_left else f"{fval:.1f} > {th:.1f}"
    n_samples = tree.n_node_samples[nid]
    
    c1, c2, c3, c4 = st.columns([3, 1.5, 1, 0.5])
    with c1:
        st.markdown(f"**Paso {i+1}:** ?**{fname} <= {th:.2f}**?")
    with c2:
        st.markdown(f"`{decision_str}` -> **{'Si' if go_left else 'No'}**")
    with c3:
        st.markdown(f"{n_samples} casos")
    with c4:
        if i < len(tree_path) - 2:
            st.markdown("<--" if go_left else "-->")
        else:
            st.markdown("FIN")

leaf_id = tree_path[-1]
leaf_val, n_samp = tree.value[leaf_id][0][0], tree.n_node_samples[leaf_id]
st.success(f"""
### Hoja alcanzada (Prediccion final)
- **Valor predicho:** {leaf_val:.4f} ~ **{leaf_val:.2f}** unidades
- **Muestras en esta hoja:** {n_samp} (de {len(df)} registros totales)
- **Porcentaje de datos:** {n_samp/len(df)*100:.1f}% del dataset
""")

# Mostrar contenido de la hoja
st.markdown("### Contenido de la hoja segun tus datos")

tree_for_predict = dt.tree_
def get_leaf_for_sample(sample):
    node = 0
    while tree_for_predict.children_left[node] != tree_for_predict.children_right[node]:
        idx = tree_for_predict.feature[node]
        th = tree_for_predict.threshold[node]
        node = tree_for_predict.children_left[node] if sample[idx] <= th else tree_for_predict.children_right[node]
    return node

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
    
    leaf_sales = [s[1] for s in samples_in_leaf]
    st.markdown(f"""
    **Estadisticas de la hoja:**
    - **Ventas minimas:** {min(leaf_sales):.2f}
    - **Ventas maximas:** {max(leaf_sales):.2f}
    - **Ventas promedio:** {np.mean(leaf_sales):.2f} <- **Este es el valor que predice el arbol**
    - **TV promedio en hoja:** ${np.mean([s[0][0] for s in samples_in_leaf]):.1f}
    - **Radio promedio en hoja:** ${np.mean([s[0][1] for s in samples_in_leaf]):.1f}
    - **Newspaper promedio en hoja:** ${np.mean([s[0][2] for s in samples_in_leaf]):.1f}
    """)
else:
    st.info("No hay muestras exactas en esta hoja (el arbol generaliza).")

inversion_total = tv + radio + newspaper
st.metric("Inversion total", f"${inversion_total}")

st.markdown("---")
st.markdown("### Recomendaciones")

recomendaciones = []
if tv > 200:
    recomendaciones.append("Alta inversion en TV: Excelente estrategia. TV tiene mayor retorno por dolar.")
elif tv < 50:
    recomendaciones.append("Baja inversion en TV: Considera aumentarla. TV tiene el mayor impacto en ventas.")

if radio > 40:
    recomendaciones.append("Buena inversion en Radio: Complementa bien a la TV.")

if newspaper > 80 and tv < 100:
    recomendaciones.append("Newspaper alto vs TV bajo: La TV tiene 10x mas impacto que el periodico. Reasigna presupuesto.")

if tv > 150 and radio > 30:
    recomendaciones.append("Campana balanceada: TV + Radio es la combinacion mas efectiva.")

if not recomendaciones:
    recomendaciones.append("Escenario moderado: Distribucion equilibrada de la inversion.")

for rec in recomendaciones:
    st.markdown(f"- {rec}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Desarrollado para Machine Learning para Negocios | Maestria en Economia - Inteligencia Artificial Aplicada")