# ML Predictor - Análisis de Campañas Publicitarias

## 📊 Aplicación Web con Streamlit para Machine Learning en Negocios

### Descripción
Aplicación interactiva para análisis predictivo de ventas en campañas publicitarias utilizando **Regresión Lineal Múltiple** y **Árboles de Decisión**. Desarrollada para la materia de **Machine Learning para Negocios** de la Maestría en Economía con mención en Inteligencia Artificial Aplicada.

### Dataset
El dataset `advertising.csv` contiene datos de inversión publicitaria en 3 medios (TV, Radio, Newspaper) y las ventas generadas (Sales).

### Funcionalidades
- **Carga de datos**: Usar dataset precargado o subir archivo CSV propio
- **EDA (Análisis Exploratorio)**: Heatmap de correlación, scatter plots, boxplots, detección de outliers
- **Regresión Lineal Múltiple**: Coeficientes, ecuación del modelo, métricas R²/MAE/RMSE
- **Árbol de Decisión**: Hiperparámetros ajustables, importancia de variables, visualización del árbol
- **Comparación de Modelos**: Tabla comparativa, gráficos reales vs predichos
- **Predicciones**: Sliders interactivos para simular escenarios de inversión

### Requisitos
- Python 3.8+
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

### Instalación y Ejecución

```bash
# Clonar el repositorio
git clone https://github.com/FidelRomeroL/advertising.git
cd advertising

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la app
streamlit run streamlit_app.py
```

### Despliegue en Streamlit Cloud
1. Sube el repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio y selecciona `streamlit_app.py`
4. ¡App desplegada! 🚀

### Estructura del Proyecto
```
app_ml/
├── streamlit_app.py      # Aplicación principal
├── advertising.csv       # Dataset de ejemplo
├── requirements.txt      # Dependencias
├── .gitignore           # Archivos ignorados
└── README.md            # Documentación
```

### Autor
Desarrollado para la Maestría en Economía con mención en Inteligencia Artificial Aplicada.