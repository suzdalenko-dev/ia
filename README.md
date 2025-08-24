# 📚 Calendario IA 12 Meses + Predicción de Compras (Versión Detallada)

Este plan está **explicado paso a paso**, con *qué es*, *para qué sirve*, *cómo estudiarlo*, *pitfalls*, *entregables* y *recursos* en cada etapa.  
Perfil objetivo: dev full‑stack (**HTML/JS/PHP/Java/Python/Kotlin/Android/SQL/SQLite/Laravel/Symfony/Django/Vue**).

> **Resultado**: portafolio de 10+ proyectos, APIs listas para producción, y un **motor de compras** (forecast → cantidad a pedir) con métricas de negocio.

---

## Cómo usar este plan
- **Estructura de aprendizaje por tema**: *Qué es → Para qué sirve → Cómo estudiarlo (pasos) → Pitfalls → Entregables → Recursos.*  
- **Tiempo**: 6–10 h/semana. Cada semana tiene objetivos concretos.  
- **Repositorio**: usa esta estructura base:
```
project/
  data/            # raw/processed/interim/external
  notebooks/
  src/
  models/
  api/
  reports/
  docker/
  .env.example
  requirements.txt / pyproject.toml
  Makefile
```
- **Workflow**: (1) Leer, (2) Reproducir, (3) Re-implementar sin mirar, (4) Aplicar a tus datos, (5) Documentar.

---

## Mes 1 — Fundamentos prácticos

### Semana 1 — Python científico y entorno
**Qué es**: base de Python para datos (entornos, paquetes, tipado, testing).  
**Para qué sirve**: reproducibilidad, calidad, facilidad para desplegar.  
**Cómo estudiarlo**:
1) Crea entorno (`venv`/`conda`/`poetry`) y *Makefile* con tareas (`make setup`, `make test`).  
2) Domina `numpy` (arrays, broadcasting) y `pandas` (DataFrames, `merge`, `groupby`, `pivot`).  
3) Visualiza con `matplotlib` (líneas, barras, boxplots) y `plotly` (opcional).
**Pitfalls**: mezclar versiones de Python; `SettingWithCopy` en pandas; no fijar semillas.  
**Entregables**: notebook con 10 operaciones `pandas` típicas + script `src/etl.py`.  
**Recursos**: Documentación oficial de Python, Numpy y Pandas; “Effective Pandas” (Tom Augspurger).

### Semana 2 — Álgebra/Estadística aplicada
**Qué es**: conceptos mínimos (vectores, matrices, derivadas, media, varianza, percentiles).  
**Para qué sirve**: entender *loss/gradientes* y cómo evaluar modelos.  
**Cómo estudiarlo**: implementa a mano media/varianza; calcula z‑scores; deriva MSE respecto a w.  
**Pitfalls**: confundir población vs muestra (usa `ddof=1` para std muestral).  
**Entregables**: notebook con medidas de tendencia/dispersion + gráfico de distribución.  
**Recursos**: “Mathematics for ML” (capítulos intro); Khan Academy (probabilidad).

### Semana 3 — EDA y limpieza
**Qué es**: *Exploratory Data Analysis* para conocer datos antes de modelar.  
**Para qué sirve**: descubrir *outliers*, *missing*, *leakage*, estacionalidad.  
**Cómo**: perfiles (`describe`, `info`), IQR y Z‑score para outliers, imputación (media/mediana/forward‑fill).  
**Pitfalls**: imputar objetivos; eliminar filas sin entender el negocio.  
**Entregables**: reporte EDA (notebook) + checklist de calidad de datos.  
**Recursos**: “Practical Statistics for Data Scientists”, Pandas Profiling (ydata-profiling).

### Semana 4 — Dashboard inicial
**Qué es**: panel con KPIs y gráficos.  
**Para qué sirve**: comunicar hallazgos y validar con stakeholders.  
**Cómo**: Streamlit con tres vistas (resumen, tendencias, detalle por categoría).  
**Entregables**: app Streamlit + README con decisiones.  
**Recursos**: docs de Streamlit; “Storytelling with Data” (Cole Nussbaumer).

---

## Mes 2 — ML clásico I

### Semana 1 — Tareas y *pipelines*
**Qué es**: ML supervisado (regresión/clasificación) y no supervisado (clustering). *Pipeline* = preprocesamiento + modelo.  
**Para qué**: evitar *data leakage* y tener procesos repetibles.  
**Cómo**: `sklearn.Pipeline` con escalado (`StandardScaler`) + modelo; *split* `train/test` temporal si hay fechas.  
**Pitfalls**: escalar con todo el dataset; barajar datos temporales.  
**Entregables**: notebook con pipeline y validación inicial.  
**Recursos**: docs scikit‑learn (User Guide → Pipeline & Model Evaluation).

### Semana 2 — Regresión
**Qué es**: predecir valores numéricos (lineal, ridge, lasso).  
**Para qué**: precios, tiempos, demanda agregada.  
**Cómo**: evalúa con **MAE**, **RMSE**, **MAPE**; compara L1 vs L2; curvas de aprendizaje.  
**Pitfalls**: multicolinealidad; usar R² en series temporales puras.  
**Entregables**: benchmark de 3 modelos + análisis de residuos.  
**Recursos**: Géron “Hands‑On ML”, capítulos de regresión.

### Semana 3 — Clasificación
**Qué es**: predecir clases (logística, árboles, random forest).  
**Para qué**: fraude, churn, spam.  
**Cómo**: métricas **Precision/Recall/F1**, **ROC‑AUC** (clases balanceadas) y **PR‑AUC** (desbalance).  
**Pitfalls**: accuracy engañoso; *threshold tuning* ignorado.  
**Entregables**: matriz de confusión + *report* con umbral óptimo.  
**Recursos**: “Introduction to Statistical Learning” (ISL).

### Semana 4 — Clustering y reducción
**Qué es**: **k‑means**, **DBSCAN**; **PCA** para reducir dimensiones.  
**Para qué**: segmentar clientes, detectar outliers, acelerar modelos.  
**Cómo**: elige k con codo/silhouette; PCA: varianza explicada.  
**Pitfalls**: escalar antes de k‑means y PCA; interpretar PCA como “features reales”.  
**Entregables**: segmentos con perfiles + visualización 2D PCA.  
**Recursos**: User Guide scikit‑learn (Clustering, Decomposition).

---

## Mes 3 — ML clásico II + API

### Semana 1 — Feature Engineering
**Qué es**: crear variables útiles (lag, medias móviles, *one‑hot*, *target encoding*).  
**Para qué**: mejorar señal del modelo.  
**Cómo**: *lags* y *rolling* con `pandas`; `CategoricalEncoder`/`TargetEncoder`.  
**Pitfalls**: fuga por *target encoding* mal validado; usar *future info*.  
**Entregables**: script `src/features.py` + pruebas unitarias.  

### Semana 2 — Validación e *Hyper‑tuning*
**Qué es**: **K‑Fold**, **TimeSeriesSplit**; **Grid/Random/Bayes Search**.  
**Para qué**: estimar performance honesta y elegir hiperparámetros.  
**Cómo**: `TimeSeriesSplit(n_splits=5)`; `RandomizedSearchCV`.  
**Pitfalls**: mezclar validación aleatoria con series temporales.  
**Entregables**: informe de *tuning* + importancias (Permutation/SHAP).  

### Semana 3 — Proyecto: Spam
**Qué es**: clasificador textual con TF‑IDF + LogReg/LinearSVM.  
**Para qué**: detectar spam/soporte automatizado.  
**Cómo**: `TfidfVectorizer` (word/char n‑grams), normaliza y regulariza.  
**Entregables**: notebook + `src/train.py` + `models/`.  

### Semana 4 — API y Docker
**Qué es**: servir modelos vía **FastAPI/Django REST**; empaquetar con Docker.  
**Para qué**: consumo desde Vue/Android/otros servicios.  
**Cómo**: endpoint `/predict`, validación con Pydantic, healthcheck; `Dockerfile` multi‑stage.  
**Pitfalls**: no fijar versiones; falta de *timeout* y *rate‑limit*.  
**Entregables**: API funcional + imagen Docker + README.

---

## Mes 4 — Deep Learning básico

### Semana 1 — Redes y entrenamiento
**Qué es**: MLP (capas densas), **backprop**, funciones de activación (ReLU, GELU), pérdidas (MSE, CE), optimizadores (SGD, Adam).  
**Para qué**: aprender representaciones no lineales.  
**Cómo**: Keras: `Model.fit` con `callbacks` (EarlyStopping, ReduceLROnPlateau).  
**Pitfalls**: overfitting → usa dropout/L2/early stopping.  
**Entregables**: MLP MNIST con >98% accuracy validación.

### Semana 2 — Regularización y *Augmentation*
**Qué es**: Dropout, L1/L2, BatchNorm; *augmentation* (ruido, flips).  
**Para qué**: generalización.  
**Cómo**: compara runs con/ sin regularización (MLflow).  
**Entregables**: informe comparativo + curvas loss/accuracy.

### Semana 3 — PyTorch vs Keras
**Qué es**: dos *frameworks* líderes.  
**Para qué**: elegir según preferencia/ecosistema.  
**Cómo**: re‑implementa el mismo MLP en ambos.  
**Entregables**: dos scripts de entrenamiento equivalentes.

### Semana 4 — *Experiment tracking*
**Qué es**: registrar hiperparámetros/artefactos (**MLflow**).  
**Para qué**: reproducibilidad, auditoría.  
**Cómo**: `mlflow.sklearn.log_model`, `mlflow.log_params/metrics`.  
**Entregables**: 5+ ejecuciones registradas.

---

## Mes 5 — Visión por Computadora (CNN)

### Semana 1 — Convoluciones y *Transfer Learning*
**Qué es**: CNN (conv, stride, padding, pooling); **transfer learning** con ResNet/MobileNet.  
**Para qué**: clasificar/detectar con pocos datos.  
**Cómo**: congelar capas, re‑entrenar *head*.  
**Pitfalls**: *overfitting* sin augmentation; LR muy alta.  
**Entregables**: clasificador con F1>0.9 en tu dataset.

### Semana 2 — Proyecto visión
**Qué es**: caso aplicado (defectos de producto o clasificación SKU).  
**Para qué**: calidad y automatización.  
**Cómo**: dataset balanceado, *stratified split* por clase.  
**Entregables**: modelo + informe de errores (top‑confusions).

### Semana 3 — Exportación
**Qué es**: **TensorFlow Lite** / **ONNX** / cuantización.  
**Para qué**: inferencia en móvil/edge.  
**Cómo**: exporta y mide latencia/precisión.  
**Entregables**: archivo `.tflite`/`.onnx` + pruebas.

### Semana 4 — App Android
**Qué es**: inferencia on‑device (Kotlin).  
**Para qué**: experiencia offline, rápida y privada.  
**Entregables**: demo cámara → predicción.

---

## 🔴 Meses 6–7 — Predicción de Compras e Inventario (EN DETALLE)

### Mes 6 — Forecast de demanda (Parte I)
**Qué es**: **series temporales** con nivel/tendencia/estacionalidad y ruido.  
**Para qué**: anticipar demanda por SKU para planificar compras/producción.  
**Cómo estudiarlo**:
1) **Re‑muestreo** a diario/semanal; completar ceros; *cut‑over* por alta/baja de SKUs.  
2) Baselines: *naïve* (mañana = hoy), *seasonal‑naïve* (próx. lunes = lunes pasado).  
3) **ETS (Holt‑Winters)**: captura tendencia/estacionalidad aditiva/multiplicativa.  
4) **ARIMA/SARIMA/SARIMAX**: modela autocorrelación; `SARIMAX(endog, exog=calendar/promos/precio)`.  
5) **Validación temporal** (*walk‑forward*): entrena en [t0,t1], predice [t1,t2]; desliza ventana.  
6) Métricas: **WAPE** (= Σ|y−ŷ| / Σy), **sMAPE**, **MASE** (robusta a escala).  
**Pitfalls**: mezclar datos de futuro; no separar por SKU; ignorar días cero válidos.  
**Entregables**: *pipeline* por SKU con backtesting + tablero Vue (serie real vs predicción y error).  
**Recursos**: `statsmodels` (ETS/ARIMA), “Forecasting: Principles and Practice” (Hyndman).

#### Tabla rápida de Z (nivel de servicio → Z)
| Servicio | Z aproximado |
|---|---|
| 90% | 1.28 |
| 95% | 1.645 |
| 97.5% | 1.96 |
| 99% | 2.33 |

### Mes 7 — De forecast a órdenes de compra (Parte II)
**Qué es**: traducir demanda esperada a **cuándo** y **cuánto** comprar.  
**Para qué**: minimizar roturas y capital inmovilizado.  
**Cómo estudiarlo**:
1) **Demanda intermitente** (muchos ceros): **Croston/SBA/TSB** (frecuencia × tamaño).  
2) **Lead time**: media y desviación por proveedor (usa `suppliers`); incluye festivos.  
3) **Safety Stock (SS)** y **ROP**:  
   - Fijo: `SS = Z * σ_d * sqrt(L)`; `ROP = μ_d * L + SS`.  
   - Variable: `SS = Z * sqrt( μ_L σ_d² + μ_d² σ_L² )`.  
4) **Políticas**: (Q,R) cantidad fija/pedido cuando stock < ROP; (s,S) revisar cada *s* días y llevar a *S*.  
5) **EOQ**: `sqrt(2 K D / h)`; ajusta por **MOQ** y **price breaks**.  
6) **Priorización ABC/XYZ**: A por valor, X por estabilidad → nivel de servicio alto primero.  
**Pitfalls**: usar media global para todos; no considerar **on_order/backorders**; ignorar variabilidad del lead time.  
**Entregables**: microservicio “**Purchase Recommender**”: endpoint `/purchase_suggestions?sku=...` con `next_order_date`, `qty`, `reason`.  
**Recursos**: libros de *Inventory Control*, notas de Silver/Zaik, material de Hyndman sobre intermitentes.

**Glosario compra/inventario**:  
- **WAPE/sMAPE/MASE**: errores relativos para comparar SKUs.  
- **Bullwhip effect**: variabilidad amplificada río arriba; mitígalo con ventanas y *smoothing*.  
- **Fill rate**: % de demanda atendida sin stockout.  
- **Coverage**: días de inventario disponibles.

---

## Mes 8 — NLP clásico

### Semana 1 — Preprocesamiento
**Qué es**: normalizar (minúsculas), tokenizar, *stemming/lemmatización*.  
**Para qué**: preparar texto para modelos.  
**Cómo**: `nltk/spacy`; limpia HTML, emojis, stopwords (con cuidado).  
**Pitfalls**: eliminar negaciones (“no”); romper entidades.  
**Entregables**: script `src/nlp_clean.py`.

### Semana 2 — Vectorización
**Qué es**: representar texto como números. **TF‑IDF**, *n‑grams*, **Word2Vec/GloVe**.  
**Para qué**: alimentar clasificadores o regresores.  
**Cómo**: `TfidfVectorizer(ngram_range=(1,2))`; compara con embeddings preentrenados.  
**Entregables**: notebook comparativo.

### Semana 3 — Clasificación de texto
**Qué es**: intents/sentimientos.  
**Cómo**: LinearSVC/LogReg con TF‑IDF; evalúa con F1 macro.  
**Pitfalls**: *data leakage* al construir vocabulario.  
**Entregables**: modelo + reporte.

### Semana 4 — API
**Entregables**: endpoint `/analyze` (Django/Laravel), tests y CI básica.

---

## Mes 9 — Transformers y RAG

### Semana 1 — Transformers
**Qué es**: atención, *positional encoding*, modelos **BERT/GPT**.  
**Para qué**: SOTA en NLP (QA, clasificación, resumen).  
**Cómo**: `transformers` (Hugging Face): *tokenizer* → modelo → *pipeline*.  

### Semana 2 — LoRA y Embeddings
**Qué es**: **LoRA** = *fine‑tuning ligero*; **embeddings** = vectores semánticos.  
**Para qué**: adaptar un LLM barato; búsqueda semántica.  
**Cómo**: *adapters* con PEFT; FAISS para indexar y buscar.  

### Semana 3 — RAG
**Qué es**: **Retrieval‑Augmented Generation**.  
**Para qué**: respuestas con tus documentos privados.  
**Cómo**: *chunking* (512–1024 tokens), top‑k, **re‑ranking**, *prompt templates*.  
**Pitfalls**: *chunking* muy grande; no manejar fuentes.  
**Entregables**: chatbot con citas a documentos.

### Semana 4 — Producto
**Entregables**: servicio RAG (backend) + UI Vue con historial y feedback.

---

## Mes 10 — Visión avanzada

### Semana 1 — Detección/Segmentación
**Qué es**: **YOLO/Faster R‑CNN/Mask R‑CNN**; métricas **IoU**/**mAP**.  
**Para qué**: localizar objetos, contar, segmentar.  
**Cómo**: anotar datos (Label Studio/Roboflow); *train* y evaluar.  
**Pitfalls**: *class imbalance*; *non‑max suppression* mal ajustado.  

### Semana 2 — OCR
**Qué es**: **Tesseract** (+ preprocesado OpenCV); post‑proceso con reglas/LLM.  
**Para qué**: digitalizar facturas/albaranes.  
**Entregables**: microservicio OCR con colas (RQ/Celery).

### Semana 3–4 — Proyecto completo
**Entregables**: pipeline OCR + validaciones + dashboard operativo (errores por campo, tasa de corrección).

---

## Mes 11 — IA generativa

### Semana 1 — LLMs en producción
**Qué es**: consumo por API o modelos *open‑source*.  
**Para qué**: asistentes, resúmenes, extracción, *function calling*.  
**Cómo**: *prompting* (instrucciones, ejemplos, restricciones), *guardrails*.  

### Semana 2 — Imágenes
**Qué es**: **Stable Diffusion** (text‑to‑image).  
**Para qué**: marketing, prototipos, variantes de imágenes.  
**Pitfalls**: costes GPU; política de contenido.  

### Semana 3 — Audio
**Qué es**: **Whisper** (STT), TTS.  
**Para qué**: dictado, bots de voz.  
**Entregables**: pipeline audio → texto → acción.

### Semana 4 — App generativa
**Entregables**: app full‑stack (Vue + backend) combinando texto/imagen/audio.

---

## Mes 12 — MLOps y producción

### Semana 1 — Versionado y datos
**Qué es**: **DVC/MLflow** para versiones de datos/modelos.  
**Para qué**: reproducibilidad.  
**Entregables**: *pipeline* con caché y artefactos.

### Semana 2 — CI/CD y tests
**Qué es**: integración y despliegue continuo; **Great Expectations** para tests de datos.  
**Para qué**: prevenir degradaciones.  
**Entregables**: workflow CI (lint/test/build/push).

### Semana 3 — Observabilidad y *drift*
**Qué es**: **drift** de *features* (covariate) o de etiquetas (concept).  
**Para qué**: detectar cuándo re‑entrenar.  
**Entregables**: alertas + tablero métricas en producción.

### Semana 4 — *Runbook* y *SLA*
**Qué es**: manual de operación y objetivos (latencia, error).  
**Entregables**: deploy cloud + *runbook* + post‑mortem simulado.

---

## Apéndice A — API de Sugerencia de Compras (ejemplo)

**Endpoints**  
- `GET /health` → ok.  
- `POST /forecast` → body: `{ sku, horizon_days }` → devuelve serie pronosticada.  
- `POST /purchase_suggestions` → body: `{ sku, on_hand, on_order, backorders, lead_time_mean, lead_time_std, service_level, moq, budget }` → `{ next_order_date, recommended_qty, reason, coverage_days }`.

**Lógica**  
1) Construye demanda diaria neta.  
2) Elige modelo (ETS/SARIMA/Croston) por CV.  
3) Calcula SS/ROP y **Q** (EOQ/heurística).  
4) Aplica reglas de presupuesto y **ABC/XYZ**.  
5) Devuelve explicación de la recomendación (**reason**).

---

## Apéndice B — Checklists útiles

**Calidad de datos (antes de modelar)**  
- [ ] % de `missing` por columna < 10% (o imputado justificado)  
- [ ] Outliers explicados/recortados  
- [ ] Duplicados tratados  
- [ ] Consistencia de unidades y moneda  
- [ ] Alineación de fechas y TZ

**Forecast**  
- [ ] *Walk‑forward* implementado  
- [ ] Baselines comparados  
- [ ] Exógenas (calendario/promo/precio) evaluadas  
- [ ] Métrica negocio (**WAPE**, *fill‑rate*) incluida

**Inventario**  
- [ ] Lead time por proveedor (media y std)  
- [ ] Políticas (Q,R) o (s,S) definidas por familia  
- [ ] Tabla de Z validada con negocio  
- [ ] Alertas por *stockout* y exceso

---

## Apéndice C — Glosario relámpago (términos clave)

- **Data leakage**: usar información del futuro en entrenamiento; da métricas irreales.  
- **ROC‑AUC / PR‑AUC**: calidad de ranking de probabilidades; PR‑AUC mejor con clases desbalanceadas.  
- **PCA**: combinación lineal ortogonal que maximiza varianza; útil para visualización/compresión.  
- **MLflow**: seguimiento de experimentos y modelos.  
- **ETS/ARIMA/SARIMA/SARIMAX**: familias de modelos de series (tendencia/estacionalidad/exógenas).  
- **Croston/SBA/TSB**: métodos para demanda intermitente (estimación de intervalo y tamaño).  
- **EOQ**: calcula lote económico balanceando coste de pedido y de inventario.  
- **ABC/XYZ**: priorización por valor (ABC) y variabilidad (XYZ).  
- **FAISS**: índice para búsqueda de vectores (embeddings).  
- **RAG**: recuperación + generación, para respuestas basadas en documentos.  
- **Drift**: cambio en la distribución (covariate) o en la relación X→y (concept).  
- **Great Expectations**: framework de tests de datos.  
- **ONNX/TFLite**: formatos/entornos para ejecutar modelos portables y ligeros.

---

## Apéndice D — Recursos de estudio (curado, breve)

- **Libros**: 
  - “Hands‑On Machine Learning” (Géron) — ML/DL práctico.  
  - “Forecasting: Principles and Practice” (Hyndman & Athanasopoulos) — forecasting aplicado.  
  - “Deep Learning” (Goodfellow et al.) — fundamentos teóricos.

- **Docs oficiales**: scikit‑learn, pandas, statsmodels, PyTorch, TensorFlow, MLflow, FastAPI, Hugging Face.

- **Cursos**: 
  - Machine Learning (Andrew Ng) — fundamentos.  
  - NLP with Transformers (HF) — práctico.  
  - MLOps básicos (varios proveedores).

- **Herramientas**: Label Studio (anotación), DVC/MLflow (versionado/seguimiento), Great Expectations (calidad), Airflow (orquestación).

---

## Siguientes pasos inmediatos
1) Elige un dataset propio del negocio y ejecuta **Mes 1–2** con EDA + 1º modelo.  
2) Prepara **series por SKU** y aplica **Mes 6** (ETS/SARIMA, backtesting).  
3) Implementa **Mes 7** para convertir pronósticos en **órdenes** (API + dashboard).  
4) Documenta decisiones y resultados en el README del proyecto.

