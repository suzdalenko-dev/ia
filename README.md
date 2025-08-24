# Calendario IA 12 Meses + Predicción de Compras

**📆 Calendario de aprendizaje en IA (12 meses)**  
Pensado para dev full‑stack (**HTML/JS/PHP/Java/Python/Kotlin/Android/SQL/SQLite/Laravel/Symfony/Django/Vue**).  
Incluye un **módulo práctico de Predicción de Compras e Inventario (Meses 6–7)** con fórmulas, esquema de datos y entregables de producción.

---

## Tabla de contenido
- [Cómo usar este plan](#cómo-usar-este-plan)
- [Mes 1 — Fundamentos prácticos](#mes-1--fundamentos-prácticos)
- [Mes 2 — ML clásico I](#mes-2--ml-clásico-i)
- [Mes 3 — ML clásico II + API](#mes-3--ml-clásico-ii--api)
- [Mes 4 — Deep Learning básico](#mes-4--deep-learning-básico)
- [Mes 5 — Visión por Computadora (CNN)](#mes-5--visión-por-computadora-cnn)
- [🔴 Meses 6–7 — Predicción de Compras e Inventario](#-meses-67--predicción-de-compras-e-inventario)
  - [Mes 6 — Series temporales y forecast de demanda (Parte I)](#mes-6--series-temporales-y-forecast-de-demanda-parte-i)
  - [Mes 7 — Inventario, compras y optimización (Parte II)](#mes-7--inventario-compras-y-optimización-parte-ii)
  - [Esquema de datos sugerido](#esquema-de-datos-sugerido)
  - [Transformaciones clave](#transformaciones-clave)
  - [Stack recomendado](#stack-recomendado)
  - [Métricas de negocio](#métricas-de-negocio)
- [Mes 8 — NLP clásico](#mes-8--nlp-clásico)
- [Mes 9 — Transformers y RAG](#mes-9--transformers-y-rag)
- [Mes 10 — Visión avanzada](#mes-10--visión-avanzada)
- [Mes 11 — IA generativa](#mes-11--ia-generativa)
- [Mes 12 — MLOps y producción](#mes-12--mlops-y-producción)
- [📦 Apéndice — Plantillas y snippets útiles](#-apéndice--plantillas-y-snippets-útiles)
  - [1) Consulta SQL base (demanda diaria por SKU)](#1-consulta-sql-base-demanda-diaria-por-sku)
  - [2) Pseudocódigo Python: forecast → compra sugerida](#2-pseudocódigo-python-forecast--compra-sugerida)
  - [3) Priorización ABC/XYZ](#3-priorización-abcxyz)
  - [4) Métricas de éxito](#4-métricas-de-éxito)
- [Siguientes pasos](#siguientes-pasos)

---

## Cómo usar este plan
- Cada mes tiene **objetivos**, **tareas semanales** y **entregables**.
- Mantén un **repositorio por proyecto** y documenta con `README` + resultados.
- **Time‑boxing** sugerido: _6–10 h/semana_. Acelera si puedes.

---

## Mes 1 — Fundamentos prácticos
**Objetivo:** base matemática y stack de datos en Python.

**Semana 1**
- Repaso rápido de Python (virtualenv/poetry, tipado, testing).
- Numpy (vectores, matrices), Pandas (DataFrames, joins, groupby), Matplotlib.

**Semana 2**
- Álgebra lineal aplicada (producto punto, normas, descomposiciones a nivel intuitivo).
- Probabilidad y estadística (media/varianza, distribuciones, muestreo, intervalos).

**Semana 3**
- EDA (análisis exploratorio) sobre dataset público.
- Limpieza de datos, valores faltantes, outliers.

**Semana 4**
- **Entregable:** notebook EDA + dashboard simple (Streamlit) con KPIs.
- Publica un post corto con hallazgos.

**Entregables del mes:** 1 notebook EDA, 1 mini dashboard.

---

## Mes 2 — ML clásico I
**Objetivo:** dominar tareas y métricas base.

**Semana 1**
- Conceptos: supervisado vs no supervisado; train/valid/test; leakage.
- Pipelines y escalado de features.

**Semana 2**
- Regresión lineal/ridge/lasso; evaluación: MAE, RMSE, MAPE.

**Semana 3**
- Clasificación: logística, árboles, random forest, métricas (accuracy, ROC‑AUC, F1).

**Semana 4**
- Clustering: k‑means, DBSCAN; reducción de dimensionalidad (PCA).
- **Entregable:** benchmark de 3 modelos por tarea.

**Entregables del mes:** scripts reproducibles + reporte de métricas.

---

## Mes 3 — ML clásico II + API
**Objetivo:** producción mínima viable.

**Semana 1**
- Feature engineering: variables temporales, categóricas (one‑hot/target), leakage checks.

**Semana 2**
- Validación cruzada; búsqueda de hiperparámetros; importancia de variables.

**Semana 3**
- **Proyecto:** clasificador de spam (scikit‑learn).

**Semana 4**
- Exponer modelo: **Django REST** o **FastAPI** (Python) y alternativa **Laravel** (PHP).
- Contenerizar con Docker.

**Entregables del mes:** API funcional + imagen Docker.

---

## Mes 4 — Deep Learning básico
**Objetivo:** redes feed‑forward y práctica con Keras/PyTorch.

**Semana 1**
- Teoría: perceptrón, activaciones, pérdida, optimizadores, regularización.

**Semana 2**
- MLP para MNIST/Fashion‑MNIST. Early stopping y data augmentation simple.

**Semana 3**
- PyTorch vs Keras; loops de entrenamiento; checkpoints.

**Semana 4**
- Experimentos con MLflow; comparación de runs.

**Entregables del mes:** notebook + reporte comparativo + tracking con MLflow.

---

## Mes 5 — Visión por Computadora (CNN)
**Objetivo:** transfer learning y despliegue móvil.

**Semana 1**
- CNNs (convolución, pooling), fine‑tuning con ResNet/MobileNet.

**Semana 2**
- **Proyecto:** clasificador de imágenes (p.ej., defectos de producto).

**Semana 3**
- Exportar a TensorFlow Lite / ONNX.

**Semana 4**
- App Android (Kotlin) con inferencia on‑device (camera → predicción).

**Entregables del mes:** modelo CNN + app Android demo.

---

## 🔴 Meses 6–7 — Predicción de Compras e Inventario
Aquí respondes a: **“¿puedo predecir la necesidad y cantidad de compras futuras con mis datos de entradas (compras) y salidas (ventas/consumo)?”**  
**Sí.** Se trabaja en profundidad en estos 2 meses.

### Mes 6 — Series temporales y forecast de demanda (Parte I)
**Objetivo:** construir pronósticos por SKU a corto/mediano plazo.

**Semana 1**
- Fundamentos de series: nivel, tendencia, estacionalidad, ruido; re‑muestreo diario/semanal.
- Métricas para forecast: MAE, RMSE, **WAPE**, **sMAPE**, **MASE**.
- Validación temporal (rolling origin / walk‑forward).

**Semana 2**
- Baselines: naïve, seasonal‑naïve, media móvil.
- **Suavizado exponencial** (ETS/Holt‑Winters) para tendencia+estacionalidad.

**Semana 3**
- **ARIMA/SARIMA/SARIMAX** (con regresores externos: calendario, promos, precio).
- Selección de órdenes, diagnóstico de residuos.

**Semana 4 — Proyecto**
- Pipeline por SKU: consolidar **consumo neto** (ventas + consumo interno − devoluciones).
- Backtesting con ventanas rodantes; comparar ETS vs SARIMA.
- Dashboard **Vue** (línea de tiempo, error por SKU, mapa de calor por estacionalidad).

**Entregables del mes:** repos de datos + pipeline de forecasting + dashboard.

### Mes 7 — Inventario, compras y optimización (Parte II)
**Objetivo:** convertir demanda pronosticada en **órdenes de compra** concretas.

**Semana 1 — Demanda intermitente**
- Si hay muchos ceros: **Croston**, **SBA**, **TSB**; evaluación con sMAPE/WAPE.

**Semana 2 — Políticas de inventario**
- **Lead time** (promedio y variabilidad), **nivel de servicio** (prob. de stockouts).
- **Safety stock (SS)** y **reorder point (ROP)**.

**Lead time fijo (L) y demanda ~ N(μ_d, σ_d):**
$$
SS = Z \\cdot \\sigma_d \\sqrt{L}
$$
$$
ROP = \\mu_d \\cdot L + SS
$$

**Lead time variable (\\(\\mu_L, \\sigma_L\\)):**
$$
SS = Z \\cdot \\sqrt{\\mu_L\\,\\sigma_d^2 + \\mu_d^2\\,\\sigma_L^2}
$$

**Semana 3 — Cantidad a ordenar**
- **EOQ** (Economic Order Quantity):
$$
EOQ = \\sqrt{\\tfrac{2KD}{h}}
$$
> Donde: **K** = coste por pedido, **D** = demanda anual, **h** = coste de posesión por unidad/año.

- Reglas prácticas: **cobertura objetivo** (semanas) vs capacidad de caja/**MOQ**.
- Heurística final:
$$
Q = \\max\\big( MOQ,\\; TargetStock - OnHand + Backorders - OnOrder \\big)
$$

**Semana 4 — Sistema de recomendación de compras**
- Servicio que, por SKU, entregue: **próxima fecha de pedido** y **cantidad sugerida**.
- Considera calendario (cierres, festivos), presupuesto y **priorización ABC/XYZ**.
- Exponer API (FastAPI/Django REST o Laravel) + tarea de recalculo diario (cron).
- Monitor de métricas: roturas, cobertura, rotación, obsolescencia.

**Entregables del mes:** motor de compras + API + panel de control + documentación.

### Esquema de datos sugerido
```
purchases(date, sku_id, qty, unit_cost, supplier_id, po_id)
sales(date, sku_id, qty, price, channel)
consumption_internal(date, sku_id, qty, reason)
returns(date, sku_id, qty)
inventory_snapshot(date, sku_id, on_hand, on_order, backorders)
suppliers(supplier_id, lead_time_days_avg, lead_time_days_std, MOQ, price_breaks)
calendar(date, is_holiday, is_month_end, week_of_year, promo_flag)
```

### Transformaciones clave
- `demanda_diaria = sales.qty + consumption_internal.qty − returns.qty` (por fecha, sku).
- Unificar granularidad (rellenar ceros), manejar cortes de catálogo, outliers.

### Stack recomendado
- **Python:** pandas, statsmodels (ETS/ARIMA), scikit‑learn; opcional: Prophet.
- **SQL:** vistas materializadas para series por SKU; particiones por fecha.
- **API:** FastAPI/Django REST o Laravel; **Vue** para dashboards.
- **Jobs:** cron/Airflow; **tracking:** MLflow; **contenedores:** Docker.

### Métricas de negocio
- **WAPE** global, roturas (% días con stock=0), cobertura (días), rotación, % obsolescencia.
- **SLA** de proveedores (cumplimiento de lead time), **fill rate**.

---

## Mes 8 — NLP clásico
**Objetivo:** limpiar, vectorizar y clasificar texto.

**Semana 1**
- Preprocesamiento: normalización, tokenización, stemming/lematización.

**Semana 2**
- TF‑IDF, n‑grams, embeddings (Word2Vec/GloVe).

**Semana 3**
- Clasificador de sentimientos / intents.

**Semana 4**
- **Entregable:** API de análisis de texto (Django/Laravel) + pruebas unitarias.

---

## Mes 9 — Transformers y RAG
**Objetivo:** LLMs aplicados y búsqueda aumentada (RAG).

**Semana 1**
- Arquitectura Transformer; BERT vs GPT; tokenización subword.

**Semana 2**
- Fine‑tuning ligero / LoRA; embeddings y **búsqueda semántica** (FAISS).

**Semana 3**
- **RAG** sobre PDFs/documentos de negocio; manejo de contexto.

**Semana 4**
- **Entregable:** chatbot interno (Vue + backend) con fuentes privadas.

---

## Mes 10 — Visión avanzada
**Objetivo:** detección/segmentación + OCR.

**Semana 1**
- YOLO/Detectron conceptos; datasets con etiquetas.

**Semana 2**
- OCR (Tesseract + postproceso) y pipelines mixtos (CV clásica + DL).

**Semana 3**
- **Proyecto:** lectura de albaranes/facturas + validaciones.

**Semana 4**
- **Entregable:** microservicio OCR con colas (RQ/Celery) y dashboard.

---

## Mes 11 — IA generativa
**Objetivo:** texto, imagen y audio.

**Semana 1**
- LLMs en producción (APIs comerciales o modelos open‑source).

**Semana 2**
- Imágenes: Stable Diffusion; colas y caché de resultados.

**Semana 3**
- Audio: Whisper (STT), TTS; UX conversacional.

**Semana 4**
- **Entregable:** app full‑stack que combine generación de texto/imagen/audio.

---

## Mes 12 — MLOps y producción
**Objetivo:** fiabilidad, escalado, monitoreo.

**Semana 1**
- Versionado de datos/modelos (DVC/MLflow), artefactos, repos monorepo.

**Semana 2**
- CI/CD para modelos; tests de datos (Great Expectations) y de performance.

**Semana 3**
- Observabilidad: drift de datos, monitoreo de métricas y alertas.

**Semana 4**
- **Entregable final:** deploy cloud (AWS/GCP/Azure/DigitalOcean) + runbook.

---

## 📦 Apéndice — Plantillas y snippets útiles

### 1) Consulta SQL base (demanda diaria por SKU)
```sql
-- Ventas + consumo interno − devoluciones → demanda neta diaria
WITH s AS (
  SELECT date::date AS d, sku_id, SUM(qty) AS q FROM sales GROUP BY 1,2
), c AS (
  SELECT date::date AS d, sku_id, SUM(qty) AS q FROM consumption_internal GROUP BY 1,2
), r AS (
  SELECT date::date AS d, sku_id, SUM(qty) AS q FROM returns GROUP BY 1,2
), calendar AS (
  SELECT generate_series(
      (SELECT MIN(date) FROM sales)::date,
      (SELECT MAX(date) FROM sales)::date,
      interval '1 day'
  )::date AS d
)
SELECT cal.d, skus.sku_id,
       COALESCE(s.q,0) + COALESCE(c.q,0) - COALESCE(r.q,0) AS demand
FROM (SELECT DISTINCT sku_id FROM sales) skus
CROSS JOIN calendar cal
LEFT JOIN s ON s.d = cal.d AND s.sku_id = skus.sku_id
LEFT JOIN c ON c.d = cal.d AND c.sku_id = skus.sku_id
LEFT JOIN r ON r.d = cal.d AND r.sku_id = skus.sku_id
ORDER BY skus.sku_id, cal.d;
```

### 2) Pseudocódigo Python: forecast → compra sugerida
```python
# Input: serie diaria por SKU (pandas Series), on_hand, on_order, backorders,
# lead_time_mean (L), lead_time_std, service_level (Z), horizon_dias, MOQ

y = serie_diaria.fillna(0)

# 1) Modelo (elige ETS/ARIMA según CV)
# Ejemplo con auto_arima (pmdarima)
from pmdarima import auto_arima
model = auto_arima(y, seasonal=True, m=7, error_action="ignore", suppress_warnings=True)
forecast = model.predict(n_periods=horizon_dias)  # demanda esperada próxima ventana

mu_d = y[-90:].mean()            # demanda promedio reciente
sigma_d = y[-90:].std(ddof=1)    # desviación estándar reciente

# 2) Seguridad y punto de pedido
import math
if lead_time_std == 0:
    SS = service_level * sigma_d * math.sqrt(lead_time_mean)
else:
    SS = service_level * math.sqrt(lead_time_mean * (sigma_d**2) + (mu_d**2) * (lead_time_std**2))

ROP = mu_d * lead_time_mean + SS

# 3) Cantidad sugerida (cobertura objetivo = horizon_dias)
target_stock = forecast.sum() + SS
Q = max(MOQ, target_stock - on_hand + backorders - on_order)
```

### 3) Priorización ABC/XYZ
- **ABC** por valor anual (precio × demanda): A = top 80%, B = 15%, C = 5%.
- **XYZ** por estabilidad (coef. de variación): X = estable, Y = media, Z = volátil.
- **Políticas:** A/X con alto servicio y revisiones frecuentes; C/Z con compras puntuales.

### 4) Métricas de éxito
- **WAPE** global < 15–20% (buen inicio); **sMAPE** por SKU; **fill‑rate** > 95% en A/X.
- KPI de negocio: ↓ roturas, ↓ capital inmovilizado, ↑ rotación.

---

## Siguientes pasos
1. Reúne **12–24 meses** de datos (mínimo 9 si el catálogo rota).  
2. Ejecuta la **consulta de demanda diaria** y valida outliers.  
3. Implementa **Mes 6 Semana 4 (backtesting)**.  
4. Avanza al **Mes 7** para convertir forecast en **órdenes de compra**.

