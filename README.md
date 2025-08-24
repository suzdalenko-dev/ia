# 📚 Calendario IA 12 Meses + Predicción de Compras — **README.md**

> Plan paso a paso para convertirte en profesional de IA aplicando tu stack full‑stack (**HTML/JS/PHP/Java/Python/Kotlin/Android/SQL/SQLite/Laravel/Symfony/Django/Vue**).  
> Resultado: **10+ proyectos** en portafolio, **APIs listas para producción** y un **motor de compras** (forecast → cantidad a pedir) con métricas de negocio.

---

## 🧭 Cómo usar este plan

- **Estructura por tema**: *Qué es → Para qué sirve → Cómo estudiarlo → Pitfalls → Entregables → Recursos → Cómo lo usarás en proyectos*.
- **Tiempo**: 6–10 h/semana. Cada semana tiene objetivos concretos.
- **Repositorio base** (recomendado):
  ```text
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
- **Workflow**: (1) Leer, (2) Reproducir, (3) Re‑implementar sin mirar, (4) Aplicar a tus datos, (5) Documentar.

---

# A) Álgebra, Cálculo y Probabilidad “mínimas” para IA (Semana 1.5)

> Inserta esta **Semana 1.5** entre tus Semanas 1 y 2. Úsala también como **ficha de consulta** todo el año.

## 1) Álgebra lineal esencial

**Qué es**  
- Espacios vectoriales y **normas**: \(\ell_1,\ \ell_2,\ \ell_\infty\); normalización/escalado.  
- **Producto escalar** y **proyecciones** (descomposición ortogonal). Proyección: \(P = X(X^\top X)^{-1} X^\top\).  
- **Matrices**: rango, traza, determinante, inversa, simétricas, **PSD** (\(x^\top A x \ge 0\)).  
- **Autovalores/autovectores**: estabilidad numérica, condicionamiento.  
- **SVD**: \(X = U\,\Sigma\,V^\top\) → reducción de dimensión, compresión, **PCA**.  
- **PCA**: varianza explicada, *whitening*.

**Para qué sirve**  
- Entender la **geometría de los datos**, compresión/ruido, reducción de dimensión y **descorrelación**.
- Fundamental para **recomendadores**, **visión** (bases ortogonales), **clustering** y **explicabilidad**.

**Cómo estudiarlo**  
- Implementa **PCA** con **SVD** desde cero y compáralo con `sklearn`.  
- Juega con **escalado** (z‑score, min‑max) y mide el impacto en k‑means/svm.

**Pitfalls**  
- Invertir matrices “a pelo”. Preferir **Cholesky/QR/SVD**.  
- Aplicar PCA sin estandarizar *features* que están en escalas distintas.

**Entregables**  
- **Cheat‑sheet** A4 con identidades (proyección, traza, derivadas).  
- Notebook de **PCA por SVD** + *plot* de varianza explicada.

**Recursos rápidos**  
- “Linear Algebra Review” (CS229).  
- “Matrix Cookbook” (identidades y derivadas).

**Cómo lo usarás en proyectos**  
- **Mes 2–4**: PCA para visualización 2D de clientes (segmentación).  
- **Mes 9 (RAG)**: reducción de dimensión para *debug* de **embeddings**.  
- **Mes 10 (visión)**: comprensión de **timm/Conv** y compresión con SVD/ONNX.

---

## 2) Cálculo (vectorial y matricial)

**Qué es**  
- **Gradiente**, **Jacobiano**, **Hessiano**; **regla de la cadena**.  
- **Cálculo matricial** útil:
  - \(\frac{\partial}{\partial X}\,\mathrm{tr}(AX) = A^\top\)  
  - \(\frac{\partial}{\partial X}\,\lVert AX-b\rVert_2^2 = 2A^\top(AX-b)\)  
  - \(\frac{\partial}{\partial X}\,\log\det X = (X^{-1})^\top\) (X simétrica PD).

**Para qué sirve**  
- Entender **backprop**, **regularización** y **optimización** de pérdidas (por qué funciona Adam, por qué *early stopping* regulariza).

**Cómo estudiarlo**  
- Deriva la **MSE** y la **logística**.  
- Calcula a mano un paso de **descenso de gradiente** en regresión.

**Entregables**  
- Notebook con derivadas y comparación con *autograd* (PyTorch).

**Recurso claro**  
- “The Matrix Calculus You Need for Deep Learning”.

**Cómo lo usarás en proyectos**  
- Ajustar **LR/weight decay** con criterio y diagnosticar **overfitting** vs **underfitting**.  
- Implementar pérdidas custom (ej. **WAPE/SMAPE** para **forecast**).

---

## 3) Optimización

**Qué es**  
- **Convexidad** (Jensen), **Lipschitz**, **condicionamiento**.  
- **GD/Momentum/Adam**; **early stopping** y **regularización** (L1/L2).  
- **Lagrangiano** y **multiplicadores** (intuición práctica).

**Para qué sirve**  
- Convergencia **más estable** y modelos que **generalizan** mejor.  
- Diseñar *schedules* de LR y *weight decay* adecuados a tu problema.

**Cómo estudiarlo**  
- Comparativa de **SGD vs Adam** en el mismo MLP (curvas *loss/val*).

**Entregables**  
- Experimentos registrados en **MLflow**, con tabla de *runs* y métricas.

**Cómo lo usarás en proyectos**  
- **Mes 4–5**: entrenos de **transfer learning** estables.  
- **Mes 6–7**: *tuning* de ETS/ARIMA con *grid/random* bien acotado.

---

## 4) Probabilidad y estadística

**Qué es**  
- Distribuciones: **Bernoulli, Binomial, Poisson, Exponencial, Normal**.  
- Momentos: \(\mathbb E[X]\), **Var/Cov**, matrices de covarianza.  
- **MLE/MAP**, intervalos de confianza, **bootstrap**.  
- **Entropía**, **cross‑entropy**, **KL** (clasificación y VAEs).  
- Series: **autocovarianza**, **ACF/PACF** (para **Mes 6**).

**Para qué sirve**  
- Elegir **métricas** y **intervalos** realistas; entender **riesgo** y **incertidumbre** en predicción de demanda.

**Pitfalls**  
- Confundir varianza poblacional vs muestral.  
- Mezclar escalas/unidades sin normalizar.

**Entregables**  
- Notebook: estimación de \(\mu\) y \(\sigma\) con **bootstrap** + bandas de confianza.

**Cómo lo usarás en proyectos**  
- **Mes 7**: stock de seguridad (\(SS\)) y punto de pedido (\(ROP\)) con **Z‑scores**.  
- **Mes 3**: *threshold tuning* de clasificadores con curvas **PR** y coste.

---

## 5) Entregables rápidos (resumen)

- **Hoja de fórmulas** (A4) con 25 identidades (traza, log‑det, proyecciones).  
- Notebook de ejercicios:
  1. Deriva **MSE** y **logística**.
  2. Implementa **PCA por SVD** y compáralo con `sklearn`.
  3. Estima \(\sigma\) y \(\mu\) con **bootstrap** y grafica intervalos.

---

# B) Ecosistema de bibliotecas 2025 (curado y explicado)

> Para cada bloque: **Para qué sirve** y **Cómo lo usarás** (conectado a tu plan).

## Núcleo de ciencia de datos

- **NumPy / pandas** — estándar de arrays y DataFrames.  
  **Usarás**: EDA, *feature engineering*, *pipelines* tabulares.
- **Polars** — DataFrames **muy rápidos** (núcleo en Rust, *lazy*).  
  **Usarás**: ETL y analítica a gran escala en 1 máquina (suele reemplazar pandas cuando el dataset crece).
- **DuckDB** — OLAP **in‑process** (SQL), Parquet/CSV/S3.  
  **Usarás**: *joins* y agregaciones masivas desde notebooks o scripts, complementando pandas/Polars.

## ML clásico

- **scikit‑learn** — *pipelines*, métricas, validación, *model selection*.  
  **Usarás**: base de regresión/clasificación/clustering + **TimeSeriesSplit**.
- **XGBoost / LightGBM / CatBoost** — *boosting* SOTA en tabular.  
  **Usarás**: tabular serio (churn, fraude). CatBoost para categóricas pesadas.

## Deep Learning

- **PyTorch 2.x** — DL flexible (eager) y ecosistema enorme.  
  **Usarás**: MLP/CNN/Transformers, *transfer learning* y pérdidas custom.  
- **TensorFlow/Keras** — API alto nivel + **TFLite** en móvil/edge.  
  **Usarás**: exportar a **Android** (on‑device).  
- **JAX** (opcional) — *jit/vmap/pmap* para investigación numérica.  
- **timm / torchmetrics / Lightning** — *model zoo*, métricas, orquestación de *training*.  
- **MLflow** — *experiment tracking* y **model registry** (estándar de facto).

## Visión por Computadora

- **OpenCV** — preprocesado y utilidades de imagen/video.  
- **Albumentations** — *augmentation* rápido y flexible.  
- **Ultralytics YOLO** — detección/segmentación/pose con *training* sencillo.

## NLP / LLM

- **Hugging Face (Transformers/Datasets/Tokenizers)** — *pipelines* y *model zoo*.  
  **Usarás**: *fine‑tuning* ligero (LoRA), inferencia y **embeddings**.  
- **spaCy** — NLP productivo (tokenización/NER).  
- **SentencePiece/Tokenizers** — *subword* para vocabularios personalizados.
- **Servidores LLM**: **vLLM** (alto throughput), **TGI** (server listo), **llama.cpp/Ollama** (local).  
  **Usarás**: servir modelos en **RAG** y prototipos locales.

## RAG / Búsqueda vectorial

- **FAISS** / **pgvector** / **Qdrant/Weaviate/Milvus** — índices y DBs vectoriales.  
  **Usarás**: búsqueda semántica en RAG y recomendadores híbridos.  
- **LangChain / LlamaIndex** — *chains*, *retrievers*, *re‑ranking*.  
  **Usarás**: orquestar RAG con herramientas y *guardrails*.

## Evaluación, calidad y observabilidad

- **Evidently** — *drift/monitoring*.  
- **Great Expectations** — **tests de datos** (calidad).  
- **Ragas / DeepEval / LM‑Eval‑Harness** — evaluación de RAG/LLM.  
- **Langfuse** — **trazas** y analítica para LLM apps.

## Serving e Inferencia

- **FastAPI + Pydantic v2** — APIs rápidas y tipadas.  
- **ONNX Runtime / NVIDIA Triton / BentoML / Ray Serve / KServe** — inferencia en CPU/GPU/K8s.  
  **Usarás**: exportar modelos a **ONNX** (portabilidad) y servir en CPU/GPU.

## Orquestación / MLOps

- **DVC** — versionado de datos.  
- **MLflow** — experimentos y **model registry**.  
- **Airflow / Prefect / Dagster** — *pipelines* y scheduling (Prefect = DX muy limpia).

## Web y Móvil (on‑device)

- **Transformers.js / ONNX Runtime Web / TensorFlow.js** — inferencia en navegador (WebGPU/WASM).  
- **TFLite/LiteRT / ML Kit / ONNX Runtime Mobile** — inferencia en Android/iOS.  
  **Usarás**: PoCs *serverless* y apps móviles privadas/offline.

---

# Stacks de referencia y **cómo los usarás**

### 1) Tabular (regresión/clasificación)
**Stack**: pandas/Polars → scikit‑learn + (XGBoost/LightGBM) → MLflow → FastAPI/ONNX Runtime → Airflow/Prefect.  
**Por qué**: máxima eficacia en datos empresariales (tabulares).  
**Cómo lo usarás**: **churn, fraude, scoring** con API `/predict`, dashboard Vue y *monitoring* con Evidently.

### 2) Forecast por SKU (Mes 6–7)
**Stack**: pandas/Polars → statsmodels (ETS/SARIMA) + baselines → backtesting (*walk‑forward*) → FastAPI + Vue → MLflow/Evidently.  
**Cómo lo usarás**: **motor de compras** → `SS/ROP/EOQ`, priorización **ABC/XYZ**, endpoint `/purchase_suggestions`.

### 3) NLP clásico (spam/intents)
**Stack**: spaCy/NLTK → TF‑IDF + LinearSVC/LogReg (o DistilBERT) → FastAPI.  
**Cómo lo usarás**: filtros de **spam** y **intents** en soporte/tickets.

### 4) RAG de documentación interna
**Stack**: ingestión/chunking → embeddings (HF/SBERT) → FAISS/pgvector → LangChain/LlamaIndex → vLLM/TGI → Ragas/Langfuse.  
**Cómo lo usarás**: **asistente** de conocimiento con **citas** y **evaluación** continua.

### 5) Visión (defectos/SKU)
**Stack**: OpenCV + Albumentations → PyTorch/timm o YOLO → ONNX/TFLite → Triton/Android.  
**Cómo lo usarás**: control de **calidad** y conteo/clasificación rápida en móvil.

### 6) In‑browser
**Stack**: Transformers.js u ONNX Runtime Web.  
**Cómo lo usarás**: demo sin backend (privacidad, cero latencia de red).

### 7) Android on‑device
**Stack**: TFLite/ML Kit/ONNX Runtime Mobile.  
**Cómo lo usarás**: **apps offline** (visión/NLP) integradas con tu stack Kotlin.

---

# Consejos prácticos (de profesional a profesional)

- **Elige DataFrame**: si haces ETL/analítica pesada en 1 máquina, **Polars + DuckDB** = combo rápido y barato.  
- **Sirve LLMs sin dolor**: empieza por **vLLM** (prod) u **Ollama** (local/dev). **TGI** si prefieres server listo.  
- **Evalúa RAG desde el día 1**: integra **Ragas** en CI y traza con **Langfuse**.  
- **Observabilidad**: *drift* con **Evidently**; **Great Expectations** para *data tests*; re‑entrenos **gated** por métricas de negocio.  
- **Exportabilidad**: compila a **ONNX** temprano → abre **Triton**, **ORT Web/Mobile**, **KServe**.

---

# Qué añadir al calendario (puntos concretos)

- **Mes 1 — Semana 1.5 (nuevo)**: repaso de álgebra/cálculo (arriba) + 2–3 *katas*/día.  
  **Entregables**: *cheat‑sheet* A4 + notebook de derivadas/PCA + mini‑quiz.
- **Mes 3 — Feature Store (1 día)**: mira **Feast** si compartirás *features* entre equipos.  
- **Mes 4 — Tracking serio**: estandariza **MLflow** (experimentos + *model registry*; tags: dataset, git sha, semilla).  
- **Mes 9 — RAG**: añade **Ragas** y *prompt hardening*.  
- **Mes 12 — Serving**: ensaya **Triton** (GPU) y **ONNX Runtime** (CPU/Edge) + *smoke tests* en CI.

---

# Mini‑glosario extra

- **SVD vs Eig**: SVD funciona para cualquier matriz; Eig solo en cuadradas.  
- **Lipschitz**: cota al cambio; te guía en *step size*.  
- **MAP**: MLE con *prior* → L2 ≈ **prior Gaussiano**, L1 ≈ **Laplace**.  
- **vLLM**: servidor LLM con *PagedAttention* (alto *throughput*).  
- **TGI**: servidor de generación (Hugging Face) listo para prod.

---

# Siguientes pasos inmediatos

1. Imprime tu **cheat‑sheet** y crea el **notebook** de derivadas + PCA.  
2. Elige **Polars + DuckDB** o **pandas** para tu EDA base y registra todo con **MLflow**.  
3. Para **RAG**: prueba **FAISS + LangChain + vLLM** con un PDF del negocio y mide con **Ragas**.

---

## Créditos y recursos (rápidos)

- Libros: *Hands‑On Machine Learning* (Géron), *Forecasting: Principles and Practice* (Hyndman), *Deep Learning* (Goodfellow).  
- Docs oficiales: scikit‑learn, pandas/Polars, statsmodels, PyTorch, TensorFlow, MLflow, FastAPI, Hugging Face, DuckDB.  
- Herramientas: Label Studio, DVC/MLflow, Great Expectations, Airflow/Prefect, Evidently, Langfuse.

---

> **Pro tip**: este README es tu “contrato” de aprendizaje. Cada semana **marca entregables**, sube *screenshots* a `reports/` y escribe un **post‑mortem** corto con lo aprendido y qué harás distinto la próxima vez.
