# NexWearAI

Microservicio de Inteligencia Artificial de NexWear. Expone 7 modelos de
Machine Learning como endpoints REST (FastAPI), consumidos por el backend
principal (NexWearAPI, .NET) vía HTTP.

## Arquitectura

```
NexWearAI/
├── main.py              # Punto de entrada, registra los routers
├── db.py                # Conexión a la base de datos (PostgreSQL)
├── etl/
│   └── etl_pipeline.py  # Extracción y limpieza de datos reales de producción
├── routers/              # Un endpoint por modelo
│   ├── demand.py
│   ├── pricing.py
│   ├── image_search.py
│   ├── recommender.py
│   ├── discount.py
│   ├── review_detector.py
│   └── stylist.py
├── training/              # Entrenamiento + generación de datasets sintéticos
├── models/                # Modelos entrenados (.pkl, no versionados)
├── data/                  # Datasets curados por el ETL (no versionados)
└── check_*.py, test_clip.py, etc.  # Scripts de verificación / debug puntual
```

## Modelos servidos

| Endpoint | Modelo | Tipo |
|---|---|---|
| `/predict/demand` | Demand Forecast AI | Regresión |
| `/predict/pricing` | Fashion Pricing Intelligence | Regresión |
| `/search/image` | Búsqueda por imagen | CLIP (zero-shot) |
| `/recommend` | Recomendador de productos | Similitud (TF-IDF) |
| `/predict/discount` | Smart Personalized Discounts | Clasificación |
| `/detect/fake-review` | Detección de reseñas falsas | Clasificación (NLP) |
| `/stylist` | Fashion Stylist | Clustering + red neuronal |

## Instalación local

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Crear un archivo `.env` en la raíz con:

```
DATABASE_URL=postgresql://usuario:password@host:puerto/basededatos
```

## Ejecutar el servicio

```bash
uvicorn main:app --reload
```

Swagger disponible en `http://localhost:8000/docs`.

## Equipo y ownership de módulos

Grupo IDyGS9A — Emiliano Monge Osuna, Nicol Amairani Gastélum Díaz, Alexa Marian Gastélum Díaz.

| Módulo | Responsable |
|---|---|
| Núcleo (`main.py`, `db.py`, `etl/`), infraestructura y despliegue | Emiliano |
| Demand Forecast, Pricing Intelligence, Smart Discounts, Recomendador | Nicol |
| Búsqueda por Imagen, Fashion Stylist, Detección de Reseñas Falsas | Alexa |

## Pruebas y calidad

Ver `Unidad_4_Actividad_2_Pruebas_de_software` para el plan de pruebas completo
(caja negra, unitarias, integración y usabilidad) y el registro de incidencias
resueltas durante el desarrollo.
