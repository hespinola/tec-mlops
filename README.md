# TC5061 MLOps

Repositorio individual de actividades de la materia TC5061, de la Maestría en Inteligencia Artificial Aplicada del Tecnológico de Monterrey.

**Alumno:** Humberto Alejandro Espinola González  
**Matrícula:** A01840066

## Actividades

| Carpeta | Contenido |
|---|---|
| [actividad_1](actividad_1/) | Introducción a MLflow Tracking mediante el registro de parámetros, métricas y artefactos de un ejemplo sintético. |
| [actividad_2](actividad_2/) | Exploración del dataset Wine, preparación de datos y entrenamiento de un modelo baseline de clasificación multiclase. |
| [actividad_3](actividad_3/) | Migración del baseline a un script parametrizado, comparación de experimentos y comprobación de reproducibilidad con MLflow. |

La actividad 3 continúa el trabajo de la actividad 2: conserva el dataset Wine y el pipeline de imputación por mediana, estandarización y regresión logística.

## Ejecutar la actividad 3

Desde la raíz del repositorio:

```bash
cd actividad_3
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Para ejecutar una corrida con valores por defecto:

```bash
python train.py
```

Para ejecutar la comparación de configuraciones:

```bash
python run_experiments.py --batch entrega_20261011 --seed 42
```

Para consultar los resultados en MLflow:

```bash
python view_results.py
```

La interfaz estará disponible en http://127.0.0.1:5001.

Las instrucciones completas, los hiperparámetros y los resultados se encuentran en el [README de la actividad 3](actividad_3/README.md).

## Organización y reproducibilidad

Cada actividad conserva sus archivos y dependencias. La actividad 3 utiliza una estructura generada con Cookiecutter Data Science y mantiene el notebook exclusivamente para exploración.

El código y los resúmenes de resultados se versionan en Git. Los entornos virtuales, las cachés y el almacenamiento local de MLflow se excluyen mediante `.gitignore`.

## Referencias

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Wine en UCI](https://archive.ics.uci.edu/dataset/109/wine)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)