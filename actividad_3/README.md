# Actividad 3 Experimentación con MLflow

**Alumno:** Humberto Alejandro Espinola González  
**Matrícula:** A01840066  
**Materia:** TC5061

## Objetivo

Convertir el modelo baseline de la actividad anterior en código reutilizable y parametrizado, registrar sus ejecuciones con MLflow Tracking y comparar configuraciones mediante resultados reproducibles.

El problema consiste en clasificar los tres cultivares del dataset Wine, distribuido con scikit-learn. El conjunto contiene 178 muestras y 13 predictores.

## Modelo y evaluación

Se conserva el pipeline de la actividad 2:

```text
SimpleImputer(strategy="median")
    → StandardScaler
    → LogisticRegression con regularización L2
```

La partición estratificada, con semilla 42, contiene 142 muestras de entrenamiento y 36 de prueba.

Las configuraciones se comparan mediante validación cruzada estratificada de cinco folds dentro del entrenamiento. La imputación y la estandarización se ajustan nuevamente dentro de cada fold.

La métrica principal es `cv_f1_macro_mean`. También se registran accuracy media, desviaciones entre folds, métricas de entrenamiento e iteraciones del clasificador. La configuración seleccionada se evalúa posteriormente en la prueba histórica.

## Estructura

```text
actividad_3/
├── README.md
├── pyproject.toml
├── requirements.txt
├── train.py
├── run_experiments.py
├── view_results.py
├── notebooks/
│   └── 0.01-he-wine-eda.ipynb
├── reports/
│   └── figures/
├── results/
│   ├── comparacion.csv
│   ├── comparacion.json
│   ├── reproducibilidad.json
│   └── runs/                      # Generado localmente
├── tracking/                     # Generado localmente
└── wine_mlops/
    ├── config.py
    └── modeling/
        └── train.py
```

`train.py` es la entrada de línea de comandos. La implementación principal se encuentra en `wine_mlops/modeling/train.py`, organizada en funciones para cargar datos, particionar, construir el pipeline, validar, entrenar, evaluar y registrar resultados.

El notebook se conserva exclusivamente para exploración.

## Instalación

El entorno utilizado para los resultados registrados fue Python 3.13.9.

Desde `actividad_3`:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

El archivo de dependencias incluye la instalación editable del paquete mediante `-e .`.

Las versiones principales utilizadas son MLflow 3.16.1, scikit-learn 1.7.2, NumPy 2.3.5, pandas 2.3.3 y SciPy 1.16.3. Sus versiones y las demás dependencias principales están registradas en `requirements.txt` y en el artefacto `entorno.json` de cada corrida.

Wine se carga mediante `load_wine`; no requiere descargar un archivo de datos.

## Ejecución

Corrida con valores por defecto:

```bash
python train.py
```

Consultar todos los argumentos:

```bash
python train.py --help
```

Ejemplo de parametrización:

```bash
python train.py --c 0.01 --class-weight balanced --tol 0.001 --seed 42
```

| Argumento | Valor por defecto | Función |
|---|---|---|
| `--c` | `1.0` | Inversa de la fuerza de regularización L2. |
| `--class-weight` | `none` | Ponderación de clases: `none` o `balanced`. |
| `--tol` | `0.0001` | Tolerancia de convergencia. |
| `--max-iter` | `1000` | Máximo de iteraciones. |
| `--solver` | `lbfgs` | Optimizador del clasificador. |
| `--seed` | `42` | Semilla de reproducibilidad. |
| `--cv-folds` | `5` | Número de folds de validación. |
| `--evaluate-test` | Desactivado | Añade la evaluación de la prueba histórica. |

## Comparación de experimentos

```bash
python run_experiments.py --batch entrega_20261011 --seed 42
```

Este comando ejecuta siete configuraciones, repite el baseline con argumentos idénticos y evalúa la configuración seleccionada en prueba.

| Corrida | C | class_weight | tol | F1 macro medio CV | Accuracy media CV |
|---|---:|---|---:|---:|---:|
| R01_C0001 | 0.001 | none | 0.0001 | 0.608686 | 0.689901 |
| R02_C001 | 0.01 | none | 0.0001 | 0.979290 | 0.978818 |
| R03_C01 | 0.1 | none | 0.0001 | 0.972028 | 0.972167 |
| R04_baseline | 1.0 | none | 0.0001 | **0.979637** | **0.979064** |
| R05_C10 | 10.0 | none | 0.0001 | 0.972223 | 0.971921 |
| R06_balanced | 1.0 | balanced | 0.0001 | 0.972285 | 0.972167 |
| R07_tol001 | 1.0 | none | 0.01 | 0.972028 | 0.972167 |

Todas mantienen solver=lbfgs, max_iter=1000, seed=42 y cinco folds.

La regla de selección es mayor media de F1 macro; ante un empate exacto, se prefiere menor C y después el orden prefijado de las configuraciones.

- **R08_repeticion_baseline:** comprobación de reproducibilidad de R04. Se renombra después de entrenar para mantener idénticos los argumentos ejecutados.
- **R09_seleccion_prueba:** evaluación de la configuración elegida en las 36 muestras de prueba.

Cada ejecución crea nuevas corridas. Para una nueva comparación puedes utilizar otra etiqueta `--batch`; los archivos de resumen locales se actualizan con la última serie.

## Consultar MLflow

```bash
python view_results.py
```

Abre http://127.0.0.1:5001 y selecciona:

```text
TC5061_Wine_LogisticRegression_EspinolaHumberto
```

Para mostrar las siete configuraciones de la entrega, utiliza el filtro:

```text
tags.batch = 'entrega_20261011' AND tags.role = 'candidate'
```

Selecciona R01 a R07 y pulsa **Compare**. Muestra los parámetros C, class_weight y tol, junto con `cv_f1_macro_mean` y `cv_accuracy_mean`.

Si repetiste la misma etiqueta, utiliza los IDs de `results/comparacion.json` para identificar la serie más reciente.

## Registro y reproducibilidad

Cada corrida registra:

- Hiperparámetros y configuración con `mlflow.log_param`.
- Métricas con `mlflow.log_metric`.
- Predicciones, matrices de confusión, resultados por fold, código y metadatos con `mlflow.log_artifact`.
- El pipeline completo con `mlflow.sklearn.log_model`, incluyendo firma, ejemplo de entrada y dependencias del modelo.

La semilla controla Python, NumPy, partición, folds y clasificador. Los folds se procesan secuencialmente y el cálculo numérico se limita a un hilo.

`results/reproducibilidad.json` confirma la igualdad de argumentos, datos, particiones, métricas y estado aprendido entre R04 y su repetición. La comprobación corresponde al mismo entorno y plataforma.

El modelo registrado se recarga durante cada ejecución y se comprueba que reproduce las predicciones de entrenamiento.

## Interpretación

R04 obtuvo la mayor media de F1 macro CV, aunque su diferencia frente a R02 fue de solo 0.000347. El resultado permite conservar el baseline, sin afirmar una superioridad estadística.

C fue el parámetro más influyente en el rango probado. La regularización muy fuerte de C=0.001 redujo notablemente el desempeño; entre C=0.01 y C=10 las diferencias fueron menores.

En prueba, R09 obtuvo accuracy=0.972222 y F1 macro=0.970962, con 35 aciertos de 36 muestras.

El dataset pequeño, la selección sobre la misma validación cruzada y el uso de una prueba examinada previamente limitan las conclusiones sobre generalización. La desviación entre folds es descriptiva y no constituye un intervalo de confianza.

## Evidencias y entrega

Guardar las capturas originales en `reports/figures/`:

1. Tabla comparativa de las siete configuraciones.
2. Vista Compare de R04 y R02.
3. Detalle de R09 con métricas, artefactos y modelo.

Insertar las capturas en páginas independientes y comprobar su legibilidad en el PDF exportado.

Nombre del documento de entrega:

```text
TC5061_Experimentacion_MLFlow_EspinolaHumberto.pdf
```

El PDF debe incluir el enlace al repositorio, los comandos exactos, el análisis y las evidencias. El máximo de texto es de dos cuartillas, sin contar capturas ni anexos.

## Referencias

- [Wine en UCI](https://archive.ics.uci.edu/dataset/109/wine)
- [LogisticRegression en scikit-learn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)

## Apoyo de inteligencia artificial

Se utilizó OpenAI Codex como apoyo para refactorización, revisión y redacción.