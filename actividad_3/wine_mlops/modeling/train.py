"""Entrena la regresión logística Wine y registra una corrida en MLflow."""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

from wine_mlops.config import PROJ_ROOT as ROOT
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

EXPERIMENT = "TC5061_Wine_LogisticRegression_EspinolaHumberto"
PACKAGES = ["mlflow", "scikit-learn", "numpy", "pandas", "scipy", "matplotlib",
            "cloudpickle", "joblib", "threadpoolctl"]


def parse_args(argv=None):
    """CLI con configuración baseline por defecto y validación de argumentos."""
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--c", type=float, default=1.0, help="Inversa de la fuerza de regularización L2; debe ser positiva.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none", help="Ponderación de clases del clasificador.")
    parser.add_argument("--tol", type=float, default=0.0001, help="Tolerancia positiva del criterio de convergencia.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Máximo de iteraciones del optimizador.")
    parser.add_argument("--solver", choices=["lbfgs", "newton-cg"], default="lbfgs", help="Optimizador compatible con L2 multiclase.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para Python, NumPy, partición, folds y modelo.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds estratificados dentro de entrenamiento (2 a 38).")
    parser.add_argument("--experiment-name", default=EXPERIMENT, help="Experimento compartido por las corridas.")
    parser.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'tracking' / 'mlflow.db'}", help="Ubicación del backend de Tracking.")
    parser.add_argument("--run-name", default="baseline", help="Nombre legible de la corrida.")
    parser.add_argument("--batch", default="manual", help="Etiqueta para agrupar una comparación.")
    parser.add_argument("--role", choices=["candidate", "final"], default="candidate", help="Rol de la corrida en el ejercicio.")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluar además la prueba histórica; usar tras seleccionar por CV.")
    args = parser.parse_args(argv)
    if not np.isfinite(args.c) or args.c <= 0 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--c y --tol deben ser positivos y finitos")
    if args.max_iter < 1 or not 2 <= args.cv_folds <= 38 or not 0 <= args.seed < 2**32:
        parser.error("Revisa --max-iter (>=1), --cv-folds (2..38) y --seed (0..2**32-1)")
    return args


def load_data():
    """Carga Wine local de scikit-learn y comprueba el esquema del baseline."""
    dataset = load_wine(as_frame=True)
    frame = dataset.frame.rename(columns={"target": "clase"})
    features = list(dataset.feature_names)
    if frame.shape != (178, 14) or set(frame.clase.unique()) != {0, 1, 2}:
        raise ValueError("El esquema de Wine cambió")
    if frame.duplicated().any() or frame.duplicated(subset=features).any():
        raise ValueError("Revisa duplicados antes de particionar")
    if frame.clase.isna().any():
        raise ValueError("No se permiten etiquetas faltantes")
    digest = hashlib.sha256(frame.to_csv(index=True, float_format="%.17g",
        lineterminator="\n").encode()).hexdigest()
    X = frame[features].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan).astype("float64")
    return X, frame.clase.astype("int64"), digest


def split_data(X, y, seed):
    """Conserva la partición estratificada 80/20 del notebook anterior."""
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)


def build_pipeline(args):
    """Todas las estadísticas aprendidas se ajustan dentro del pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=args.c, penalty="l2", solver=args.solver,
            tol=args.tol, max_iter=args.max_iter, random_state=args.seed,
            class_weight=None if args.class_weight == "none" else "balanced")),
    ])


def scores(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def cross_validate(pipeline, X, y, folds, seed):
    """Reajusta todo el pipeline en cada fold y conserva predicciones fuera de fold."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions = np.empty(len(y), dtype="int64")
    fold_assignment = np.empty(len(y), dtype="int64")
    rows, split_manifest = [], []
    for fold, (fit, val) in enumerate(cv.split(X, y), start=1):
        estimator = clone(pipeline).fit(X.iloc[fit], y.iloc[fit])
        pred = estimator.predict(X.iloc[val])
        predictions[val] = pred
        fold_assignment[val] = fold
        rows.append({"fold": fold, "validation_rows": len(val), **scores(y.iloc[val], pred),
            "iterations": int(estimator.named_steps["classifier"].n_iter_.max())})
        split_manifest.append({"fold": fold, "fit_rows": X.index[fit].tolist(),
                               "validation_rows": X.index[val].tolist()})
    frame = pd.DataFrame(rows)
    metrics = {f"cv_{metric}_mean": float(frame[metric].mean()) for metric in ["accuracy", "f1_macro"]}
    metrics.update({f"cv_{metric}_std": float(frame[metric].std(ddof=0)) for metric in ["accuracy", "f1_macro"]})
    oof = pd.DataFrame({"fila_original": X.index, "fold": fold_assignment,
                        "clase_real": y.to_numpy(), "clase_predicha": predictions})
    return metrics, frame, oof, split_manifest


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_confusion(y, pred, destination, title):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, labels=[0, 1, 2],
        display_labels=["Clase 0", "Clase 1", "Clase 2"], ax=ax, cmap="Blues", colorbar=False)
    ax.set(title=title, xlabel="Clase predicha", ylabel="Clase real")
    fig.tight_layout()
    fig.savefig(destination, dpi=240)
    plt.close(fig)


def configure_tracking(args):
    (ROOT / "tracking").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    if client.get_experiment_by_name(args.experiment_name) is None:
        artifacts = ROOT / "tracking" / "artifacts"
        artifacts.mkdir(exist_ok=True)
        mlflow.create_experiment(args.experiment_name, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(args.experiment_name)
    return client


def run_training(args):
    """Orquesta una ejecución completa, con trazabilidad de datos, código y modelo."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    client = configure_tracking(args)
    X, y, data_hash = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, args.seed)
    params = {"C": args.c, "class_weight": args.class_weight, "tol": args.tol,
        "max_iter": args.max_iter, "solver": args.solver, "penalty": "l2", "seed": args.seed,
        "cv_folds": args.cv_folds, "test_size": 0.20, "train_rows": len(y_train),
        "test_rows": len(y_test), "n_features": X.shape[1], "dataset": "Wine_UCI_sklearn",
        "dataset_sha256": data_hash, "imputer": "median", "scaler": "StandardScaler",
        "primary_metric": "cv_f1_macro_mean", "evaluate_test": args.evaluate_test,
        "thread_limit": 1}
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        output = ROOT / "results" / "runs" / run_id
        output.mkdir(parents=True, exist_ok=True)
        mlflow.set_tags({"autor": "Humberto Alejandro Espinola González", "actividad": "TC5061_Experimentacion_MLFlow",
            "batch": args.batch, "role": args.role, "selection_metric": "cv_f1_macro_mean"})
        for key, value in params.items():
            mlflow.log_param(key, value)
        pipeline = build_pipeline(args)
        # El contexto controla BLAS aunque NumPy ya esté importado; sin paralelismo entre folds.
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            metrics, folds, oof, cv_splits = cross_validate(pipeline, X_train, y_train, args.cv_folds, args.seed)
            pipeline.fit(X_train, y_train)
            metrics.update({f"train_{k}": v for k, v in scores(y_train, pipeline.predict(X_train)).items()})
            test_pred = pipeline.predict(X_test) if args.evaluate_test else None
        metrics["fit_iterations"] = float(pipeline.named_steps["classifier"].n_iter_.max())
        folds.to_csv(output / "cv_folds.csv", index=False)
        oof.to_csv(output / "predicciones_cv.csv", index=False)
        save_confusion(oof.clase_real, oof.clase_predicha, output / "matriz_confusion_cv.png", "Validación cruzada: predicciones fuera de fold")
        write_json(output / "reporte_cv.json", classification_report(oof.clase_real,
            oof.clase_predicha, output_dict=True, zero_division=0))
        split = {"train_rows": X_train.index.tolist(), "test_rows": X_test.index.tolist(), "cv_splits": cv_splits}
        write_json(output / "particiones.json", split)
        params["split_sha256"] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
        mlflow.log_param("split_sha256", params["split_sha256"])
        if args.evaluate_test:
            metrics.update({f"test_{k}": v for k, v in scores(y_test, test_pred).items()})
            pd.DataFrame({"fila_original": X_test.index, "clase_real": y_test.to_numpy(),
                "clase_predicha": test_pred}).to_csv(output / "predicciones_prueba.csv", index=False)
            save_confusion(y_test, test_pred, output / "matriz_confusion_prueba.png", "Configuración seleccionada: prueba histórica")
            write_json(output / "reporte_prueba.json", classification_report(y_test, test_pred, output_dict=True, zero_division=0))
        write_json(output / "metricas.json", metrics)
        write_json(output / "parametros.json", params)
        environment = {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: version(p) for p in PACKAGES}, "thread_limit": 1}
        write_json(output / "entorno.json", environment)
        classifier = pipeline.named_steps["classifier"]
        learned = {"coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
            "imputer": pipeline.named_steps["imputer"].statistics_.tolist(),
            "mean": pipeline.named_steps["scaler"].mean_.tolist(),
            "scale": pipeline.named_steps["scaler"].scale_.tolist()}
        model_hash = hashlib.sha256(json.dumps(learned, sort_keys=True).encode()).hexdigest()
        write_json(output / "estado_aprendido.json", learned)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        for path in sorted(output.iterdir()):
            mlflow.log_artifact(str(path), artifact_path="evaluacion" if path.suffix == ".png" or path.name.startswith(("predicciones", "reporte", "metricas", "cv_folds")) else "reproducibilidad")
        training_source = Path(__file__).resolve()
        for source in (
            training_source,
            ROOT / "wine_mlops" / "config.py",
            ROOT / "requirements.txt",
        ):
            mlflow.log_artifact(
                str(source),
                artifact_path="codigo",
            )

        mlflow.set_tag(
            "train_code_sha256",
            hashlib.sha256(training_source.read_bytes()).hexdigest(),
        )
        example = X_train.head(3)
        model_info = mlflow.sklearn.log_model(pipeline, name="modelo_wine",
            signature=infer_signature(example, pipeline.predict(example)), input_example=example,
            pip_requirements=[f"{p}=={version(p)}" for p in ["scikit-learn", "numpy", "pandas", "scipy", "cloudpickle", "joblib", "threadpoolctl"]],
            serialization_format="cloudpickle")
        restored = mlflow.sklearn.load_model(model_info.model_uri)
        if not np.array_equal(pipeline.predict(X_train), restored.predict(X_train)):
            raise RuntimeError("El modelo recargado cambió las predicciones")
        mlflow.set_tag("model_reload_verified", "true")
        summary = {"run_id": run_id, "run_name": args.run_name, "experiment_id": run.info.experiment_id,
            "experiment_name": args.experiment_name, "batch": args.batch,
            "model_uri": model_info.model_uri, "learned_state_sha256": model_hash,
            "params": params, "metrics": metrics, "argv": sys.argv[1:]}
        write_json(output / "resumen.json", summary)
        mlflow.log_artifact(str(output / "resumen.json"), artifact_path="reproducibilidad")
    summary["status"] = client.get_run(run_id).info.status
    write_json(output / "resumen.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
