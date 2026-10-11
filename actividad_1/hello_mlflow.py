"""Hello World de MLflow"""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("MLOps - Hello World")

    # Ejemplo sintético: evaluar predicciones prefijadas, sin entrenar un modelo.
    threshold = 0.5
    labels = [0, 1, 1, 0, 1]
    probabilities = [0.1, 0.9, 0.4, 0.2, 0.8]
    predictions = [int(p >= threshold) for p in probabilities]
    correct = sum(y == pred for y, pred in zip(labels, predictions))
    accuracy = correct / len(labels)
    error_rate = (len(labels) - correct) / len(labels)

    with mlflow.start_run(run_name="evaluacion_basica") as run:
        # Parámetros: configuración y descripción del experimento.
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_samples", len(labels))
        mlflow.log_param("dataset", "sintetico_v1")

        # Métricas calculadas a partir de los ejemplos.
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("error_rate", error_rate)

        # Artefacto: CSV con los datos que permiten revisar los resultados.
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "predicciones.csv"
            with artifact.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["etiqueta_real", "probabilidad", "prediccion"])
                writer.writerows(zip(labels, probabilities, predictions))
            mlflow.log_artifact(str(artifact), artifact_path="resultados")

        run_id = run.info.run_id

    print(f"Run registrado: {run_id}")
    print(f"Accuracy: {accuracy:.2f} | Error rate: {error_rate:.2f}")
    print("Abre http://localhost:5000 y selecciona MLOps - Hello World.")


if __name__ == "__main__":
    main()
