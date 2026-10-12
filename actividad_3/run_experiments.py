"""Ejecuta siete configuraciones, repite baseline y evalúa la elegida por CV."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd

from wine_mlops.modeling.train import EXPERIMENT, ROOT, write_json

CONFIGURATIONS = [
    ("R01_C0001", 0.001, "none", 0.0001),
    ("R02_C001", 0.01, "none", 0.0001),
    ("R03_C01", 0.1, "none", 0.0001),
    ("R04_baseline", 1.0, "none", 0.0001),
    ("R05_C10", 10.0, "none", 0.0001),
    ("R06_balanced", 1.0, "balanced", 0.0001),
    ("R07_tol001", 1.0, "none", 0.01),
]


def execute(command):
    """Invoca el mismo script de CLI; captura su resumen y conserva el comando."""
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr + process.stdout)
    result = json.loads(process.stdout.strip().splitlines()[-1])
    result["command"] = "python " + " ".join(command[1:])
    print(f"{result['run_name']}: CV F1={result['metrics']['cv_f1_macro_mean']:.6f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=datetime.now(timezone.utc).strftime("wine_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    common = ["--max-iter", "1000", "--solver", "lbfgs", "--seed", str(args.seed),
              "--cv-folds", "5", "--batch", args.batch]
    candidates, commands = [], []
    for name, c, weight, tol in CONFIGURATIONS:
        command = [sys.executable, "train.py", "--c", str(c), "--class-weight", weight,
                   "--tol", str(tol), "--run-name", name, *common]
        commands.append(command)
        candidates.append(execute(command))
    repeat = execute(commands[3])  # Argumentos idénticos, incluida la etiqueta de corrida.
    baseline = candidates[3]
    checks = {"identical_arguments": repeat["argv"] == baseline["argv"],
        "identical_metrics": repeat["metrics"] == baseline["metrics"],
        "identical_dataset": repeat["params"]["dataset_sha256"] == baseline["params"]["dataset_sha256"],
        "identical_splits": repeat["params"]["split_sha256"] == baseline["params"]["split_sha256"],
        "identical_learned_state": repeat["learned_state_sha256"] == baseline["learned_state_sha256"]}
    reproducibility = {"baseline_run_id": baseline["run_id"], "repeat_run_id": repeat["run_id"],
        "command": baseline["command"], "checks": checks,
        "scope": "Dos procesos independientes; mismo entorno, plataforma y versiones."}
    if not all(checks.values()):
        raise RuntimeError("Falló la comprobación de reproducibilidad: " + str(checks))
    # Regla prefijada: mayor media F1; empate exacto, menor C y orden de la lista.
    best = sorted(candidates, key=lambda r: (-r["metrics"]["cv_f1_macro_mean"], r["params"]["C"]))[0]
    params = best["params"]
    final = execute([sys.executable, "train.py", "--c", str(params["C"]),
        "--class-weight", params["class_weight"], "--tol", str(params["tol"]),
        "--run-name", "R09_seleccion_prueba", "--role", "final", "--evaluate-test", *common])
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'tracking' / 'mlflow.db'}")
    client = mlflow.tracking.MlflowClient()
    client.set_tag(repeat["run_id"], "role", "reproducibility")
    client.set_tag(repeat["run_id"], "mlflow.runName", "R08_repeticion_baseline")
    client.set_tag(best["run_id"], "selected_by_cv", "true")
    for r in [baseline, repeat]:
        client.log_artifact(r["run_id"], str(ROOT / "run_experiments.py"), "codigo")
    results = ROOT / "results"
    write_json(results / "reproducibilidad.json", reproducibility)
    write_json(results / "comparacion.json", {"batch": args.batch, "experiment": EXPERIMENT,
        "selection_rule": "Mayor cv_f1_macro_mean; empate exacto: menor C, luego orden prefijado.",
        "candidates": candidates, "best_candidate": best, "repeat": repeat, "final": final})
    rows = [{"run_name": r["run_name"], "run_id": r["run_id"],
             **{p: r["params"][p] for p in ["C", "class_weight", "tol", "max_iter", "seed"]},
             **r["metrics"]} for r in candidates]
    pd.DataFrame(rows).to_csv(results / "comparacion.csv", index=False)
    for r in [baseline, repeat, final]:
        client.log_artifact(r["run_id"], str(results / "reproducibilidad.json"), "reproducibilidad")
    print("Seleccionada:", best["run_name"], "| Prueba:", final["metrics"]["test_f1_macro"])


if __name__ == "__main__":
    main()
