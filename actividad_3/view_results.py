"""Abre el servidor local de MLflow usando rutas resueltas desde el proyecto."""
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
environment = {**os.environ, "MLFLOW_DISABLE_TELEMETRY": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}
print("Interfaz: http://127.0.0.1:5001 | Termina con Ctrl+C", flush=True)
subprocess.run([sys.executable, "-m", "mlflow", "server", "--backend-store-uri",
    f"sqlite:///{root / 'tracking' / 'mlflow.db'}", "--host", "127.0.0.1",
    "--port", "5001", "--workers", "1"], env=environment, check=True)
