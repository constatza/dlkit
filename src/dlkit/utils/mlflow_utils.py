import requests

def is_mlflow_server_running(host: str = "localhost", port: int = 5000) -> bool:
    url = f"http://{host}:{port}/api/2.0/mlflow/experiments/list"
    try:
        response = requests.get(url, timeout=5)  # Set a timeout to avoid hanging
        if response.status_code == 200:
            return True
    except requests.ConnectionError:
        return False
    except requests.Timeout:
        return False
    return False