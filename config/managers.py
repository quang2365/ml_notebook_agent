import json 
from pathlib import Path

CONFIG_DIR = Path.home()/".qiu"

CONFIG_FILE = CONFIG_DIR/"config.json"


def remove_config() -> bool:
    try:
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink(),
            return True
        return False
    except OSError:
        return False

def save_config(provider:str,base_url:str | None,model:str):
    CONFIG_DIR.mkdir(parents=True,exist_ok=True)
    data = {"provider": provider,"base_url":base_url,"model": model}
    CONFIG_FILE.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2),
            encoding="utf-8")
def load_config() -> dict | None:
    if not CONFIG_FILE.exists():
        return None
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    
        return config
    except (OSError,json.JSONDecodeError): 
        return None
def is_config() -> bool:
    data = load_config()

    if not data or not isinstance(data,dict):
        return False
    return bool(data.get("provider") and data.get("model"))
