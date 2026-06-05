from pathlib import Path
import yaml

def load_yaml_config(path: Path, must_exist: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        if must_exist:
            raise FileNotFoundError(f"Config file not found: {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except Exception as e:
        raise RuntimeError(f"Failed to parse YAML '{path.name}': {e}")
