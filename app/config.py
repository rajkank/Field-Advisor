import os
import re
from pathlib import Path

# .env is in the same directory as this config file (app folder)
_CONFIG_DIR = Path(__file__).resolve().parent
_ENV_PATH = _CONFIG_DIR / ".env"

def _read_key_from_env_file():
    """Read weather_api_key directly from .env if present (handles BOM, dotenv disabled)."""
    if not _ENV_PATH.exists():
        return None
    try:
        raw = _ENV_PATH.read_text(encoding="utf-8-sig").strip()  # utf-8-sig strips BOM
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"weather_api_key\s*=\s*(.+)", line, re.IGNORECASE)
            if m:
                value = m.group(1).strip().strip('"').strip("'")
                return value if value else None
    except Exception:
        pass
    return None

# 1) Load .env into os.environ (override=True so .env always wins over existing env)
if _ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV_PATH, override=True)
    except ImportError:
        pass

# 2) Prefer env vars, then direct .env read (for BOM / PYTHON_DOTENV_DISABLED)
weather_api_key = (
    os.environ.get("weather_api_key") or
    os.environ.get("WEATHER_API_KEY") or
    _read_key_from_env_file() or
    ""
)
weather_api_key = (weather_api_key or "").strip()