from pathlib import Path
from typing import Dict, List, Optional, Sequence

print('hi')
import yaml

with open("config_copy.yml", "r") as stream:
    try:
        print(yaml.safe_load(stream)['features'])
    except yaml.YAMLError as exc:
        print(exc)
print()
