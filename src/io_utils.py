from pathlib import Path
import json
import pandas as pd


def save_dataframe(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(filename, index=False)


def save_json(data, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)