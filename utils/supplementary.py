# ------------------
# IMPORTS
# ------------------

import os
import yaml
from pathlib import Path


# F1
def create_root_dir():
    """
    Creates the root directory in the parent directory where the hackathon specific dirs would
    be stored.
    """
    func_dir = Path(__file__).resolve().parents[1]
    y_path = os.path.join(func_dir, "files", "config.yaml")

    with open(y_path, 'r') as y:
        data = yaml.safe_load(y)

        par_dir = data["defaults"]["parent_dir"]
        root_dir_name = str(input("Enter the name for the root directory: "))
        full_path = os.path.join(par_dir, root_dir_name)
        os.mkdir(full_path)
        print("Root Directory has been created successfully!!")

        # Appending the Root Directory Path to the YAML file
        data.setdefault("defaults", {})["root_dir_path"] = full_path
        with open(y_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        print("YAML file has been updated with the ROOT Directory")
