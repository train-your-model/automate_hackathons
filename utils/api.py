# --------
# IMPORTS
# --------
from abc import ABC, abstractmethod
from typing import Dict, Type
import os
import yaml
from pathlib import Path
from urllib.parse import urlparse
import json
import subprocess

class CompAPI(ABC):
    def __init__(self, url, dest_folder, yaml_filepath, site_abv):
        self.url = str(url)
        self.dest_folder = dest_folder
        self.y_path = yaml_filepath
        self.site_abv = site_abv

    @abstractmethod
    def dump_token_path(self):
        pass

    @abstractmethod
    def run(self):
        pass

# =====================
# REGISTRY + DECORATOR
# =====================


API_REGISTRY: Dict[str, Type[CompAPI]] = {}


def register_api(site_abv: str):
    """
    Decorator to auto-register site API implementations.
    """
    def wrapper(cls: Type[CompAPI]):
        API_REGISTRY[site_abv.upper()] = cls
        return cls
    return wrapper

# =======================
# YAML BOOTSTRAP UTILITY
# =======================
def ensure_api_entry(yaml_path, site_abv):
    """
    Ensures:

    api:
        <site_abv>: null
    """
    site_abv = site_abv.upper()

    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    config.setdefault("api", {})

    if site_abv not in config["api"]:
        config["api"][site_abv] = None

        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

# ==============================
# SITE-SPECIFIC IMPLEMENTATIONS
# =============================
@register_api("KG")
class KaggleAPI(CompAPI):
    """
    This class performs the following tasks:
        1. Setting the environment variables from a JSON file
        2. Extract the competition name from the URL
        3. Run the subprocess to initiate downloading
    """

    def __init__(self, url, dest_folder, yaml_filepath, site_abv):
        super().__init__(url, dest_folder, yaml_filepath, site_abv)

    def dump_token_path(self):

        with open(self.y_path, "r") as f:
            data = yaml.safe_load(f)

        if data.get("api", {}).get("KG") is not None:
            pass

        else:
            ensure_api_entry(yaml_path=self.y_path, site_abv=self.site_abv)

            func_dir = Path(__file__).resolve().parents[1]
            yaml_file_path = os.path.join(func_dir, "files", ".kaggle", "kaggle.json")
            data.setdefault("api", {})["KG"] = yaml_file_path
            with open(self.y_path, "w") as y:
                yaml.safe_dump(data, y, sort_keys=False)

    def run(self):
        print("Running Kaggle API workflow")
        self.dump_token_path()
        with open(self.y_path, 'r') as f:
            data = yaml.safe_load(f)
            json_path = data["api"]["KG"]
            with open(json_path, "r") as j:
                creds = json.load(j)

        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']

        path_parts = urlparse(self.url).path.strip('/').split('/')
        if len(path_parts) < 2 or path_parts[0] != "competitions":
            raise ValueError("Invalid Kaggle competition URL")

        subprocess.run(
            f"kaggle competitions download -c {path_parts[1]} -p {self.dest_folder}",
            check=True, shell=True
        )

        return self



# ==============
# EXECUTOR CLASS
# ==============
class ApiRunner:
    """
    Resolves and Executes a site-specific competition API
    """
    def __init__(self, site_abv, yaml_path, dest_folder, url):
        self.site_abv = site_abv
        self.url = url
        self.dest_folder = dest_folder
        self.y_path = yaml_path

    def run(self):
        ensure_api_entry(self.y_path, self.site_abv)
        api_cls = API_REGISTRY.get(self.site_abv.upper())

        if not api_cls:
            raise ValueError(f"No API registered for '{self.site_abv}'")

        api = api_cls(
            url=self.url,
            dest_folder=self.dest_folder,
            yaml_filepath=self.y_path,
            site_abv=self.site_abv
        )

        return api.run()