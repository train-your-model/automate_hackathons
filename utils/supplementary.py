# ------------------
# IMPORTS
# ------------------
import json
import platform
import os
import yaml
from pathlib import Path


# C1
class PathDump:

    def __init__(self):
        self.os_type = platform.system()

    def get_desktop_path(self):
        """
        The path for Desktop depending upon the OS version
        """
        if self.os_type == "Windows":
            full_path = os.path.join(Path(os.environ["USERPROFILE"]), "Desktop")
            return full_path

    def get_download_path(self):
        """
        The path for Desktop depending upon the OS version
        """
        if self.os_type == "Windows":
            full_path = os.path.join(Path(os.environ["USERPROFILE"]), "Downloads")
            return full_path

    def get_site_names_file_path(self):
        """
        :return: The path for the saved json file for site names
        """
        if self.os_type == "Windows":
            func_dir = Path(__file__).resolve().parents[1]
            full_path = os.path.join(func_dir, "files", "site_names.json")
            return full_path

    def get_templates_names_file_path(self):
        """
        :return: The path for the saved json file for templates names
        """
        if self.os_type == "Windows":
            func_dir = Path(__file__).resolve().parents[1]
            full_path = os.path.join(func_dir, "files", "templates_names.json")
            return full_path

    def get_templates_path(self):
        """
        :return: The path for the ipynb templates
        """
        if self.os_type == "Windows":
            func_dir = Path(__file__).resolve().parents[1]
            full_path = os.path.join(func_dir, "templates")
            return full_path

    def paths_dump(self):
        config = {
            "defaults":{
                "download_dir":self.get_download_path(),
                "parent_dir":self.get_desktop_path(),
                "templates_dir":self.get_templates_path()
            },
            "json_files":{
                "site_names":self.get_site_names_file_path(),
                "templates_names":self.get_templates_names_file_path()
            },
            "site_dirs":{}
        }
        func_dir = Path(__file__).resolve().parents[1]
        config_file_path = os.path.join(func_dir, "files", "config.yaml")
        with open(config_file_path, "w") as f:
            yaml.safe_dump(config, f)
            f.close()

    def run(self):
        self.paths_dump()
        return self

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
