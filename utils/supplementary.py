# ------------------
# IMPORTS
# ------------------
import json
import platform
import os
import yaml
from pathlib import Path


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
            }
        }
        func_dir = Path(__file__).resolve().parents[1]
        config_file_path = os.path.join(func_dir, "files", "config.yaml")
        with open(config_file_path, "w") as f:
            yaml.safe_dump(config, f)
            f.close()

    def run(self):
        self.paths_dump()
        return self


def yaml_path_dump():
    func_dir = Path(__file__).resolve().parents[1]
    config_file_path = os.path.join(func_dir, "files", "config.yaml")

    # Creating a Blank YAML file
    with open(config_file_path, "w"):
        pass

    json_file_path = os.path.join(func_dir, "files", "config.json")

    # Dumping the file path to the JSON file
    with open(json_file_path, 'r') as f:
        state = json.load(f)
        if state['yaml_path'] is None:
            with open(json_file_path, "w") as d:
                json.dump({'yaml_path': config_file_path}, d)
                d.close()
            print("YAML Filepath has been updated successfully in the JSON file")
        f.close()


def create_config_json():
    """"
    Creates the JSON file which would save the paths for YAML file, API tokens
    """
    func_dir = Path(__file__).resolve().parents[1]
    json_file_path = os.path.join(func_dir, "files", "config.json")

    with open(json_file_path, "w") as f:
        json.dump({"yaml_path": None}, f, indent=2)
        f.close()
