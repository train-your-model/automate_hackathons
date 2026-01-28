# --------
# IMPORTS
# --------
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

            with open(full_path, "w") as f:
                json.dump({}, f, indent=2)

            return full_path

    def get_templates_names_file_path(self):
        """
        :return: The path for the created json file for templates names
        """
        if self.os_type == "Windows":
            func_dir = Path(__file__).resolve().parents[1]
            full_path = os.path.join(func_dir, "files", "templates_names.json")

            with open(full_path, "w") as f:
                json.dump({}, f, indent=2)

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
            "site_dirs":{},
            "templates_dir":{}
        }
        func_dir = Path(__file__).resolve().parents[1]
        config_file_path = os.path.join(func_dir, "files", "config.yaml")
        with open(config_file_path, "w") as f:
            yaml.safe_dump(config, f)

    def run(self):
        self.paths_dump()
        return self
