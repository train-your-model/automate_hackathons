# ------------------
# IMPORTS
# ------------------
import json
import os
from pathlib import Path
import yaml


class NormOps:
    """
    For a normal operation, this function performs the following tasks:
        1. Checks for the presence of user defined site name in the site json file. If it is a new entry;
            1.1. Adds the site abbreviation and full site name into the site json file.
            1.2. Extracts and Creates the Site Specific Directory
            1.3. Extracts and Creates Project Specific Directory
    """

    # Parameters
    yaml_file_path = ""

    def __init__(self, site_abv):
        self.site_abv = site_abv
        self.site_name = ""
        self.new_site = 0

        self.site_json_path = ""
        self.templates_dir_path = ""

        self.proj_dir_path = ""

    def check_site_names(self):
        try:
            with open(self.site_json_path, "r") as f:
                data = json.load(f)

        except json.JSONDecodeError:
            with open(self.site_json_path, "w") as j:
                self.site_name = str(input("Enter the Full Name of the Hackathon Site: "))
                j.write(f"{str(self.site_abv).upper()}:{str(self.site_name).upper()}")
                self.new_site = 1

        else:
            if self.site_name not in data.keys():
                self.new_site = 1
                data.update({str(self.site_abv).upper():str(self.site_name).upper()})
                print("SITE JSON FILE has been updated with the New Site Name")

            else:
                self.site_name = data.get(self.site_abv)

    def get_paths(self):
        func_dir = Path(__file__).resolve().parents[1]
        yaml_file_path = os.path.join(func_dir, "files", "config.yaml")
        with open(yaml_file_path, 'r') as y:
            self.site_json_path = yaml.safe_load(y)["json_files"]["site_names"]
            self.templates_dir_path = yaml.safe_load(y)["json_files"]["templates_names"]

        NormOps.yaml_file_path = yaml_file_path

    def create_site_dir(self):
        """
        :return: Site Specific Directory is created. Path to that directory is also returned.
        """
        with open(NormOps.yaml_file_path, "r") as f:
            data = yaml.safe_load(f)
            root_dir = data["defaults"]["root_dir"]
            full_path = os.path.join(root_dir, self.site_name)
            os.mkdir(full_path)

            # Appending to the YAML file
            data.setdefault("site_dirs",{})[self.site_name] = full_path
            with open(NormOps.yaml_file_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)

        print("Site-specific folders have been created")

    def create_proj_dir(self, proj_dir_name):
        """
        :return: A Project Specific Directory, which would be equivalent to Working Directory
        """
        with open(NormOps.yaml_file_path, "r") as f:
            data = yaml.safe_load(f)
            site_path = data["site_dirs"][self.site_name]
            self.proj_dir_path = os.path.join(site_path, proj_dir_name)

    def run(self, proj_name, api=None):
        self.get_paths()
        self.check_site_names()

        if (self.new_site == 1) and (api is not None):
            self.create_site_dir()
            self.create_proj_dir(proj_dir_name=proj_name)

        elif (self.new_site == 0) and (api is None):
            self.create_proj_dir(proj_dir_name=proj_name)

        elif (self.new_site == 0) and (api is not None):
            self.create_proj_dir(proj_dir_name=proj_name)

        return self
