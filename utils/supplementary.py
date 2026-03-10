# ------------------
# IMPORTS
# ------------------
import os
import shutil
import numpy as np
import yaml
from pathlib import Path
from zipfile import ZipFile

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

# C1
class DealDataFolders:
    """
    This class performs the following tasks:
        1. Locate the Zip Folder in the Project Directory, Unzip, Extract the Contents, Remove the empty folder.
    """

    def __init__(self, project_dir: str, delete_zip: bool = False):
        self.project_dir = Path(project_dir).resolve()
        self.delete_zip = delete_zip

        if not self.project_dir.exists():
            raise FileNotFoundError(f"{self.project_dir} does not exist")

    def _zip_files(self):
        return list(self.project_dir.glob("*.zip"))

    def _extract_archives(self):
        for zip_file in self._zip_files():

            with ZipFile(zip_file, "r") as z:
                z.extractall(self.project_dir)

            if self.delete_zip:
                zip_file.unlink()

    def _structure(self):
        items = [
            p for p in self.project_dir.iterdir()
            if not p.name.endswith(".zip")
        ]

        files = [p for p in items if p.is_file()]
        dirs = [p for p in items if p.is_dir()]

        return files, dirs

    def _flatten(self, directory: Path):
        for item in directory.iterdir():
            dest = self.project_dir / item.name

            if dest.exists():
                raise FileExistsError(
                    f"Cannot move {item.name}, destination exists"
                )

            shutil.move(str(item), str(dest))

        directory.rmdir()

    def _normalize_structure(self):
        files, dirs = self._structure()

        # Files already independent → stop
        if files:
            return "Files already present at project root."

        # Single directory → flatten
        if len(dirs) == 1:
            self._flatten(dirs[0])
            return f"Flattened directory: {dirs[0].name}"

        # Ambiguous case
        if len(dirs) > 1:
            return "Multiple directories detected. Manual inspection required."

        return "No files detected."

    def run(self):

        if self._zip_files():
            self._extract_archives()

        return self._normalize_structure()