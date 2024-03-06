#!/usr/bin/env python

"""
This file contains the functions to organize the files in the directory.
This file should be called when create new files or directories 
"""

import os
from pathlib import Path
import json
import datetime
from typing import Union
from itertools import islice
import shutil
import re

# set the workpath to the parent directory of the file "script-tools/" also preserve it as a global variable
script_base_dir: Path = Path(__file__).resolve().parents[1]
today = datetime.date.today()
os.chdir(script_base_dir)

class FileOrganizer:
    """A class to manage file and directory operations."""

    # define static variables to store the file paths
    local_database_dir = script_base_dir / "data_files"
    out_database_dir: Path = None # defined in out_database method(static)
    trash_dir: Path = None # defined in out_database method(static)
    # load the json files to dicts for storing important records information
    # take note that the dicts are static variables created with the definition of the class and shared by all instances of the class and keep changing
    measure_types_json: dict
    """the changes should ALWAYS be synced RIGHT AFTER EVERY CHANGES"""
    proj_rec_json: dict
    """the changes should ALWAYS be synced RIGHT AFTER EVERY CHANGES"""

    with open(local_database_dir / "measure_types.json", "r", encoding="utf-8") as __measure_type_file:
        measure_types_json: dict = json.load(__measure_type_file)
    with open(local_database_dir / "project_record.json", "r", encoding="utf-8") as __proj_rec_file:
        proj_rec_json: dict = json.load(__proj_rec_file)

    def __init__(self, proj_name:str)->None:
        """
        initialize the class with the project name and judge if the name is in the accepted project names. Only out_database_path is required, as the local_database_dir is attached with the base_dir

        Args:
            proj_name: str
                The name of the project, used as the name of the base directory
            out_database_path: str
                The ABSOLUTE path to the directory where the projects' main data has been or will be stored
        """
        if FileOrganizer.out_database_dir is None:
            raise ValueError("The out_database_dir has not been set, please call the out_database_init method first.")
        # defined vars for two databases of the project
        self.out_database_dir_proj = FileOrganizer.out_database_dir / proj_name
        self.proj_name = proj_name

        # try to find the project in the record file, if not, then add a new item in record
        if proj_name not in FileOrganizer.proj_rec_json:
            FileOrganizer.proj_rec_json[proj_name] = {
                "created_date": today.strftime("%Y-%m-%d"),
                "last_modified": today.strftime("%Y-%m-%d"), 
                "measurements": [], 
                "plan": {}}
            print(f"{proj_name} is not found in the project record file, a new item has been added.")
            # not dump the json file here, but in the write method, to avoid the file being dumped multiple times

        # create project folder in the out database for storing main data
        self.out_database_dir_proj.mkdir(exist_ok=True)
        # sync the project record file at the end of the function
        FileOrganizer._sync_json("proj_rec")

    def __del__(self) -> None:
        """Make sure the files are closed when the class is deleted."""
        if not FileOrganizer.__proj_rec_file.closed:
            FileOrganizer.__proj_rec_file.close()
        if not FileOrganizer.__measure_type_file.closed:
            FileOrganizer.__measure_type_file.close()
 
    @staticmethod
    def filename_format(name_str: str, *var_tuple) -> str:
        """This method is used to format the filename"""
        # Extract variable names from the format string
        var_names = re.findall(r'\{(\w+)\}', name_str)
        # Create a dictionary that maps variable names to values
        var_dict = dict(zip(var_names, var_tuple))
        # Substitute variables into the format string
        return name_str.format(**var_dict)

    @staticmethod
    def query_namestr(measure_name: str) -> str:
        """
        This method is for querying the naming string of a certain measure type
        """
        if measure_name in FileOrganizer.measure_types_json:
            return FileOrganizer.measure_types_json[measure_name]
        else:
            print("measure type not found, please add it first")
            return

    @staticmethod
    def out_database_init(out_database_path: str) -> None:
        """
        Set the out_database_dir variable to the given path, should be called before any instances of the class are created
        """
        FileOrganizer.out_database_dir = Path(out_database_path)
        FileOrganizer.out_database_dir.mkdir(parents=True,exist_ok=True)
        FileOrganizer.trash_dir = FileOrganizer.out_database_dir / "trash"
        FileOrganizer.trash_dir.mkdir(exist_ok=True)

    @staticmethod
    def _sync_json(which_file: str) -> None:
        """
        sync the json dictionary with the file, should avoid using this method directly, as the content of json may be uncontrolable

        Args:
            which_file: str
                The file to be synced with, should be either "measure_type" or "proj_rec"
        """
        if which_file == "measure_type":
            with open(FileOrganizer.local_database_dir / "measure_types.json", "w", encoding="utf-8") as __measure_type_file:
                json.dump(FileOrganizer.measure_types_json, __measure_type_file, indent=4)
        elif which_file == "proj_rec":
            with open(FileOrganizer.local_database_dir / "project_record.json", "w", encoding="utf-8") as __proj_rec_file:
                json.dump(FileOrganizer.proj_rec_json, __proj_rec_file, indent=4)

    def create_folder(self, folder_name: str) -> None:
        """
        create a folder in the project folder

        Args:
            folder_name: str
                The name(relative path if not in the root folder) of the folder to be created
        """
        (self.out_database_dir_proj / folder_name).mkdir(exist_ok=True)

    def add_measurement(self, measure_name: str) -> None:
        """
        Add a measurement to the project record file.

        Args:
            measure_name: str
                The name of the measurement to be added, preferred to be one of current measurements, if not then use “add_measurement_type” to add a new measurement type first
            measure_paras: Tuple[int, str, float]
                a tuple containing all parameters for the measurement
        """

        # first add it into the project record file
        if measure_name in FileOrganizer.measure_types_json:
            if measure_name in FileOrganizer.proj_rec_json[self.proj_name]["measurements"]:
                print(f"{measure_name} is already in the project record file.")
                return
            FileOrganizer.proj_rec_json[self.proj_name]["measurements"].append(measure_name)
            FileOrganizer.proj_rec_json[self.proj_name]["last_modified"] = today.strftime("%Y-%m-%d")
            print(f"{measure_name} has been added to the project record file.")
        else:
            print(f"{measure_name} is not in the measure type file, please add it first.")
            return

        # add the measurement folder if not exists
        self.create_folder(measure_name)
        print(f"{measure_name} folder has been created in the project folder.")
        # sync the project record file
        FileOrganizer._sync_json("proj_rec")

    def add_plan(self, plan_title: str, plan_item: str) -> None:
        """
        Add/Supplement a plan_item to the project record file. If the plan_title is already in the project record file, then supplement the plan_item to the plan_title, otherwise add a new plan_title with the plan_item. (each plan_item contains a list)

        Args:
            plan_title: str
                The title of the plan_item to be added
            plan_item: str
                The content of the plan
        """
        if plan_title in FileOrganizer.proj_rec_json[self.proj_name]["plan"]:
            FileOrganizer.proj_rec_json[self.proj_name]["plan"][plan_title].append(plan_item)
            print(f"plan is added to {plan_title}")
        else:
            FileOrganizer.proj_rec_json[self.proj_name]["plan"][plan_title] = [plan_item]
            print(f"{plan_title} has been added to the project record file.")

    @staticmethod
    def add_measurement_type(measure_name: str, name_str: Union[str, dict]) -> None:
        """
        Add a new measurement type to the measure type file.

        Args:
            measure_name: str or dict
                The name(s) of the measurement type to be added
            name_str: str or dict
                The name string of the naming rules in this measurement type, use dict when there are many subtypes in the measurement type
        """
        if measure_name in FileOrganizer.measure_types_json:
            print(f"{measure_name} is already in the measure type file.")
        else:
            FileOrganizer.measure_types_json[measure_name] = name_str
            print(f"{measure_name} has been added to the measure type file.")
        # sync the measure type file
        FileOrganizer._sync_json("measure_type")

    def query_proj(self) -> dict:
        """
        Query the project record file to find the project.
        """
        return FileOrganizer.proj_rec_json[self.proj_name]

    @staticmethod
    def del_proj(proj_name:str) -> None:
        """To delete a project from the project record file."""
        del FileOrganizer.proj_rec_json[proj_name]
        FileOrganizer._sync_json("proj_rec")
        #move the project folder to the trash bin
        shutil.move(FileOrganizer.out_database_dir / proj_name, FileOrganizer.trash_dir / proj_name)
        print(f"{proj_name} has been moved to the trash bin.")

    def tree(self, level: int=-1, limit_to_directories: bool=True, length_limit: int=300):
        """
        Given a directory Path object print a visual tree structure
        Cited from: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
        """
        # prefix components:
        space =  '    '
        branch = '│   '
        # pointers:
        tee =    '├── '
        last =   '└── '

        dir_path = self.out_database_dir_proj
        files = 0
        directories = 0
        def inner(dir_path: Path, prefix: str='', level=-1):
            nonlocal files, directories
            if not level:
                return # 0, stop iterating
            if limit_to_directories:
                contents = [d for d in dir_path.iterdir() if d.is_dir()]
            else: 
                contents = list(dir_path.iterdir())
            pointers = [tee] * (len(contents) - 1) + [last]
            for pointer, path in zip(pointers, contents):
                if path.is_dir():
                    yield prefix + pointer + path.name
                    directories += 1
                    extension = branch if pointer == tee else space 
                    yield from inner(path, prefix=prefix+extension, level=level-1)
                elif not limit_to_directories:
                    yield prefix + pointer + path.name
                    files += 1
        print(dir_path.name)
        iterator = inner(dir_path, level=level)
        for line in islice(iterator, length_limit):
            print(line)
        if next(iterator, None):
            print(f'... length_limit, {length_limit}, reached, counted:')
        print(f'\n{directories} directories' + (f', {files} files' if files else ''))

if __name__ == "__main__":
    FileOrganizer.out_database_init(r"C:\Users\Dongkai\Downloads\testtmp")
    test = FileOrganizer("test")
    FileOrganizer.add_measurement_type("RT","I-_iin_-_iout_-_currstr_-Vup-_v1high_-_v1low_-Vdown-_v2high_-v2low_-_temp1str_-temp2str__fileappen_")
    test.add_measurement("RT")
    test.tree()
    test.del_proj("test")
