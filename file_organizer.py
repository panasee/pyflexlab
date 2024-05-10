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

def print_help_if_needed(func: callable) -> callable:
    """decorator used to print the help message if the first argument is '-h'"""
    def wrapper(self,measurename_all, *var_tuple, **kwargs):
        if var_tuple[0] == "-h":
            measure_name,_ = FileOrganizer.measurename_decom(measurename_all)
            print(FileOrganizer.query_namestr(measure_name))
            return None
        return func(self, measurename_all,*var_tuple, **kwargs)
    return wrapper

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

    def __init__(self, proj_name:str, copy_from:str = None, special_mode = False)->None:
        """
        initialize the class with the project name and judge if the name is in the accepted project names. Only out_database_path is required, as the local_database_dir is attached with the base_dir

        Args:
            proj_name: str
                The name of the project, used as the name of the base directory
        """
        ##TODO: add a special mode to allow the user to create a project without the need of the out_database_dir, store the data directly in the local_database_dir
        if FileOrganizer.out_database_dir is None:
            raise ValueError("The out_database_dir has not been set, please call the out_database_init method first.")
        # defined vars for two databases of the project
        self.out_database_dir_proj = FileOrganizer.out_database_dir / proj_name
        self.proj_name = proj_name

        # try to find the project in the record file, if not, then add a new item in record
        if proj_name not in FileOrganizer.proj_rec_json and copy_from is None:
            FileOrganizer.proj_rec_json[proj_name] = {
                "created_date": today.strftime("%Y-%m-%d"),
                "last_modified": today.strftime("%Y-%m-%d"), 
                "measurements": [], 
                "plan": {}}
            print(f"{proj_name} is not found in the project record file, a new item has been added.")
            # not dump the json file here, but in the sync method, to avoid the file being dumped multiple times
        elif proj_name not in FileOrganizer.proj_rec_json and copy_from is not None:
            if copy_from not in FileOrganizer.proj_rec_json:
                print(f"{copy_from} is not found in the project record file, please check the name.")
                return
            FileOrganizer.proj_rec_json[proj_name] = FileOrganizer.proj_rec_json[copy_from].copy()
            FileOrganizer.proj_rec_json[proj_name]["created_date"] = today.strftime("%Y-%m-%d")
            FileOrganizer.proj_rec_json[proj_name]["last_modified"] = today.strftime("%Y-%m-%d")
            print(f"{proj_name} has been copied from {copy_from}.")

        # create project folder in the out database for storing main data
        self.out_database_dir_proj.mkdir(exist_ok=True)
        if not os.path.exists(self.out_database_dir_proj / "assist_post.ipynb"):
            shutil.copy(FileOrganizer.local_database_dir / "assist.ipynb", self.out_database_dir_proj / "assist_post.ipynb")
        if not os.path.exists(self.out_database_dir_proj / "assist_measure.ipynb"):
            shutil.copy(FileOrganizer.local_database_dir / "assist.ipynb", self.out_database_dir_proj / "assist_measure.ipynb")
        # sync the project record file at the end of the function
        FileOrganizer._sync_json("proj_rec")

    def __del__(self) -> None:
        """Make sure the files are closed when the class is deleted."""
        if not FileOrganizer.__measure_type_file.closed:
            FileOrganizer.__measure_type_file.close()
 
    def get_filepath(self, measure_name_all: str, *var_tuple, tmpfolder: str=None) -> Path:
        """
        Get the filepath of the measurement file.

        Args:
            measure_name: str
                The name of the measurement type
            var_tuple: Tuple[int, str, float]
                a tuple containing all parameters for the measurement
        """
        measure_name,measure_sub = FileOrganizer.measurename_decom(measure_name_all)

        try:
            if measure_sub is None:
                filename = FileOrganizer.filename_format(FileOrganizer.measure_types_json[measure_name], *var_tuple)
            else:
                filename = FileOrganizer.filename_format(FileOrganizer.measure_types_json[measure_name][measure_sub], *var_tuple)

            if tmpfolder is not None:
                filepath = self.out_database_dir_proj / measure_name / tmpfolder / filename
            else:
                filepath = self.out_database_dir_proj / measure_name / filename
            return filepath

        except Exception:
            print("Wrong parameters, please ensure the parameters are correct.")
            FileOrganizer.query_namestr(measure_name)
            return None

    @staticmethod
    def measurename_decom(measurename_all: str) -> tuple[str]:
        """this method will decompose the measurename string into a tuple of measurename and submeasurename(None if not exist)"""
        measure_name_list = measurename_all.split("__")
        if len(measure_name_list) > 2:
            raise ValueError("The measurename string is not in the correct format, please check.")
        if_sub = (len(measure_name_list) == 2)
        measure_name = measure_name_list[0]
        measure_sub = measure_name_list[1] if if_sub else None
        return (measure_name, measure_sub)

    @staticmethod
    def filename_format(name_str: str, *var_tuple) -> str:
        """This method is used to format the filename"""
        # Extract variable names from the format string
        var_names = re.findall(r'{(\w+)}', name_str)
        # Create a dictionary that maps variable names to values
        var_dict = dict(zip(var_names, var_tuple))
        # Substitute variables into the format string
        return name_str.format(**var_dict)

    @staticmethod
    def query_namestr(measure_name: str) -> str | None:
        """
        This method is for querying the naming string of a certain measure type
        """
        if measure_name in FileOrganizer.measure_types_json:
            if isinstance(FileOrganizer.measure_types_json[measure_name], str):
                var_names = re.findall(r'{(\w+)}', FileOrganizer.measure_types_json[measure_name])
                print(FileOrganizer.measure_types_json[measure_name])
                print(var_names)
                return None
            elif isinstance(FileOrganizer.measure_types_json[measure_name], dict):
                for key, value in FileOrganizer.measure_types_json[measure_name].items():
                    var_names = re.findall(r'{(\w+)}', value)
                    print(f"{key}: {value}")
                    print(var_names)
                return None
        else:
            print("measure type not found, please add it first")
            return None

    @staticmethod
    def out_database_init(out_database_path: str | Path) -> None:
        """
        Set the out_database_dir variable to the given path, should be called before any instances of the class are created
        """
        FileOrganizer.out_database_dir = Path(out_database_path)
        FileOrganizer.out_database_dir.mkdir(parents=True,exist_ok=True)
        FileOrganizer.trash_dir = FileOrganizer.out_database_dir / "trash"
        FileOrganizer.trash_dir.mkdir(exist_ok=True)
        with open(FileOrganizer.out_database_dir / "project_record.json", "r", encoding="utf-8") as __proj_rec_file:
            FileOrganizer.proj_rec_json = json.load(__proj_rec_file)

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
            with open(FileOrganizer.out_database_dir / "project_record.json", "w", encoding="utf-8") as __proj_rec_file:
                json.dump(FileOrganizer.proj_rec_json, __proj_rec_file, indent=4)

    def create_folder(self, folder_name: str) -> None:
        """
        create a folder in the project folder

        Args:
            folder_name: str
                The name(relative path if not in the root folder) of the folder to be created
        """
        (self.out_database_dir_proj / folder_name).mkdir(exist_ok=True)

    def add_measurement(self, measurename_all: str) -> None:
        """
        Add a measurement to the project record file.

        Args:
            measure_name: str
                The name of the measurement(not with subcat) to be added, preferred to be one of current measurements, if not then use “add_measurement_type” to add a new measurement type first
        """
        measurename_main, _ = FileOrganizer.measurename_decom(measurename_all)
        # first add it into the project record file
        if measurename_main in FileOrganizer.measure_types_json:
            if measurename_main in FileOrganizer.proj_rec_json[self.proj_name]["measurements"]:
                print(f"{measurename_main} is already in the project record file.")
                return
            FileOrganizer.proj_rec_json[self.proj_name]["measurements"].append(measurename_main)
            FileOrganizer.proj_rec_json[self.proj_name]["last_modified"] = today.strftime("%Y-%m-%d")
            print(f"{measurename_main} has been added to the project record file.")
        else:
            print(f"{measurename_main} is not in the measure type file, please add it first.")
            return

        # add the measurement folder if not exists
        self.create_folder(measurename_main)
        print(f"{measurename_main} folder has been created in the project folder.")
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
            if plan_item not in FileOrganizer.proj_rec_json[self.proj_name]["plan"][plan_title]:
                FileOrganizer.proj_rec_json[self.proj_name]["plan"][plan_title].append(plan_item)
                print(f"plan is added to {plan_title}")
            else:
                print(f"{plan_item} is already in the plan.")
        else:
            FileOrganizer.proj_rec_json[self.proj_name]["plan"][plan_title] = [plan_item]
            print(f"{plan_title} has been added to the project record file.")
        # sync the measure type file
        FileOrganizer._sync_json("proj_rec")

    @staticmethod
    def add_measurement_type(measure_name_all: str, name_str: str, overwrite: bool = False) -> None:
        """
        Add a new measurement type to the measure type file.

        Args:
            measure_name: str
                The name(whole with subcat) of the measurement type to be added
            name_str: str
                The name string of the naming rules in this measurement type, use dict when there are many subtypes in the measurement type
            overwrite: bool
                Whether to overwrite the existing measurement type, default is False
        """
        measure_name,measure_sub = FileOrganizer.measurename_decom(measure_name_all)

        if measure_name in FileOrganizer.measure_types_json:
            if measure_sub in FileOrganizer.measure_types_json[measure_name] and not overwrite:
                print(f"{measure_name} is already in the measure type file.")
            else:
                FileOrganizer.measure_types_json[measure_name][measure_sub] = name_str
                print(f"{measure_name} has been added to the measure type file.")
        else:
            FileOrganizer.measure_types_json[measure_name] = {measure_sub: name_str}
            print(f"{measure_name} has been added to the measure type file.")
        # sync the measure type file
        FileOrganizer._sync_json("measure_type")

    def query_proj(self) -> dict:
        """
        Query the project record file to find the project.
        """
        return FileOrganizer.proj_rec_json[self.proj_name]

    @staticmethod
    def query_proj_all() -> dict:
        """
        Query the project record file to find all the projects.
        """
        return FileOrganizer.proj_rec_json

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
    FileOrganizer.out_database_init(r"C:\Users\Downloads\testtmp")
    test = FileOrganizer("test")
    FileOrganizer.add_measurement_type("RT","I-_iin_-_iout_-_currstr_-Vup-_v1high_-_v1low_-Vdown-_v2high_-v2low_-_temp1str_-temp2str__fileappen_")
    test.add_measurement("RT")
    test.tree()
    test.del_proj("test")
