#!/usr/bin/env python

"""
This module is responsible for managing the measure-related folders and data
"""
from common.file_organizer import FileOrganizer

class MeasureManager(FileOrganizer):
    """This class is a subclass of FileOrganizer and is responsible for managing the measure-related folders and data"""

    def __init__(self, proj_name: str) -> None:
        """Note that the FileOrganizer.out_database_init method should be called to assign the correct path to the out_database attribute. This method should be called before the MeasureManager object is created."""
        super().__init__(proj_name) # Call the constructor of the parent class

    def measure_init(self, measure_name: str) -> None:
        pass
    
    def Nonlinear(self, paras) -> None:
        pass