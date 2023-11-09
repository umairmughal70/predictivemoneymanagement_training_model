# -*- coding: utf-8 -*-
"""
Copyright 2020-2022 Spiretech.co, Inc. All rights reserved.
Licenses: LICENSE.md
Description: AI Model processor component. This is a special component having an AI based predictions processor.

Reference: https://spiretech.atlassian.net/wiki/spaces/PA/pages/592183297/HE-FUC-015+Authorization


Utility class for loading AI model and its meta data from directory

Reference
- https://www.geeksforgeeks.org/getter-and-setter-in-python/

# Change History

"""
import os
import sys
import pickle
from utils.logging_init              import logging


class FileLoaderUtil:
    """
    Utility class for loading files from a specific directory
    """
    __instance  = None
    __sqlCxn    = None
    _sqlCursor  = None

    @staticmethod
    def getInstance():
        if FileLoaderUtil.__instance == None:
            FileLoaderUtil()
        return FileLoaderUtil.__instance
    
    def __init__(self):
        """
        Constructor for initializing the file loader isntance
        """
        if FileLoaderUtil.__instance == None:
            FileLoaderUtil.__instance = self
            self.__loader()

    def __loader(self):
        """
        Load file from directory
        """
        logging.info("loading file from directory...")

    def loadFile(self, filePath, fileName):
        """
        Execute the file loading process
        """
        try:
            #Loading pickle object used in training to get CategoryTypeCode Mapping

            with open(os.path.join(os.path.dirname(__file__), filePath, fileName), 'rb') as file:
                meta_data = pickle.load(file)

            return meta_data

        except Exception as ex:
            logging.error(f"Error while loading file ")
            logging.error(ex)
            sys.exit('Error while loading file from directory')
            
    def writeFile(self, meta_data,  filePath, fileName):
        """
        Execute the file loading process
        """
        try:
            #Loading pickle object used in training to write data in file
            with open(os.path.join(filePath, fileName), 'wb') as file:
                pickle.dump(meta_data, file)
                file.close()
            return True

        except Exception as ex:
            logging.error(f"Error while loading file ")
            logging.error(ex)
            sys.exit('Error while loading file from directory')