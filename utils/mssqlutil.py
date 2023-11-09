# -*- coding: utf-8 -*-
"""
Copyright 2020-2022 Spiretech.co, Inc. All rights reserved.
Licenses: LICENSE.md
Description: Authorization processor component. This is a special component having a grpc based
            interface.
Reference: https://spiretech.atlassian.net/wiki/spaces/PA/pages/592183297/HE-FUC-015+Authorization


Utility class for managing connection with Redis Server

Reference
- https://www.geeksforgeeks.org/getter-and-setter-in-python/

# Change History
[08042022/MFB/SIN-416] - Initial commit
"""

import pyodbc
import sys
from utils.configloader         import ConfigUtil
from utils.logging_init              import logging


class MSSQLUtil:
    """
    Utility class for connecting with MSSQL server and perform required operation
    """
    __instance  = None
    __sqlCxn    = None
    _sqlCursor  = None

    @staticmethod
    def getInstance():
        if MSSQLUtil.__instance == None:
            MSSQLUtil()
        return MSSQLUtil.__instance
    
    def __init__(self):
        """
        Constructor for initializing the database connection
        """
        if MSSQLUtil.__instance == None:
            MSSQLUtil.__instance = self
            self.__connect()

    def __connect(self):
        """
        Connect to the database server
        """
        logging.info("Connecting with database server")
        connString = "Driver={driver};Server={hostname},{port};Database={dbName};UID={userID};PWD={password};autocommit=True;".format(
                                                                                                                                                                hostname=ConfigUtil.getInstance().configJSON["db.host"],                                                                                                                            
                                                                                                                                                                port=ConfigUtil.getInstance().configJSON["db.port"],
                                                                                                                                                                dbName=ConfigUtil.getInstance().configJSON["db.name"],
                                                                                                                                                                userID=ConfigUtil.getInstance().configJSON["db.user"],
                                                                                                                                                                password=ConfigUtil.getInstance().configJSON["db.password"],
                                                                                                                                                                driver='ODBC Driver 17 for SQL Server')

        logging.debug("Connection string: {}".format(connString))                                                                                                                                                            
        self.__sqlCxn = pyodbc.connect(connString)
        self.__sqlCxn.timeout = int(ConfigUtil.getInstance().configJSON["db.stmttimeout"])
        self._sqlCursor = self.__sqlCxn.cursor()
        logging.info("Connected with database server")

    def executeQuery(self, query, *parameters):
        """
        Execute the query and return the result
        @param query: Query to be executed
        """
        try:
            logging.debug("Executing query: {}".format(query))
            self._sqlCursor.execute(query, *parameters)
            return self._sqlCursor.fetchall()
        except pyodbc.Error as ex:
            sqlstate = ex.args[1]
            sqlstate = sqlstate.split(".")
            logging.error(f"Error while validating the transaction SQL State: {sqlstate} - SQL Error: {ex}")
            sys.exit('Error while executing query')
        except Exception as ex:
            logging.error(f"Error while validating the transaction ")
            logging.error(ex)
            sys.exit('Error while executing query')
        
    def transactionQuery(self, query, *parameters):
        """
        Execute the query and return the result
        @param query: Query to be executed
        """
        try:
            logging.debug("Executing insert query: {}".format(query))
            self._sqlCursor.execute(query, *parameters)
            self.__sqlCxn.commit()
            return True
        except pyodbc.Error as ex:
            sqlstate = ex.args[1]
            sqlstate = sqlstate.split(".")
            logging.error(f"Error while validating the transaction SQL State: {sqlstate} - SQL Error: {ex}")
            sys.exit('Error while executing query')
        except Exception as ex:
            logging.error(f"Error while validating the transaction ")
            logging.error(ex)
            sys.exit('Error while executing query')

    @property
    def sqlCursor(self):
        return self._sqlCursor
    
    @property
    def sqlCxn(self):
        return self.__sqlCxnBUDGET_SQL
        