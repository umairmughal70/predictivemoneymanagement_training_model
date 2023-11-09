# -*- coding: utf-8 -*-
"""
Copyright 2020-2022 Spiretech.co, Inc. All rights reserved.
Licenses: LICENSE.md
Description: Utility file for managing configurations
Reference: https://spiretech.atlassian.net/wiki/spaces/PA/pages/592183297/HE-FUC-015+Authorization

# Reference:
- https://developers.google.com/protocol-buffers/docs/proto3
-

# Change History
[23042022/MFB/SIN-416] - Authentication and Authorization
"""

from environs               import Env
from marshmallow.validate import Length
from utils.logging_init          import logging

class ConfigUtil:

    __instance      = None
    configJSON     = {
        ## Load all configurations here        
        "db.host"                   : "", # PINO_ACCADAP_DB_SERVER
        "db.port"                   : "", # PINO_ACCADAP_DB_PORT
        "db.name"                   : "", # PINO_ACCADAP_DB_NAME
        "db.user"                   : "", # PINO_ACCADAP_DB_USERNAME
        "db.password"               : "", # PINO_ACCADAP_DB_PASSWORD
        "db.aiDbName"               : "", # PINO_ACCADAP_AI_DB_NAME
        "db.aiTrainLogTableName"    : "", # PINO_ACCADAP_TRAIN_LOG_TABLE_NAME
        "db.budgetWeeklyQuery"      : "", # PINO_ACCADAP_QUERY_BUDGETWEEKLY
        "db.budgetTable"            : "", # PINO_ACCADAP_TABLE_NAME
        "db.tunrcateBudgtTableQry"  : "", # PINO_ACCADAP_TRUNC_TABLE_QUERY
        "db.queryBudgetTrans"       : "", # PINO_ACCADAP_QUERY_BUDGETTRANS
        "db.queryBudget"            : "", # PINO_ACCADAP_QUERY_BUDGET
        "db.queryCatTypes"          : "", # PINO_ACCADAP_QUERY_CATTYPES
        "db.trainDay"               : "", # PINO_ACCADAP_TRAIN_DAY
        "db.generateDataDay"        : "", # PINO_ACCADAP_GENERATE_DATA_DAY
        "db.stmttimeout"            : "", # PINO_ACCADAP_DB_STMT_TIMEOUT
        "db.connTimeout"            : "", # PINO_ACCADAP_DB_CONN_TIMEOUT
        "dir.modelBaseLocation"     : "", # PINO_ACCADAP_DATA_PATH
        "dir.modelName"             : "", # PINO_ACCADAP_MODEL_NAME
        "dir.dateCol"               : "", # PINO_ACCADAP_DATE_COLUMN
        "dir.itemCol"               : "", # PINO_ACCADAP_ITEM_COLUMN
        "dir.targetCol"             : "", # PINO_ACCADAP_TARGET_COLUMN
        "dir.frequency"             : "", # PINO_ACCADAP_FREQUENCY
        "dir.features"              : "", # PINO_ACCADAP_FEATURES
        "dir.forecastHorizon"       : "", # PINO_ACCADAP_FORECAT_HORIZAN
        "dir.lookBack"              : "", # PINO_ACCADAP_LOOK_BACK
        "dir.splitType"             : "", # PINO_ACCADAP_SPLIT_TYPE
        "dir.useValidation"         : "", # PINO_ACCADAP_USE_VALIDATION
        "dir.modelType"             : "", # PINO_ACCADAP_MODEL_TYPE
        "dir.dimensions"            : "", # PINO_ACCADAP_DIMENSIONS
        "dir.dimensionsCol"         : "", # PINO_ACCADAP_DIMENSION_COLS
        "dir.dataTypes"             : "", # PINO_ACCADAP_DATATYPES
        "dir.monthsBack"            : "", # PINO_ACCADAP_MONTHS_BACK
        "dir.leastMonths"           : "", # PINO_ACCADAP_LEAST_MONTHS
        "dir.totalMonths"           : "", # PINO_ACCADAP_TOTAL_MONTHS
        # "dir.trainTime"             : "", # PINO_ACCADAP_TRAIN_TIME
    }
        
    @staticmethod
    def getInstance():
        """ Static access method for gettign the class isntance """
        if ConfigUtil.__instance == None:
            ConfigUtil()
        return ConfigUtil.__instance        

    def __init__(self):
        if ConfigUtil.__instance == None:
            ConfigUtil.__instance = self            
            self.__loadEnvironmentVariables()            

    def __loadEnvironmentVariables(self):
        """ Loads environment variables in configJSON dictionary """
        logging.info("Loading environment variables...")
        env = Env()
        env.read_env()
        # Database Related Variablees
        self.configJSON['db.host']                      = env.str('PINO_ACCADAP_DB_SERVER',                 validate=[Length(min=1,max=128, error='Invalid host name. Should not be > 128 chars')])
        self.configJSON['db.port']                      = env.str('PINO_ACCADAP_DB_PORT',                   validate=[Length(min=4,max=5,   error ='Invalid database port. Should not be > 5 chars')])
        self.configJSON['db.name']                      = env.str('PINO_ACCADAP_DB_NAME',                   validate=[Length(min=1,max=128, error ='Invalid database name. Should not be > 128 chars')])
        self.configJSON['db.user']                      = env.str('PINO_ACCADAP_DB_USERNAME',               validate=[Length(min=1,max=128, error ='Invalid database username, Should not be > 128 chars')])
        self.configJSON['db.password']                  = env.str('PINO_ACCADAP_DB_PASSWORD',               validate=[Length(min=1,max=128, error ='Invalid database password. Should not be > 128 chars')])
        self.configJSON['db.aiDbName']                  = env.str('PINO_ACCADAP_AI_DB_NAME',                validate=[Length(min=1,max=128, error ='Invalid AI database name. Should not be > 128 chars')])
        self.configJSON['db.aiTrainLogTableName']       = env.str('PINO_ACCADAP_TRAIN_LOG_TABLE_NAME',      validate=[Length(min=1,max=128, error ='Invalid AI train log table name. Should not be > 128 chars')])
        self.configJSON['db.budgetWeeklyQuery']         = env.str('PINO_ACCADAP_QUERY_BUDGETWEEKLY',        validate=[Length(min=1,         error ='Invalid weekly budget query. Should not be < 0 chars')])
        self.configJSON['db.budgetTable']               = env.str('PINO_ACCADAP_TABLE_NAME',                validate=[Length(min=1,max=128, error ='Invalid table name length, Should not be > 128 chars')])
        self.configJSON['db.tunrcateBudgtTableQry']     = env.str('PINO_ACCADAP_TRUNC_TABLE_QUERY',         validate=[Length(min=1,max=256, error ='Invalid table name length, Should not be > 128 chars')])
        self.configJSON['db.queryBudgetTrans']          = env.str('PINO_ACCADAP_QUERY_BUDGETTRANS',         validate=[Length(min=1,         error ='Invalid budget transactions query. Should not be < 0 chars')])
        self.configJSON['db.queryBudget']               = env.str('PINO_ACCADAP_QUERY_BUDGET',              validate=[Length(min=1,         error ='Invalid budget query. Should not be < 0 chars')])
        self.configJSON['db.queryCatTypes']             = env.str('PINO_ACCADAP_QUERY_CATTYPES',            validate=[Length(min=1,         error ='Invalid category type query. Should not be < 0 chars')])
        self.configJSON['db.trainDay']                  = env.str('PINO_ACCADAP_TRAIN_DAY',                 validate=[Length(min=1,max=3,   error ='Invalid training day value. Should not be > 3 chars')])
        self.configJSON['db.generateDataDay']           = env.str('PINO_ACCADAP_GENERATE_DATA_DAY',         validate=[Length(min=1,max=3,   error ='Invalid data generation day value. Should not be > 3 chars')])
        self.configJSON['db.stmttimeout']               = env.str('PINO_ACCADAP_DB_STMT_TIMEOUT',           validate=[Length(min=1,max=5,   error ='Invalid statement timeout. Should not be > 5 chars')])
        self.configJSON['db.connTimeout']               = env.str('PINO_ACCADAP_DB_CONN_TIMEOUT',           validate=[Length(min=1,max=5,   error ='Invalid connection timeout. Should not be > 5 chars')])
        self.configJSON['dir.modelBaseLocation']        = env.str('PINO_ACCADAP_DATA_PATH',                 validate=[Length(min=1,max=500, error ='Invalid data path. Should not be > 128 chars')])
        self.configJSON['dir.modelName']                = env.str('PINO_ACCADAP_MODEL_NAME',                validate=[Length(min=1,max=128, error ='Invalid model name. Should not be > 128 chars')])
        self.configJSON['dir.dateCol']                  = env.str('PINO_ACCADAP_DATE_COLUMN',               validate=[Length(min=1,max=30,  error ='Invalid date column. Should not be > 30 chars')])
        self.configJSON['dir.itemCol']                  = env.str('PINO_ACCADAP_ITEM_COLUMN',               validate=[Length(min=1,max=30,  error ='Invalid item coumn. Should not be > 30 chars')])
        self.configJSON['dir.targetCol']                = env.str('PINO_ACCADAP_TARGET_COLUMN',             validate=[Length(min=1,max=30,  error ='Invalid target column. Should not be > 30 chars')])
        self.configJSON['dir.frequency']                = env.str('PINO_ACCADAP_FREQUENCY',                 validate=[Length(min=1,max=1,   error ='Invalid frequency. Should not be > 1 chars')])
        self.configJSON['dir.features']                 = env.str('PINO_ACCADAP_FEATURES',                  validate=[Length(min=1,         error ='Invalid features. Should not be < 0 chars')])
        self.configJSON['dir.forecastHorizon']          = env.str('PINO_ACCADAP_FORECAT_HORIZAN',           validate=[Length(min=1,max=1,   error ='Invalid forcast horizon. Should not be < 0 chars')])
        self.configJSON['dir.lookBack']                 = env.str('PINO_ACCADAP_LOOK_BACK',                 validate=[Length(min=1,max=1,   error ='Invalid look back. Should not be < 0 chars')])
        self.configJSON['dir.splitType']                = env.str('PINO_ACCADAP_SPLIT_TYPE',                validate=[Length(min=1,max=10,  error ='Invalid split type. Should not be > 10 chars')])
        self.configJSON['dir.useValidation']            = env.str('PINO_ACCADAP_USE_VALIDATION',            validate=[Length(min=1,         error ='Invalid use validation. Should not be < 0 chars ')])
        self.configJSON['dir.modelType']                = env.str('PINO_ACCADAP_MODEL_TYPE',                validate=[Length(min=1,max=30,  error ='Invalid model type. Should not be > 30 chars')])
        self.configJSON['dir.dimensions']               = env.str('PINO_ACCADAP_DIMENSIONS',                validate=[Length(min=1,         error ='Invalid dimensions. Should not be < 0 chars')])
        self.configJSON['dir.dimensionsCol']            = env.str('PINO_ACCADAP_DIMENSION_COLS',            validate=[Length(min=1,         error ='Invalid dimensions columns. Should not be < 0 chars')])
        self.configJSON['dir.dataTypes']                = env.str('PINO_ACCADAP_DATATYPES',                 validate=[Length(min=1,         error ='Invalid datatypes. Should not be < 0 chars')])
        self.configJSON['dir.monthsBack']               = env.str('PINO_ACCADAP_MONTHS_BACK',               validate=[Length(min=1,max=2,   error ='Invalid months back. Should not be > 2 chars')])
        self.configJSON['dir.leastMonths']              = env.str('PINO_ACCADAP_LEAST_MONTHS',              validate=[Length(min=1,max=2,   error ='Invalid least motnhs. Should not be > 2 chars')])
        self.configJSON['dir.totalMonths']              = env.str('PINO_ACCADAP_TOTAL_MONTHS',              validate=[Length(min=1,max=2,   error ='Invalid total months. Should not be > 2 chars')])
        # self.configJSON['dir.trainTime']                = env.str('PINO_ACCADAP_TRAIN_TIME',                validate=[Length(min=5,max=7,   error ='Invalid Train time. Should not be > 5 and < 7 chars')])

        # Write project specific configuration items below:
        # END        