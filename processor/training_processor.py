# -*- coding: utf-8 -*-
"""
Copyright 2020-2022 Spiretech.co, Inc. All rights reserved.
Licenses: LICENSE.md
Description: AI Model processor component. This is a special component having an AI based predictions processor.

Reference: 


Utility class for loading daat from database and processing it as per requirment of AI model

Reference
- 

# Change History

"""
import os
from unicodedata import decimal
import numpy as np
import pandas as pd
import signal
import timeit
import shutil
import multiprocessing
from pathlib import Path
from datetime import date, datetime
from utils.configloader import ConfigUtil
from utils.logging_init              import logging
from processor.time_series_forecasting import Timeseries_forecasting_using_lstm
from utils.mssqlutil import MSSQLUtil
import db.constants as pr_constant

class TrainingProcessorUtil:
    """
    Utility class for loading and processing data from database
    """
    __instance  = None

    @staticmethod
    def getInstance():
        if TrainingProcessorUtil.__instance == None:
            TrainingProcessorUtil()
        return TrainingProcessorUtil.__instance
    
    def __init__(self):
        """
        Constructor for initializing the file loader isntance
        """
        if TrainingProcessorUtil.__instance == None:
            TrainingProcessorUtil.__instance = self
            self.__run()

    def __run(self):
        """
        Load configurations
        """
        instance = ConfigUtil.getInstance()
        self.__envConfiguration = instance.configJSON
        logging.info("training processor inititated...")
 
    def init_train(self, day):
        try:
            logging.info('Triggering Training')
            if date.today().day == day:
                logging.info('Initiating Training on {0}'.format(date.today()))
   
                base_directory = self.__envConfiguration['dir.modelBaseLocation']
                features = ['AllocatedAmount']
                dimensions = ['CustomerSeries']
                dimension_cols = ['CustomerSeries']
                dtypes = {'Week': 'timestamp', 'CustomerSeries': 'numeric', 'AllocatedAmount': 'numeric', 'Spending': 'numeric'}


                if self.__envConfiguration['dir.modelType'] == 'single':
                    envDimensions = ['dummy']

                for dimension_name in envDimensions:
                    #-------------------------------------------
                    sqlInstance = MSSQLUtil.getInstance()
                    cat_list    = sqlInstance.executeQuery(self.__envConfiguration['db.queryCatTypes'])
                    np_cat_list = np.array(cat_list)
                    cat_df = pd.DataFrame (np_cat_list, columns = ['CategoryTypeCode', 'CategoryTypeName'])
                    cat_df['Code'] = list(range(1, len(cat_df)+1 ,1))
                    cat_df['Code'] = cat_df['Code'].map("{:02}".format)
                    cat_df = cat_df.astype(str)
                    #-------------------------------------------
                    obj = Timeseries_forecasting_using_lstm(base_directory,  features, dtypes,self.__envConfiguration,
                                                            dimension_name=dimension_name, dimension_cols=dimension_cols, cat_map=cat_df)
                    #Call which needed
                    obj.split_dataset()

                    no_of_models = 3
                    start = timeit.default_timer()

                    # =========================================== WITH MULTIPROCESSING ==============================================
                    trials = [no_of_models // 3 + (1 if x < no_of_models % 3 else 0) for x in range(3)]
                    model_types = list(zip(['one'], trials))
                    num_workers = 3
                    logging.info(f'Model Types: {model_types}')

                    multiprocessing.set_start_method('spawn', force=True)
                    pool = multiprocessing.Pool(num_workers, self.init_worker)

                    scores = pool.map(obj.train_model, model_types)
                    logging.info(f"Score of model {scores}")
                    model_dict = {}

                    for score in scores:
                        model_dict[score[0]] = score[1]

                    best_model = min(model_dict, key=model_dict.get)
                    best_model_rmse = round(model_dict[best_model], 2)
                    logging.info(f'Best Model: {best_model}, Best Model rmse: {best_model_rmse}')
                    # saving model results in database log table
                    sqlInstance = MSSQLUtil.getInstance()
                    sqlParams = [ self.__envConfiguration['dir.modelName'], datetime.today(), best_model_rmse]
                    sqlInstance.transactionQuery(pr_constant.INSERT_AI_TRAIN_LOG_SQL, sqlParams)
                    logging.info("Best model results saved in db.")
                    model_no = best_model.split('_')[1]
                    best_model_path = os.path.join(obj.save_directory, 'best_models', f'best_model_proc_{model_no}.h5')

                    deployed_model = os.path.join(obj.save_directory, '')

                    Path(deployed_model).mkdir(parents=True, exist_ok=True)
                    deployed_model_path = os.path.join(deployed_model, f'deployed_model.h5')

                    if os.path.exists(deployed_model_path):
                        os.remove(deployed_model_path)
                        
                    Path(best_model_path).rename(deployed_model_path)
                    path = os.path.join(obj.save_directory, 'best_models')
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    path = os.path.join(obj.save_directory, 'bayseian_opt_proc_1')
                    if os.path.exists(path):
                        shutil.rmtree(path)
                    
                    return True
            else:
                logging.info('Training not initiated on {0}'.format(date.today()))
                return

        except Exception as e:
            logging.error(e)
 
 
    def init_worker(self):
        ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
        signal.signal(signal.SIGINT, signal.SIG_IGN)
            


