# -*- coding: utf-8 -*-
"""
Copyright 2020-2022 Spiretech.co, Inc. All rights reserved.
Licenses: LICENSE.md
Description: Authorization processor component. This is a special component having a grpc based
            interface.
Reference: https://spiretech.atlassian.net/wiki/spaces/PA/pages/592183297/HE-FUC-015+Authorization

# Installation Guide

# Reference:
- https://developers.google.com/protocol-buffers/docs/proto3
- 
# Change History
"""
__author__  = "Adeel Yaqoob <adeel@spiretech.co>"
__version__ = "0.11"
__license__ = "Proprietary"

import sys
from utils.logging_init     import logging
from utils.configloader import ConfigUtil
from processor.weekly_data_processor import DataLoaderUtil
from processor.training_processor import TrainingProcessorUtil

import schedule
import time

class BootStart:

    __envConfiguration = None # Environment configuration object laoded from ConfigUtil

    """
    This is the main class for the authorization processor.
    """
    def __init__(self):
        """
        This is the constructor for the class.
        """                
        self.displayBanner()
        instance = ConfigUtil.getInstance()
        self.__envConfiguration = instance.configJSON
        self.displayBanner('ALL')

    def displayBanner(self,det=''):
        """ Display State of ENV variables"""        
        if det == 'ALL':              
            logging.info('')
        else:
            logging.info('*************************************')
            logging.info('┌─┐┌─┐┬┬─┐┌─┐┌┬┐┌─┐┌─┐┬ ┬ ┌─┐┌─┐')
            logging.info('└─┐├─┘│├┬┘├┤  │ ├┤ │  ├─┤ │  │ │')
            logging.info('└─┘┴  ┴┴└─└─┘ ┴ └─┘└─┘┴ ┴o└─┘└─┘')
            logging.info('*************************************')  

    def run(self):
        """ Start training on customer's available data """  
        dataLoader = DataLoaderUtil.getInstance()
        dataLoader.initGenerateData(int(self.__envConfiguration['db.generateDataDay']))
        
        trainingUtil = TrainingProcessorUtil.getInstance()
        trainingUtil.init_train(int(self.__envConfiguration['db.trainDay']))
        logging.info('Training completed, waiting for next training day')

        return True
        
        
if __name__ == "__main__":
    """ Main entry point. This is executed from command line """
    try:
        bootStart = BootStart()
        instance = ConfigUtil.getInstance()
        logging.info('Waiting for execution day...')
        # schedule.every().day.at(instance.configJSON['db.trainDay']).do(bootStart.run)
        # while True:
        #     schedule.run_pending()
        #     time.sleep(1)
        bootStart.run()
    except Exception as ex:
        logging.error('Error while processing the transactions')
        logging.error(ex)
        sys.exit(99)
                    