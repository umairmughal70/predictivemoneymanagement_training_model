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
import numpy as np
import pandas as pd
from datetime import date, datetime
import calendar
from utils.configloader import ConfigUtil
from utils.logging_init              import logging
from utils.mssqlutil import MSSQLUtil
import db.constants as pr_constant

class DataLoaderUtil:
    """
    Utility class for loading and processing data from database
    """
    __instance  = None

    @staticmethod
    def getInstance():
        if DataLoaderUtil.__instance == None:
            DataLoaderUtil()
        return DataLoaderUtil.__instance
    
    def __init__(self):
        """
        Constructor for initializing the file loader isntance
        """
        if DataLoaderUtil.__instance == None:
            DataLoaderUtil.__instance = self
            self.__run()

    def __run(self):
        """
        Load configurations
        """
        instance = ConfigUtil.getInstance()
        self.__envConfiguration = instance.configJSON
        logging.info("loading data from DB...")
        
    def initGenerateData(self, day):
        """ Checks if the day provided is the intended day of initiation of the Weekly Data generantion function
        and then invokes Weekly Data Generation Module."""
        try:
            logging.info('Triggering data generation')
            if date.today().day == day:
                logging.info('Initiatting data generation on {0}'.format(date.today()))
                
                response = self.retrieveDataFromDb()
                return response
            else:
                logging.info('Data generation not initiated on {0}'.format(date.today()))
                return
        
        except Exception as e:
            logging.error(e)
            
    def retrieveDataFromDb(self):
        """This method retrieves the ING_Budget and ING_BudgetTransactions datasets from INGAGAE Databases to generate weekly Budget Dataset."""
        try:
            logging.info("Retrieved Data from Database")
            sqlInstance = MSSQLUtil.getInstance()
            trans_list    = sqlInstance.executeQuery(self.__envConfiguration['db.queryBudgetTrans'])
            budget_list   = sqlInstance.executeQuery(self.__envConfiguration['db.queryBudget'])
            np_trans_list = np.array(trans_list)
            trans_df = pd.DataFrame (np_trans_list, columns = ['BudgetID', 'TransAmount', 'TransDate', 'CalendarYearMonth', 'AllocatedAmount'])
            trans_df['BudgetID']            = trans_df['BudgetID'].astype(int)
            trans_df['TransAmount']         = trans_df['TransAmount'].astype(float)
            trans_df['AllocatedAmount']     = trans_df['AllocatedAmount'].astype(float)
            np_budget_list = np.array(budget_list)
            budget_df = pd.DataFrame (np_budget_list, columns = ['BudgetID', 'CustomerID', 'CategoryTypeCode', 'AllocatedAmount'])
            budget_df['BudgetID']           = budget_df['BudgetID'].astype(int)
            budget_df['CustomerID']         = budget_df['CustomerID'].astype(int)
            budget_df['CategoryTypeCode']   = budget_df['CategoryTypeCode'].astype(str)
            budget_df['AllocatedAmount']    = budget_df['AllocatedAmount'].astype(float)
            response = self.generate_weekly_dataset(trans_df,budget_df,int(self.__envConfiguration['dir.monthsBack']) ,int(self.__envConfiguration['dir.leastMonths']),int(self.__envConfiguration['dir.totalMonths']) )

            return response

        except Exception as e:
            logging.error(e)

    def generate_weekly_dataset(self,trans_df, budget_df, no_of_months_back, least_no_of_months, total_no_of_months):
        """ This method generates weekly budget data of selected months from monthly budget data and Budget Transactions."""
        try:
            #----------------------------------
            sqlInstance = MSSQLUtil.getInstance()
            cat_list    = sqlInstance.executeQuery(self.__envConfiguration['db.queryCatTypes'])
            np_cat_list = np.array(cat_list)
            cat_df = pd.DataFrame (np_cat_list, columns = ['CategoryTypeCode', 'CategoryTypeName'])
            cat_df['Code'] = list(range(1, len(cat_df)+1 ,1))
            cat_df['Code'] = cat_df['Code'].map("{:02}".format)
            cat_df = cat_df.astype(str)
            #----------------------------------
            
            trans_df['TransDate'] = pd.to_datetime(trans_df['TransDate'])
            trans_df['CalendarYearMonth'] = pd.to_datetime(trans_df['CalendarYearMonth'])

            # Selecting Data of particular months
            startDate = trans_df['TransDate'].max() - pd.DateOffset(months = no_of_months_back)
            
            limitDate = startDate + pd.DateOffset(months = least_no_of_months)
            endDate = startDate + pd.DateOffset(months = total_no_of_months)
            # budget_df = budget_df[(budget_df['CalendarYearMonth'] >= startDate) & (budget_df['CalendarYearMonth'] <= endDate)]
            # trans_df = trans_df[(trans_df['TransDate'] >= startDate) & (trans_df['TransDate'] <= endDate)]
            
            budget_df.drop(['AllocatedAmount'], axis = 1, inplace = True)

            #Merging ING_Budget and ING_BudgetTransaction Table as to get CustomerID and Budget Category
            trans_df = pd.merge(trans_df,budget_df, on = 'BudgetID', how = 'left')
            
            #Unique 2 digit Mapping of CategoryTypeCode
            cat_map = dict(zip(cat_df.CategoryTypeCode, cat_df.Code))

            trans_df['CategoryTypeCode'] = trans_df['CategoryTypeCode'].map(cat_map)
            trans_df['TransDate'] = pd.to_datetime(trans_df['TransDate'])
            
            # #Generating All missing days (The days where No Transaction Took Place)
            
            dateRange_df = trans_df.groupby(['CustomerID', 'CategoryTypeCode']).agg(Entry_Date = ('TransDate', 'min')).reset_index()
            
            max_data_date = trans_df['TransDate'].max()
            last_day = max_data_date.replace(day = calendar.monthrange(max_data_date.year, max_data_date.month)[1])
            dateRange_df['Exit_Date'] = last_day
            #dateRange_df = dateRange_df[dateRange_df['Entry_Date'] < limitDate]
            dateRange_df = dateRange_df.loc[dateRange_df.index.repeat((dateRange_df['Exit_Date'] - dateRange_df['Entry_Date']).dt.days + 1)]
            #add counter duplicated rows to day timedeltas to new column
            dateRange_df['TransDate'] = dateRange_df['Entry_Date'] + pd.to_timedelta(dateRange_df.groupby(level=0).cumcount(), unit='d')
            #default RangeIndex
            dateRange_df = dateRange_df.reset_index(drop=True)
            dateRange_df = dateRange_df.drop(['Entry_Date','Exit_Date'], axis=1)
            trans_df = pd.merge(dateRange_df, trans_df, on = ['CustomerID','TransDate', 'CategoryTypeCode'], how = 'left')
            trans_df['TransAmount'] = trans_df['TransAmount'].fillna(0)


            # #Generating Week Column as data will be now weekly aggregated.
            trans_df['Week'] = trans_df['TransDate'].dt.day
            trans_df['Week'] = np.where(trans_df['Week']<=7,1,trans_df['Week'])
            trans_df['Week'] = np.where(trans_df['Week'].between(8,14) ,8 ,trans_df['Week'])
            trans_df['Week'] = np.where(trans_df['Week'].between(15,21) ,15 ,trans_df['Week'])
            trans_df['Week'] = np.where(trans_df['Week']>=22,22,trans_df['Week'])
            
            trans_df['Weektemp'] = trans_df['TransDate'].dt.strftime('%Y-%m').astype(str) + '-' + trans_df['Week'].astype(str)
            trans_df['Week'] = trans_df['Weektemp']
            trans_df.drop('Weektemp', axis=1, inplace=True)

            trans_df['CustomerID'] = trans_df['CustomerID'].astype(int)
            trans_df['Week'] = pd.to_datetime(trans_df['Week'])
            trans_df['Week'] = trans_df['Week'].dt.date
            trans_df.sort_values(['CustomerID', 'Week','CategoryTypeCode'], inplace = True)
            trans_df['AllocatedAmount'] = trans_df.groupby(['CustomerID', pd.to_datetime(trans_df['Week']).dt.month,'CategoryTypeCode'])['AllocatedAmount'].transform(lambda x: x.ffill())

            trans_df = trans_df.fillna(0)
            
            # # #Aggregating Weekly Transactions as per their Budget Categories
            trans_df_group = trans_df.groupby(['CustomerID', 'Week','AllocatedAmount' ,'CategoryTypeCode'])['TransAmount'].sum().reset_index()
            trans_df_group = trans_df_group.rename(columns={'TransAmount': 'Spending'})
            trans_df_group.sort_values(['CustomerID','Week','Spending'], inplace = True)
            trans_df_group.drop_duplicates(subset = ['CustomerID','Week', 'CategoryTypeCode'], keep = 'last', inplace = True)

            trans_df_group['arrival'] = pd.to_datetime(trans_df_group.groupby('CustomerID')['Week'].transform(min))
            trans_df_group = trans_df_group[trans_df_group['arrival'] <= limitDate]
            
            trans_df_group.drop(['arrival'], axis = 1, inplace = True)
            trans_df_group['Spending'] = trans_df_group['Spending'].astype(float).round(3) 
            trans_df_group['Avg_Spending'] = trans_df_group.groupby(['CustomerID','CategoryTypeCode', pd.to_datetime(trans_df_group['Week']).dt.day ])['Spending'].transform('mean')
            trans_df_group['Weekly_Sum'] = trans_df_group.groupby(['CustomerID','CategoryTypeCode', pd.to_datetime(trans_df_group['Week']).dt.day ])['Spending'].transform('sum')
            trans_df_group['All_Sum'] = trans_df_group.groupby(['CustomerID','CategoryTypeCode'])['Spending'].transform('sum')
            
            trans_df_group['percent_spending'] = (trans_df_group['Weekly_Sum'] / trans_df_group['All_Sum']) * 100
            trans_df_group.replace([np.inf, -np.inf], np.nan, inplace=True)
            trans_df_group = trans_df_group.fillna(0)
            trans_df_group['AllocatedAmount'] = trans_df_group['AllocatedAmount'] * trans_df_group['percent_spending'] / 100
            trans_df_group['AllocatedAmount'] = trans_df_group['AllocatedAmount'].astype(int)
            
            #load from weekly data
            trans_df_group_copy = trans_df_group[['CustomerID','Week','AllocatedAmount','CategoryTypeCode','Spending']]

            logging.info("Generated Weekly Data")            
            sqlInstance = MSSQLUtil.getInstance()
            sqlInstance.transactionQuery(self.__envConfiguration['db.tunrcateBudgtTableQry'])
            logging.info("Table truncated")
            for index, row in trans_df_group_copy.iterrows():
                sqlParams = [ row['CustomerID'], row['Week'], row['AllocatedAmount'], row['CategoryTypeCode'],  row['Spending']]
                sqlInstance.transactionQuery(pr_constant.INSERT_AI_BUDGET_SQL, sqlParams)
                
            logging.info("Stored Weekly Data into Database")
            return True

        except Exception as e:
            logging.error(e)

