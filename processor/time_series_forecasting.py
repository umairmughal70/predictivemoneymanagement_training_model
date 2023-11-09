import os
from pathlib import Path
import numpy as np
import pandas as pd
import keras_tuner as kt
from keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.layers import Dense,  LSTM
from collections import defaultdict
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from math import sqrt
from datetime import timedelta
from utils.logging_init     import logging
from utils.mssqlutil import MSSQLUtil
from utils.file_loader import FileLoaderUtil
import warnings
warnings.filterwarnings('ignore')
from os import path


class Timeseries_forecasting_using_lstm():

    def __init__(self, base_directory,features, dtypes,env_dict, dimension_name='', dimension_cols=[], cat_map=[]):
        """ The default constructor of the class where all class attributes are properly initiallized """

        try:
            self.__envConfiguration = env_dict
            self.save_directory = os.path.join(base_directory, self.__envConfiguration['dir.modelName'])

            Path(self.save_directory).mkdir(parents=True, exist_ok=True)

            self.model_name     = self.__envConfiguration['dir.modelName']
            self.date_column    = self.__envConfiguration['dir.dateCol']
            self.frequency      = self.__envConfiguration['dir.frequency']
            self.target_column  = self.__envConfiguration['dir.targetCol']
            self.look_back          = int(self.__envConfiguration['dir.lookBack'])
            self.forecast_horizon = int(self.__envConfiguration['dir.forecastHorizon'])
            self.dtypes = dtypes
            self.dimension_name = dimension_name
            self.dimension_cols = dimension_cols
            self.cat_map = cat_map
            self.use_validation = self.__envConfiguration['dir.useValidation']
            self.split_type = self.__envConfiguration['dir.splitType']
            self.model_type = self.__envConfiguration['dir.modelType']
            self.all_groups = []
            self.meta_data = {}
            self.meta_data['group_specific_numeric_col_means'] = defaultdict(dict)

            self.best_models_path = os.path.join(f'{self.save_directory}', 'best_models')
            Path(self.best_models_path).mkdir(parents=True, exist_ok=True)

            self.df_test = self.retrieve_data_from_db()

            target = self.df_test[self.__envConfiguration['dir.targetCol']]
            self.df_test = self.df_test[[self.__envConfiguration['dir.dateCol']] + dimension_cols + features + [self.__envConfiguration['dir.targetCol']]]
            self.df_test.drop(columns=[self.target_column], axis=1, inplace=True)
            self.df_test[self.__envConfiguration['dir.targetCol']] = target

            self.df_test[self.__envConfiguration['dir.dateCol']] = pd.to_datetime(self.df_test[self.__envConfiguration['dir.dateCol']])

            self.df_test.reset_index(drop=True, inplace=True)

            logging.info('Length of original data: {} '.format(len(self.df_test)))

            self.numeric_cols = [col for col in self.df_test.columns if
                                self.dtypes[col] == 'numeric' and col != self.target_column]
            self.object_cols = [col for col in self.df_test.columns if
                                self.dtypes[col] == 'categorical' and col != self.target_column]
            self.timestamp_cols = [col for col in self.df_test.columns if
                                self.dtypes[col] == 'timestamp' and col != self.target_column]

            logging.info('Numeric cols: {} '.format(self.numeric_cols))
            logging.info('Object cols: {}'.format(self.object_cols))
            logging.info('Timestamp cols: {}'.format(self.timestamp_cols))

            numeric_col_means = {}
            for col in self.numeric_cols:
                logging.info('Filling missing values in numeric cols.')
                self.df_test[col] = self.df_test[col].ffill().add(self.df_test[col].bfill()).div(2)
                self.df_test[col] = self.df_test[col].ffill()
                self.df_test[col] = self.df_test[col].bfill()
                numeric_col_means[col] = self.df_test[col].mean()

            if self.object_cols:
                logging.info('Encoding categorical features.')
                oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
                for col in self.object_cols:
                    self.df_test[col] = self.df_test[col].astype(str)

                oe.fit(self.df_test[self.object_cols])
                self.df_test[self.object_cols] = oe.transform(self.df_test[self.object_cols])
                self.oe = oe
                self.meta_data['categorical_encoder'] = self.oe

            self.all_columns = [self.__envConfiguration['dir.dateCol']] + dimension_cols + features + [self.__envConfiguration['dir.targetCol']]
            self.features = [self.__envConfiguration['dir.dateCol']] + dimension_cols + features
            self.df_test = self.df_test[self.all_columns]

            self.meta_data['features'] = self.features
            self.meta_data['all_columns'] = self.all_columns
            self.meta_data['object_cols'] = self.object_cols
            self.meta_data['numeric_cols'] = self.numeric_cols
            self.meta_data['timestamp_cols'] = self.timestamp_cols
            self.meta_data['timesteps'] = self.look_back
            self.meta_data['forecast_horizon'] = int(self.__envConfiguration['dir.forecastHorizon'])
            self.meta_data['n_features'] = len(self.all_columns) - 1
            self.meta_data['dimension_cols'] = self.dimension_cols
            self.meta_data['target_column'] = self.target_column
            self.meta_data['date_column'] = self.date_column
            self.meta_data['frequency'] = self.frequency
            self.meta_data['cat_df'] = self.cat_map

            self.test_predictions = []
            self.score = []
            self.y_test = []

            self.n_timesteps = self.look_back
            self.n_features = len(self.all_columns) - 1
            self.n_outputs = self.forecast_horizon
            logging.info(f'disc n_outputs : {self.n_outputs}')


            self.df_test.reset_index(drop=True, inplace=True)

            logging.info("Training Constructor Called")
        
        except Exception as e:
            logging.error(e)
    
    def fill_missing_values(self):

        """ This method fills the missing values in Target Column in our Training Dataset"""
        try:
            self.df_test[self.target_column] = self.df_test[self.target_column].ffill().add(
                self.df_test[self.target_column].bfill()).div(2)
            logging.info("Missing Values Filled in Traget Column")
        
        except Exception as e:
            logging.info(e)
    
    def fill_missing_values_in_group(self, group_name, df_group):

        """ This method fills missing values in Train-Test Split Groups"""
        try:
            for col in self.numeric_cols:
                df_group[col] = df_group[col].ffill().add(df_group[col].bfill()).div(2)
                df_group[col] = df_group[col].ffill()
                df_group[col] = df_group[col].bfill()
                self.meta_data['group_specific_numeric_col_means'][group_name][col] = df_group[col].mean()

            logging.info("Filled Missing Values")
            return df_group
            

        except Exception as e:
            logging.error(e)
            
    def get_train_data(self, x_train):
        """ This method generates training data which will be further processed to feed to the model"""
        try:
            logging.info('Train Data shape initially: {}'.format(x_train.shape))
            features, target = list(), list()
            in_start = 0
            x_train = x_train.values

            # step over the entire history one time step at a time
            for _ in range(len(x_train)):
                # define the end of the input sequence
                in_end = in_start + self.look_back
                out_end = in_end + self.forecast_horizon
                # ensure we have enough data for this instance
                if out_end <= len(x_train):
                    x_input = x_train[in_start:in_end, 0:len(self.all_columns)]  # Leaving date column
                    features.append(x_input)
                    target.append(
                        x_train[in_end:out_end, len(self.all_columns) - 1])  # Just target column which is at the last

                # move along one time step
                in_start += 1

            x_train = np.asarray(features)
            y_train = np.asarray(target)

            logging.info("Generated Training Data")
            return x_train, y_train

        except Exception as e:
            logging.error(e)

    def get_test_data(self, x_test):
        """ This method generates test data which will be used to evaluate the performance of model after training."""
        try:
            in_start = 0
            features, target = list(), list()
            x_test = x_test.values
            for _ in range(len(x_test)):
                # define the end of the input sequence
                in_end = in_start + self.look_back
                out_end = in_end + self.forecast_horizon
                # ensure we have enough data for this instance
                if out_end <= len(x_test):
                    x_input = x_test[in_start:in_end, :]  # Leaving date column
                    features.append(x_input)
                    target.append(
                        x_test[in_end:out_end, len(self.all_columns) - 1])  # Just target column which is at the last
                # move along one time step
                in_start += self.forecast_horizon

            x_test = np.array(features)
            y_test = np.array(target)
            logging.info("Generated Test Data")
            return x_test, y_test

        except Exception as e:
            logging.error(e)

    def get_valid_data(self, x_val):
        """This method generates validation data which will be used to evaluate/validate the model performance during the training"""
        try:
        
            in_start = 0
            features, target = list(), list()
            x_val = x_val.values
            for _ in range(len(x_val)):
                # define the end of the input sequence
                in_end = in_start + self.look_back
                out_end = in_end + self.forecast_horizon
                # ensure we have enough data for this instance
                if out_end <= len(x_val):
                    x_input = x_val[in_start:in_end, :]  # Leaving date column
                    features.append(x_input)
                    target.append(
                        x_val[in_end:out_end, len(self.all_columns) - 1])  # Just target column which is at the last
                # move along one time step
                in_start += self.forecast_horizon

            x_val = np.array(features)
            y_val = np.array(target)
            logging.info("Generated Validation Data")
            return x_val, y_val

        except Exception as e:
            logging.error(e)

    def split_with_validation(self, df_temp):
        """This method splits the dataset into train, test and validation sets."""
        try:
            
            # --------------------------------------- WITH OVERLAPPING ---------------------------------------------------
            train_upto = self.look_back + self.forecast_horizon + self.forecast_horizon 
            valid_from = self.look_back + self.forecast_horizon + self.forecast_horizon + self.forecast_horizon 
            valid_upto = self.forecast_horizon + self.forecast_horizon 

            test_from = self.look_back + self.forecast_horizon 

            x_train = df_temp[self.all_columns].iloc[0:-train_upto, :] 
            x_val = df_temp[self.all_columns].iloc[-valid_from: -valid_upto, :] 
            x_test = df_temp[self.all_columns].iloc[-test_from:, :] 

            x_train.reset_index(drop=True, inplace=True)
            x_val.reset_index(drop=True, inplace=True)
            x_test.reset_index(drop=True, inplace=True)

            x_train, y_train = self.get_train_data(x_train)
            x_test, y_test = self.get_test_data(x_test)
            x_val, y_val = self.get_valid_data(x_val)


            logging.debug('Final Train Data shape: {}'.format(x_train.shape))
            logging.debug('Final Train Target shape: {}'.format(y_train.shape))

            logging.debug('Final Valid Data shape: {}'.format(x_val.shape))
            logging.debug('Final Valid Target shape: {}'.format(y_val.shape))

            logging.debug('Final Test Data shape: {}'.format(x_test.shape))
            logging.debug('Final Test Target shape: {}'.format(y_test.shape))

            logging.debug("Generated Test Train Splits with Validation Data")

            return x_train, y_train, x_val, y_val, x_test, y_test

        except Exception as e:
            logging.error(e)

    def normalize_data(self):
        """This method normalizes the train, test and validation dataset before feeding to the model as different features might have different scales and it may overturn the model performance"""
        try:
            if len(self.features) > 1:  # Because date will always be in self.features and we dont standardize it.

                train_data_shape = self.x_train.shape
                val_data_shape = self.x_val.shape
                test_data_shape = self.x_test.shape

                self.x_train = self.x_train.reshape((train_data_shape[0] * train_data_shape[1], train_data_shape[2]))
                self.x_val = self.x_val.reshape((val_data_shape[0] * val_data_shape[1], val_data_shape[2]))
                self.x_test = self.x_test.reshape((test_data_shape[0] * test_data_shape[1], test_data_shape[2]))

                feature_scaler = StandardScaler()
                feature_scaler.fit(self.x_train[:, 1:len(self.all_columns) - 1])

                #Date column will always be on start. Target column will always be on last.
                self.x_train[:, 1:len(self.all_columns) - 1] = feature_scaler.transform(
                    self.x_train[:, 1:len(self.all_columns) - 1])
                self.x_val[:, 1:len(self.all_columns) - 1] = feature_scaler.transform(
                    self.x_val[:, 1:len(self.all_columns) - 1])
                self.x_test[:, 1:len(self.all_columns) - 1] = feature_scaler.transform(
                    self.x_test[:, 1:len(self.all_columns) - 1])

                self.feature_scaler = feature_scaler

                self.x_train = self.x_train.reshape(train_data_shape)
                self.x_val = self.x_val.reshape(val_data_shape)
                self.x_test = self.x_test.reshape(test_data_shape)

                self.meta_data['feature_scaler'] = feature_scaler

                logging.info("Normalized the Data")

        except Exception as e:
            logging.error(e)
    
    def split_dataset(self):
        """ This method splits the dataset into train test and validation datasets."""
        try:
            self.meta_data['numeric_col_means'] = {}
            for col in self.numeric_cols:
                self.meta_data['numeric_col_means'][col] = self.df_test[col].mean()

            grouped_data = self.df_test.groupby(self.dimension_cols)
            group_no = 1
            for group_name, df_grouped in grouped_data:
                logging.info("Started Splitting Dataset")

                if group_no == 1:
                    df_grouped = self.fill_missing_values_in_group(group_name, df_grouped)
                    self.x_train, self.y_train, self.x_val, \
                    self.y_val, self.x_test, self.y_test = self.split_with_validation(df_grouped)
                    self.all_groups.append(group_name)
                else:
                    df_grouped = self.fill_missing_values_in_group(group_name, df_grouped)
                    x_train, y_train, x_val, y_val, x_test, y_test = self.split_with_validation(df_grouped)
                    self.x_train = np.concatenate((self.x_train, x_train))
                    self.y_train = np.concatenate((self.y_train, y_train))
                    self.x_val = np.concatenate((self.x_val, x_val))
                    self.y_val = np.concatenate((self.y_val, y_val))
                    self.x_test = np.concatenate((self.x_test, x_test))
                    self.y_test = np.concatenate((self.y_test, y_test))
                    self.all_groups.append(group_name)

                group_no += 1

            indexes = np.arange(len(self.x_train))
            np.random.shuffle(indexes)
            self.x_train = self.x_train[indexes]
            self.y_train = self.y_train[indexes]

            logging.debug('Train X Final Length: {}'.format(len(self.x_train)))
            logging.debug('Train Y Final Length: {}'.format(len(self.x_train)))
            logging.debug('Valid X Final Length: {}'.format(len(self.x_train)))
            logging.debug('Valid Y Final Length: {}'.format(len(self.x_train)))
            logging.debug('Test  X Final Length: {}'.format(len(self.x_train)))
            logging.debug('Test  Y Final Length: {}'.format(len(self.x_train)))

            self.normalize_data()
            fileObj = FileLoaderUtil()
            fileObj.writeFile( self.meta_data, self.save_directory, 'meta_data' )
            
            logging.info("Generated Test Train Splits with Validation Data")

        except Exception as e:
            logging.error(e)

    def model_builder_1(self, hp):
        """This is our first model for LSTM as we are trying different LSTM Architectures to get best performing model."""
        try:
            model = keras.Sequential()
            logging.info(f'Look back:  {self.n_timesteps}')
            logging.info(f'features:  {self.n_features}')
            hp_units = hp.Int('ist_units', min_value=16, max_value=64, step=16)
            model.add(LSTM(units=hp_units, input_shape=(self.n_timesteps, self.n_features), activation='relu'))
            model.add(Dense(self.n_outputs))
            hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.002])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mse', metrics=['mse'])

            logging.info("Model Built")
            return model

        except Exception as e:
            logging.error(e)

    def model_builder_2(self, hp):
        """This is our second model for LSTM as we are trying different LSTM Architectures to get best performing model."""
        try:

            model = keras.Sequential()
            hp_units = hp.Int('ist_units', min_value=16, max_value=64, step=16)
            model.add(LSTM(units=hp_units, input_shape=(self.n_timesteps, self.n_features), activation='relu'))
            hp_units = hp.Int('2nd_units', min_value=16, max_value=64, step=16)
            model.add(Dense(units=hp_units, activation='relu'))
            model.add(Dense(self.n_outputs))
            hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.002])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mse', metrics=['mse'])
            logging.info("Model Built")
            return model

        except Exception as e:
            logging.error(e)

    def model_builder_3(self, hp):
        """This is our third model for LSTM as we are trying different LSTM Architectures to get best performing model."""
        try:

            model = keras.Sequential()
            hp_units = hp.Int('ist_units', min_value=16, max_value=64, step=16)
            model.add(LSTM(units=hp_units, input_shape=(self.n_timesteps, self.n_features), activation='relu'))
            hp_units = hp.Int('2nd_units', min_value=16, max_value=64, step=16)
            model.add(Dense(units=hp_units, activation='relu'))
            hp_units = hp.Int('3rd_units', min_value=16, max_value=64, step=16)
            model.add(Dense(units=hp_units, activation='relu'))
            model.add(Dense(self.n_outputs))
            hp_learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.002])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='mse', metrics=['mse'])
            logging.info("Model Built ")
            return model
        except Exception as e:
            logging.error(e)

    def train_model(self, model_type):

        logging.info('Training model')
        model_type, trials = model_type[0], model_type[1]

        callbacks = [EarlyStopping(monitor="val_loss", patience=10, verbose=1)]


        if model_type == 'one':
            # if(path.exists(r'G:\Use cases\Usecase #2 Budget Amount Recommendation\temp_code\data\LSTM\deployed_model.h5')):
            if(path.exists(os.path.join(self.save_directory,'deployed_model.h5'))):
            
                logging.info("Trained model already exists, Retraining model")
                pretrained_model = keras.models.load_model(os.path.join(self.save_directory,'deployed_model.h5'))
                pretrained_model.compile(loss = 'mse')
                # callbacks = [EarlyStopping(monitor="val_loss", patience=10, verbose=1)]

                pretrained_model.fit(self.x_train[:, :, 1:len(self.all_columns)].astype(np.float32),
                         self.y_train.astype(np.float32), batch_size=32, epochs=40,
                         validation_data=(
                         self.x_val[:, :, 1:len(self.all_columns)].astype(np.float32), self.y_val.astype(np.float32)),
                         callbacks=callbacks, verbose=2)

                logging.info("Model Fit Successfully")
                best_model = pretrained_model
                test_result = best_model.evaluate(self.x_test[:, :, 1:len(self.all_columns)].astype(np.float32),
                                                self.y_test.astype(np.float32), verbose=False)

            else:
                logging.info("No trained model exists, training model")

                tuner = kt.BayesianOptimization(self.model_builder_1, max_trials=trials, objective='val_loss',
                                                directory=self.save_directory, project_name='bayseian_opt_proc_1',
                                                overwrite=True)

                
                tuner.search(self.x_train[:, :, 1:len(self.all_columns)].astype(np.float32),
                            self.y_train.astype(np.float32), batch_size=32, epochs=40,
                            validation_data=(
                            self.x_val[:, :, 1:len(self.all_columns)].astype(np.float32), self.y_val.astype(np.float32)),
                            callbacks=callbacks, verbose=2)

                # Get the optimal hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

                best_model = tuner.get_best_models(1)[0]

                test_result = best_model.evaluate(self.x_test[:, :, 1:len(self.all_columns)].astype(np.float32),
                                                self.y_test.astype(np.float32), verbose=False)[0]

            best_model.save(f'{self.best_models_path}//best_model_proc_1.h5', include_optimizer = True)

            dates = self.x_test[:, :, 0]
            predictions = best_model.predict(self.x_test[:, :, 1:len(self.all_columns)].astype(np.float32))

            if self.model_type == 'single':
                # -------------------------------- Convert 3D test data back to dataframe -----------------------------------
                df_predictions = pd.DataFrame([list(l) for l in self.x_test]).stack().apply(pd.Series).reset_index(1,
                                                                                                                   drop=True)
                df_predictions.index.name = 'dummy_index'
                df_predictions.columns = self.all_columns
                df_predictions.reset_index(drop=True, inplace=True)

                # ------------------------------------------------------------------------------------------------------

                # Denormalizing data + inverse transform categorical features.
                df_predictions[self.features[1:]] = self.feature_scaler.inverse_transform(
                    df_predictions[self.features[1:]])

                if self.object_cols:
                    df_predictions[self.object_cols] = self.oe.inverse_transform(df_predictions[self.object_cols])

                # Get last testing date of each dimension group.
                df_predictions = df_predictions.groupby(self.dimension_cols).tail(1)
                df_predictions.reset_index(drop=True, inplace=True)

                # Generate test dates for each dimension group. and attach actual and predicted target variable.
                df_predicted = pd.DataFrame(columns=list(df_predictions) + [f'{self.model_name}_prediction'])

                for index, row in df_predictions.iterrows():
                    df_temp = pd.DataFrame(columns=self.all_columns)

                    if self.frequency == 'D':
                        df_temp[self.date_column] = pd.date_range(row[self.date_column] + timedelta(days=1),
                                                                  periods=self.forecast_horizon,
                                                                  freq=self.frequency)
                    else:
                        df_temp[self.date_column] = pd.date_range(row[self.date_column],
                                                                  periods=self.forecast_horizon,
                                                                  freq=self.frequency)

                    df_temp[self.all_columns[1:]] = list(row[1:])
                    df_temp[f'{self.target_column}'] = self.y_test[index]
                    df_temp[f'{self.model_name}_prediction'] = predictions[index]
                    df_predicted = pd.concat([df_predicted, df_temp], ignore_index=True)

                # df_predicted.to_csv(f'{self.best_models_path}//model_1.csv', index=False)

            return ['model_1', sqrt(test_result)]
        
       
    def generate_training_dataset(self,weekly_budget):
        """This method generates the training dataset from the Entire Weekly Budget Datset."""
        try:
            logging.info('Generating training dataset')
            trans_df_group_copy=weekly_budget.copy()

            trans_df_group_copy['CustomerSeries'] = trans_df_group_copy['CustomerID'].astype(int).astype(str) + trans_df_group_copy['CategoryTypeCode'].astype(str)
            trans_df_group_copy.drop(['CustomerID','CategoryTypeCode'], axis =1, inplace = True)
            trans_df_group_copy = trans_df_group_copy[['CustomerSeries','Week','AllocatedAmount','Spending']]

            #At this stage all weekly data has been generated from start date of each CustomerSeries till last week of Max Date.
            #Now we have to select training data from this weekly data.

            last_day = pd.to_datetime(trans_df_group_copy[trans_df_group_copy['Spending'] != 0]['Week'].max()).day

            if last_day == 1:
                df_train = trans_df_group_copy.sort_values(['Week'],ascending=True).groupby(["CustomerSeries"], as_index=False).apply(lambda x: x.iloc[:-3]).reset_index(level=0, drop=True)
            elif last_day == 8:
                df_train = trans_df_group_copy.sort_values(['Week'],ascending=True).groupby(["CustomerSeries"], as_index=False).apply(lambda x: x.iloc[:-2]).reset_index(level=0, drop=True)
            elif last_day == 15:
                df_train = trans_df_group_copy.sort_values(['Week'],ascending=True).groupby(["CustomerSeries"], as_index=False).apply(lambda x: x.iloc[:-1]).reset_index(level=0, drop=True)
            else:
                df_train = trans_df_group_copy
                
            logging.info("Generated Training Dataset")
            return df_train
        except Exception as e:
            logging.error(e)
  
    def retrieve_data_from_db(self):
        """This method retrieves the weekly budget dataset from INGAGAE Databases."""

        try:
            logging.info('Retrieving data from db')
            sqlInstance = MSSQLUtil.getInstance()
            weekly_data_list    = sqlInstance.executeQuery(self.__envConfiguration['db.budgetWeeklyQuery'])
            np_weekly_list = np.array(weekly_data_list)
            weekly_budget = pd.DataFrame (np_weekly_list, columns = ['CustomerID', 'Week', 'AllocatedAmount', 'CategoryTypeCode', 'Spending'])
            weekly_budget['CustomerID']         = weekly_budget['CustomerID'].astype(int)
            weekly_budget['Week']               = weekly_budget['Week'].astype(str)
            weekly_budget['AllocatedAmount']    = weekly_budget['AllocatedAmount'].astype(float)
            weekly_budget['CategoryTypeCode']   = weekly_budget['CategoryTypeCode'].astype(int)
            weekly_budget['Spending']           = weekly_budget['Spending'].astype(float)
            
            weekly_budget=weekly_budget.groupby(['CustomerID','CategoryTypeCode']).filter(lambda x: len(x) >=(int(self.__envConfiguration['dir.lookBack'])+int(self.__envConfiguration['dir.forecastHorizon']))*3)

            if len(weekly_budget)>0:
                df=self.generate_training_dataset(weekly_budget)
                # df.to_csv('data/training.csv', index=False)
                df=df.astype({'CustomerSeries': 'int64'})
                logging.info("Fetched Data from Database")
                return df

            else:
                logging.info("Required data not available")

        except Exception as e:
            logging.error(e)

    