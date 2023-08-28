''''
module testing churn prediction
author: mulhermaonocodigo
date: August, 27, 2023
'''
import os
import logging
import pandas as pd
import joblib
import churn_library as cls



logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(pth):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		dataframe = cls.import_data(pth)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err
	try:
		assert dataframe.shape[0] > 0
		assert dataframe.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err

def test_eda(pth):
	'''
	test perform eda function
	'''
	#read  dataframe
	dataframe = cls.import_data(pth)
	#plot histogram
	typeplot  =  'histogram'
	columns_name = ['Churn', 'Customer_Age']
	try:
		cls.perform_eda(dataframe, typeplot, columns_name)
		assert os.path.isfile("./images/eda/"+columns_name[0]+"_hist.png")
		assert os.path.isfile("./images/eda/"+columns_name[1]+"_hist.png")
		logging.info("Testing plot histogram: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing saved figures: The file wasn't found")
		raise err
	typeplot  =  'bar'
	columns_name = 'Marital_Status'
	try:
		cls.perform_eda(dataframe, typeplot, columns_name)
		assert os.path.isfile("./images/eda/"+columns_name+"_bar.png")
		logging.info("Testing plot bar: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing saved figures: The file wasn't found")
		raise err
	typeplot  =  'density'
	columns_name = 'Total_Trans_Ct'
	try:
		cls.perform_eda(dataframe, typeplot, columns_name)
		assert os.path.isfile("./images/eda/"+columns_name+"_density.png")
		logging.info("Testing plot density: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing saved figures: The file wasn't found")
		raise err
	typeplot  =  'heatmap'
	columns_name = ''
	try:
		cls.perform_eda(dataframe, typeplot, columns_name)
		assert os.path.isfile("./images/eda/heatmap.png")
		logging.info("Testing plot heatmap: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing saved figures: The file wasn't found")
		raise err

def test_encoder_helper(pth,category_lst):
	'''
	test encoder helper
	'''
	try:
		dataframe = cls.import_data(pth)
		dataframe = cls.encoder_helper(dataframe, category_lst)
		created_columns = set(dataframe.columns).intersection(set(category_lst))
		assert len(created_columns)==len(category_lst)
		logging.info("Testing create new columns: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing  create new columns: The columns wasn't found")
		raise err

def test_perform_feature_engineering(response,category_lst,pth):
	'''
	test perform_feature_engineering
	'''
	try:
		dataframe = cls.import_data(pth)
		dataframe = cls.encoder_helper(dataframe, category_lst)	
		x_train, x_test, y_train, y_test  = cls.perform_feature_engineering(dataframe, response)
	except Exception as err:
		logging.error("Testing  feature_engineering performance: the split data for training failed")
		raise err
	try:
		assert x_train.shape[0] > 0
		assert x_train.shape[1] == len(response)
		assert x_test.shape[0] > 0
		assert x_test.shape[1] == len(response)
		assert y_train.shape[0] > 0
		assert y_test.shape[0] > 0
		logging.info("Testing feature_engineering data")
	except AssertionError:
		logging.error("The returned train - test datasets didnt loaded samples")
		raise err

def test_train_models(pth,category_lst,response):
	'''
	test train_models
	'''
	try:
		dataframe = cls.import_data(pth)
		dataframe = cls.encoder_helper(dataframe, category_lst)
		x_train, x_test, y_train, y_test  = cls.perform_feature_engineering(dataframe, response)
		logging.info("Testing loading data")
	except Exception as err:
		logging.error("Testing  loading data performance: failed")
		raise err
	try:	
		cls.train_models(x_train, x_test, y_train, y_test)
		logging.info("Testing train model")
	except Exception as err:
		logging.error("Testing train model performance: failed")
		raise err
	try:
		assert os.path.isfile("./images/results/random_forest.png")
		assert os.path.isfile("./images/results/logistic_regression.png")
		logging.info("Testing model results: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing saved figures: The file wasn't found")
		raise err
	try:
		assert os.path.isfile("./models/logistic_model.pkl")
		assert os.path.isfile("./models/rfc_model.pkl")
		logging.info("Testing saving model: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing saved models: The file wasn't found")
		raise err

def test_report_results(inputpth,inputmodel):
	'''
	test train_models
	'''
	try:
		rfc_model = joblib.load(inputmodel+ 'rfc_model.pkl')
		lr_model  = joblib.load(inputmodel+ 'logistic_model.pkl')
		x_train = pd.read_csv(inputpth+'X_train.csv')
		x_test	= pd.read_csv(inputpth+'X_test.csv')
		y_test	= pd.read_csv(inputpth+'y_test.csv')
		cls.results_report(rfc_model,lr_model,x_train,x_test,y_test)
		logging.info("Testing report")
	except Exception as err:
		logging.error("Testing  report performance: failed")
		raise err
	try:
		assert os.path.isfile("./images/results/Random Forest_explain.png")
		assert os.path.isfile("./images/results/Random Forest_importance.png")
		assert os.path.isfile("./images/results/plot_roc_curve.png")
		logging.info("Testing plot results: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing saved figures: The file wasn't found")
		raise err

if __name__ == "__main__":
	#variables to use at test
	PTH= "./data/bank_data.csv"
	INPUT_PTH= "./data/"
	INPUT_MODEL = "./models/"
	cat_columns = ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
	keep_col = ['Customer_Age',
    'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn']
	#tests
	test_import(PTH)
	test_eda(PTH)
	test_encoder_helper(PTH,cat_columns)
	test_perform_feature_engineering(keep_col, cat_columns,PTH)
	test_train_models(PTH,cat_columns,keep_col)
	test_report_results(INPUT_PTH,INPUT_MODEL)
