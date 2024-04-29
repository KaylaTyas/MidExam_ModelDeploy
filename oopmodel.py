# Kayla Masayuningtyas - 2602141871 - LB09

import pandas as pd
import numpy as np
import seaborn as sn
import warnings
import pickle
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# data hendler Class
class datahandler:
    def __init__(self, filepath):
        self.file_path = filepath
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def remove(self, targetkolom=['id','CustomerId','Surname']):
        self.data = self.data.drop(columns=targetkolom)

    def create_input_output(self, targetcolumn):
        self.output_df = self.data[targetcolumn]
        self.input_df = self.data.drop(targetcolumn, axis=1)

# model handler class
class modelhandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createmodel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def splitdata(self, testsize=0.2, random=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=testsize, random_state=random)
        
    def meancolumn(self, kolom):
        return np.mean(self.x_train[kolom])
    
    def missingvalueimpute(self, kolom, value):
        self.x_train[kolom].fillna(value, inplace = True)
        self.x_test[kolom].fillna(value, inplace = True)

    def binaryencode(self, columns):
        self.train_encode={'Gender': {"Male":0, 'Female':1}, 'Geography': {'France':0, 'Spain':1, 'Germany':2}}
        self.test_encode = {'Gender': {"Male":0, 'Female':1}, 'Geography': {'France':0, 'Spain':1, 'Germany':2}}
        self.x_train=self.x_train.replace(self.train_encode)
        self.x_test=self.x_test.replace(self.test_encode)

    def createmodel(self, criterion='gini', maxdepth=10):
        self.model = RandomForestClassifier(criterion= criterion, max_depth= maxdepth)

    def trainmodel(self):
        self.model.fit(self.x_train, self.y_train)

    def dataprediction(self):
        self.y_predict = self.model.predict(self.x_test)

    def evaluatemodel(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['0', '1']))

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

# load data
file_path = "dataset.csv"
data_handler = datahandler(file_path)
data_handler.load_data()
data_handler.remove()

data_handler.create_input_output('churn')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = modelhandler(input_df, output_df)
model_handler.splitdata()

credit_fillna = model_handler.meancolumn('CreditScore')
model_handler.missingvalueimpute('CreditScore', credit_fillna)
model_handler.binaryencode(['Gender', 'Geography'])

model_handler.trainmodel()
print('Model Accuracy:', model_handler.evaluatemodel())
model_handler.dataprediction()
model_handler.createReport()
model_handler.save_model_to_file('TrainedModel.pkl')