import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import numpy as np




def main():
    #check command-line arguments
    if len(sys.argv) !=3:
        sys.exit("Usage: python FraudDetection.py data1 data2")
        
    #training data preprocessing 
    x_train,y_train=load_and_preprocess(sys.argv[1]) 
    
    #data fitting
    model,seleced_features=train_model(x_train,y_train)  
    
    #testing data loading
    x_test,y_test=load_and_preprocess(sys.argv[2]) 
       
    #subset the training data to include only selected features   
    selected_x_test=np.array(x_test)[:,seleced_features]
    
    #get the model's scores
    correct,incorrect,total=evaluate(model,selected_x_test,y_test)
    
    #print result
    print("correct: ",correct)
    print("incorrect: ",incorrect)
    print("model accuracy: ",(correct/total)*100)
    
    
def load_and_preprocess(filename):
    #load train and test data    
    train_data=pd.read_csv(filename)
    categories={train_data['category'].unique()[i]:i for i in range(len(train_data['category'].unique()))}
    #train_data.drop(['trans_date_trans_time','cc_num','merchant','first','last','gender','street','city','state','zip','job','dob','trans_num','unix_time'])
    #sns.displot(train_data, x="category")
    #plt.show()   
    #sns.countplot(train_data['is_fraud'])
    #sns.histplot(train_data['is_fraud'], kde=True, color="m")
    with open(filename) as f:
     reader=csv.DictReader(f)
     legitimate=0
     fraudulent_evidnce=[]
     legitimate_evidence=[]
     for row in reader:
        f=[]   
        f.append(categories[row["category"]])
        f.append(float(row["amt"]))
        f.append(float(row["lat"]))
        f.append(float(row["long"]))
        f.append(int(row["city_pop"]))
        f.append(float(row["merch_lat"]))
        f.append(float(row["merch_long"]))
        if row['is_fraud']=='0':
            fraudulent_evidnce.append((f,0))
        else:
            legitimate+=1
            legitimate_evidence.append((f,1))
    random.shuffle(fraudulent_evidnce)        
    evi_lab=legitimate_evidence+fraudulent_evidnce[:legitimate]
    random.shuffle(evi_lab)  
    
    evidence=[]
    label=[]
    for i in evi_lab:
        evidence.append(i[0])
        label.append(i[1])
    return (evidence,label)


def train_model(x,y):
    
  #intantiate RandomForestClassifier model
  model=LogisticRegression(random_state=42)
  #model=RandomForestClassifier(n_estimators=100, random_state=42) 
  
  #intantiate RFE with 6 features to be selected
  rfe_model=RFE(model, n_features_to_select=6)
  
  x_train=np.array(x)
  #fit and train the RFE
  rfe_model.fit(x_train,y)
  
  #get the selected features
  selected_fetures=rfe_model.support_
  
  #subset the training data to include only selected features
  selected_training_data=x_train[:,selected_fetures]
  
  #fit and train the model on the given data
  model.fit(selected_training_data,y)  
  
  
  return model ,selected_fetures


def evaluate (mo,data,data2):
    #make prediction 
    predictions=mo.predict(data)
    
    #compute how well the model performed
    correct=0
    incorrect=0
    total=0
    for prediction,actual in zip(predictions,data2):
        total+=1
        if prediction==actual:
            correct+=1
        else:
            incorrect+=1
    
    return correct,incorrect,total
  
  
if __name__ == "__main__":
    main()
