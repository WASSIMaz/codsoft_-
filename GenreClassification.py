import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm


def main():
    #check command-line arguments
    if len(sys.argv) !=4:
        sys.exit("Usage: python GenreClassification.py data1 data2 data3")
    #load train and test data    
    train_data=pd.read_csv(sys.argv[1], delimiter = ' ::: ')
    test_data=pd.read_csv(sys.argv[2], delimiter = ' ::: ')
    test_data_solutions=pd.read_csv(sys.argv[3], delimiter = ' ::: ')
    
    # adding column headings 
    train_data.columns=['Num', 'Name', 'genre','description']
    test_data.columns=['Num', 'Name','description']
    test_data_solutions.columns=['Num', 'Name','genre','description']
    # just to visualize the data
    #sns.countplot(train_data['genre'])
    #plt.show()
    
    # Instantiate the TfidfVectorizer object
    text_transformer=TfidfVectorizer(stop_words='english', ngram_range=(1,2), lowercase=True,max_features=80000)
    
    # Transform the training data into term vectors
    x_train_text=text_transformer.fit_transform(train_data['description'])
    
    # Transform the test data into term vectors
    y_test_text=text_transformer.transform(test_data['description'])
    
    #instantiating the model
    
    #model = GaussianNB()
    #model = svm.SVC()
    model=LogisticRegression(C=5e1, solver='lbfgs', multi_class='multinomial',random_state=17)
    
    
    #evaluation of the model using cross validation function and Stratified K-Fold Cross-Validation strategy 
    skf=StratifiedKFold(n_splits=15, shuffle=True,random_state=17)
    result=cross_val_score(model,x_train_text ,train_data['genre'],cv=skf,scoring='f1_micro')
    print(result.mean())


    #fitting the model on the data
    model.fit(x_train_text,train_data['genre'])
    
    #here i implemented the confusion matrix optionally to get more detail
    predictions=model.predict(y_test_text)
    cm= confusion_matrix(test_data_solutions['genre'],predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = model.classes_).plot()
    plt.show()
if __name__ == "__main__":
    main()
