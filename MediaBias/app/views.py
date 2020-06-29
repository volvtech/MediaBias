from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

# Create your views here.
def index(request):
    return render(request, 'index.html')

def ensemble_func(X,y):
    smote = SMOTE('all')
    X_sm, y_sm =  smote.fit_sample(X, y.ravel())
    
    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=42)

    model1 = MultinomialNB()
    model2 = LogisticRegression(max_iter=500)
    model3 = SVC(probability=True)
    
    model = VotingClassifier(estimators=[('mnb', model1), ('lr', model2),('svc',model3)], voting='soft')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    m_score = model.score(X_test,y_test)
    # print('m_score ',m_score)
    acc = {'accuracy': m_score}
    return acc
    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))

def mediaBiasTrain(request):
    df = pd.read_csv("assets/files/FINAL_DATASET.csv")
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.Titles).toarray()
    labels = df.Label
    features.shape
    X = features
    y = labels
    response = ensemble_func(X,y)
    return response