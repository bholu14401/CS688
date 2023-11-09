"""
    Program: Use Various sklearn Models for Classification of Newsgroup Data
    Models:
    Data:
        Features:
        Target:
"""

import os
import io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import random

random.seed(0)


def read_text_file(path):
    """ Read from Text File """
    with io.open(path, "r") as f:
        content = f.read().replace('\n', '')  # read all the lines of the text file
    return content


def model_train(features_train, tags_train, model_name, models_path):
    """ Train Classifier ML Model """

    if model_name == "model_nn":  # 1) Use NN Model Classifier (24/67)
        model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500, random_state=1,
                              solver='adam',  # solver{‘lbfgs’, ‘sgd’, ‘adam’} default=’adam’
                              activation='relu')  # activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
    elif model_name == "model_nb":  # 2) Use Naive Bayes Classifier
        model = GaussianNB()
    elif model_name == "model_nbb":  # 3) Use Bernoulli Naive Bayes Classifier
        model = BernoulliNB()
    elif model_name == "model_mnb":  # 4) Use Multinomial Naive Bayes Classifier
        model = MultinomialNB()
    elif model_name == "model_knn":  # 5) Use KNN
        model = KNeighborsClassifier(n_neighbors=2)
    elif model_name == "model_rf":  # 6) Use Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100,
                                       random_state=0)
    elif model_name == "model_lr":  # 7) Use Logistic Regression Classifier (solver: newton-cg, lbfgs, liblinear, sag, saga)
        model = LogisticRegression(
            solver='liblinear')
    elif model_name == "model_svm":  # 8) Use SVM with Probability Classifier (kernels: linear, poly, rbf, sigmoid
        model = SVC(kernel='rbf', gamma='auto',
                    probability=True)
    elif model_name == "model_svm1":  # 9) Use SVM with "rbf" Kernel Classifier
        model = SVC(kernel='rbf', random_state=1, gamma=0.001, C=20)
    elif model_name == "model_xgb":  # 10) XGBoost
        model = xgb.XGBClassifier(colsample_bytree=0.7, gamma=0, learning_rate=0.2, max_depth=7, reg_alpha=0,
                                  reg_lambda=1, subsample=0.8)
    else:
        print(" ~~~ Wrong Model Name!!! ~~~~ ")
        pass

    labels = preprocessing.LabelEncoder()
    tags_train = labels.fit_transform(tags_train)

    model.fit(features_train.toarray(), tags_train)  # Train the model on the training features
    # dump(model, models_path + model_name)  # Save the model

    return model


def model_test(model, features_test, tags_test):
    labels = preprocessing.LabelEncoder()
    tags_test_fit = labels.fit_transform(tags_test)

    predicted_test, accuracy_test, precision_test, recall_test, f1_test, cm_test = calculate_metrics(model,
                                                                                                     features_test.toarray(),
                                                                                                     tags_test_fit)
    return predicted_test, accuracy_test, precision_test, recall_test, f1_test, cm_test


def calculate_metrics(model, features, tags):
    """ Calculate Classification Effectiveness """
    predictions = model.predict(features)
    
    accuracy = accuracy_score(tags, predictions)
    precision = precision_score(tags, predictions)
    recall = recall_score(tags, predictions)
    F1_score = f1_score(tags, predictions)
    
    cm = confusion_matrix(tags, predictions)
    tn, fp, fn, tp = cm.ravel()  # Get TN, FP, FN, TP
    row_names, col_names = ["Actual Negative", "Actual Positive"], ["Predict Negative", "Predict Positive"]
    cm = pd.DataFrame(cm, index=row_names, columns=col_names)
    
    return predictions, accuracy, precision, recall, F1_score, cm


def classify_ng(model_name, transfer, x_train, x_test, y_train, y_test):
    x_train = transfer.fit_transform(x_train)  # Vectorize Train
    x_test = transfer.transform(x_test)  # Vectorize Test

    model = model_train(x_train, y_train, model_name, models_path="")  # Train Model
    predicted_test, accuracy_test, precision_test, recall_test, f1_test, cm_test = model_test(model, x_test, y_test)
    return predicted_test, accuracy_test, precision_test, recall_test, f1_test, cm_test


def print_scores(accuracy_test, precision_test, recall_test, f1_test, cm_test):
    print("Accuracy: {:.2f}%".format(accuracy_test * 100))
    print("Precision: {:.2f}%".format(precision_test * 100))
    print("Recall: {:.2f}%".format(recall_test * 100))
    print("F1-Score: {:.2f}%".format(f1_test * 100))
    cm_data = {"Predict Negative": ['TN', 'FN'],  # First Column
               "Predict Positive": ['FP', 'TP']}  # Second Column
    row_names, col_names = ["Actual Negative", "Actual Positive"], ["Predict Negative", "Predict Positive"]
    CM = pd.DataFrame(cm_test, index=pd.Index(row_names), columns=pd.Index(col_names)).to_string()
    print(CM)
    print(cm_test)
    return True


def populate_df(results, model_name, accuracy, precision, recall, f1, cm, flag):
    if flag == 'DTM':
        results['DTM_Accuracy'].loc[model_name] = round(accuracy * 100, 1)
        results['DTM_Precision'].loc[model_name] = round(precision * 100, 1)
        results['DTM_Recall'].loc[model_name] = round(recall * 100, 1)
        results['DTM_F1-Score'].loc[model_name] = round(f1 * 100, 1)
        results['DTM_TN'].loc[model_name] = round(cm.iloc[0][0], 1)
        results['DTM_FP'].loc[model_name] = round(cm.iloc[0][1], 1)
        results['DTM_FN'].loc[model_name] = round(cm.iloc[1][0], 1)
        results['DTM_TP'].loc[model_name] = round(cm.iloc[1][1], 1)
    elif flag == 'TFIDF':
        results['TFIDF_Accuracy'].loc[model_name] = round(accuracy * 100, 1)
        results['TFIDF_Precision'].loc[model_name] = round(precision * 100, 1)
        results['TFIDF_Recall'].loc[model_name] = round(recall * 100, 1)
        results['TFIDF_F1-Score'].loc[model_name] = round(f1 * 100, 1)
        results['TFIDF_TN'].loc[model_name] = round(cm.iloc[0][0], 1)
        results['TFIDF_FP'].loc[model_name] = round(cm.iloc[0][1], 1)
        results['TFIDF_FN'].loc[model_name] = round(cm.iloc[1][0], 1)
        results['TFIDF_TP'].loc[model_name] = round(cm.iloc[1][1], 1)
    else:
        print(" ~~~ Wrong Flag Name!!! ~~~~ ")
    return results


if __name__ == '__main__':
    """ 1) Define Train/Test Paths to Newsgroup Data """
    newsgroup1 = "/sci.space"  # Newsgroup to analyze
    newsgroup2 = "/rec.autos"  # Newsgroup to analyze
    path_to_20Newsgroups = "C:/Users/bhargav/Downloads/HW 4 NewsG Classification Py(1)/HW 4 NewsG Classification Py/data/20Newsgroups"
    train_folder = "/20news-bydate-train"
    test_folder = "/20news-bydate-test"

    doc1_train_path = path_to_20Newsgroups+train_folder + newsgroup1
    doc1_test_path = path_to_20Newsgroups+test_folder + newsgroup1
    doc2_train_path = path_to_20Newsgroups+train_folder + newsgroup2
    doc2_test_path = path_to_20Newsgroups+test_folder + newsgroup2

    doc1_train_files = os.listdir(doc1_train_path)
    doc1_test_files = os.listdir(doc1_test_path)
    doc2_train_files = os.listdir(doc2_train_path)
    doc2_test_files = os.listdir(doc2_test_path)

    count = 100  # Limit The Number of documents

    """ 2) Get Train/Test Data into a List """
    doc1_train, doc1_test, doc2_train, doc2_test = [], [], [], []
    for docs in range(count):
        temp = read_text_file(doc1_train_path + '/' + doc1_train_files[docs])
        doc1_train.append(temp)

        temp = read_text_file(doc1_test_path + '/' + doc1_test_files[docs])
        doc1_test.append(temp)

        temp = read_text_file(doc2_train_path + '/' + doc2_train_files[docs])
        doc2_train.append(temp)

        temp = read_text_file(doc2_test_path + '/' + doc2_test_files[docs])
        doc2_test.append(temp)

    train_data = doc1_train + doc2_train
    test_data = doc1_test + doc2_test

    # Use Tag=1 for newsgroup1 and Tag=2 for newsgroup2
    doc1_tag, doc2_tag = 'Positive', 'Negative'
    train_tags = [doc1_tag] * len(doc1_train) + [doc2_tag] * len(doc2_train)
    test_tags = [doc1_tag] * len(doc1_test) + [doc2_tag] * len(doc2_test)

    """ 3) Define All the Models"""
    all_models = {1: 'model_nn', 2: 'model_nb', 3: 'model_nbb', 4: 'model_mnb', 5: 'model_knn', 6: 'model_rf',
                  7: 'model_lr', 8: 'model_svm', 9: 'model_svm1', 10: 'model_xgb'}

    """ 4) Define DF for the Results """
    cols = ['DTM_Accuracy', 'DTM_Precision', 'DTM_Recall', 'DTM_F1-Score', 'DTM_TN', 'DTM_FP', 'DTM_FN', 'DTM_TP',
            'TFIDF_Accuracy', 'TFIDF_Precision', 'TFIDF_Recall', 'TFIDF_F1-Score', 'TFIDF_TN', 'TFIDF_FP', 'TFIDF_FN', 'TFIDF_TP']
    rows = all_models.values()
    results = pd.DataFrame(columns=cols, index=rows)

    """ 4) Loop over All Models """
    for model_number in all_models.keys():
        """ Select a Model & Make Predictions """
        model_name = all_models[model_number]
        print("")
        print("~~~~~~~~~~~~ Model Name:" + model_name + ' ~~~~~~~~~~~~')
        predicted_test, accuracy_test, precision_test, recall_test, f1_test, cm_test = classify_ng(model_name,
                                                                                                   CountVectorizer(),
                                                                                                   train_data,
                                                                                                   test_data,
                                                                                                   train_tags,
                                                                                                   test_tags)  # 1) Use DTM
        """ ) Populate Results Dataframe with DTM Metrics"""
        results = populate_df(results, model_name, accuracy_test, precision_test, recall_test, f1_test,
                              cm_test, flag="DTM")

        print("   ======== Using DTM =======")
        print_scores(accuracy_test, precision_test, recall_test, f1_test, cm_test)
        predicted_test, accuracy_test, precision_test, recall_test, f1_test, cm_test = classify_ng(model_name,
                                                                                                   TfidfVectorizer(),
                                                                                                   train_data,
                                                                                                   test_data,
                                                                                                   train_tags,
                                                                                                   test_tags)  # 2) Use TFIDF
        print("   ======== Using TFIDF ========")
        print_scores(accuracy_test, precision_test, recall_test, f1_test, cm_test)

        """ ) Populate Results Dataframe with TFIDF Metrics """
        results = populate_df(results, model_name, accuracy_test, precision_test, recall_test, f1_test,
                              cm_test, flag="TFIDF")

results.to_csv('C:/Users/bhargav/Downloads/HW 4 NewsG Classification Py(1)/HW 4 NewsG Classification Py/data/MyResults.csv')
print("End")
