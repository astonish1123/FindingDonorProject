import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from functions import train_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score

if __name__=='__main__':
    data = pd.read_csv('census.csv', sep=',')
    # print(data.head())

    n_records = data.shape[0]
    n_greater_50k = len(data[data['income']=='>50K'])
    n_at_most_50k = len(data[data['income']=='<=50K'])
    greater_percent = float(n_greater_50k)/n_records
    print('Total number of records: {}'.format(n_records))
    print('Individuals making more than $50,000: {}'.format(n_greater_50k))
    print('Individuals making at most $50,000: {}'.format(n_at_most_50k))
    print('Percentage of individuals making more than $50,000: {}'.format(greater_percent))

    income_raw = data['income']
    features_raw = data.drop('income', axis=1)
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x+1))

    scaler = MinMaxScaler()
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    features_final = pd.get_dummies(data=features_log_minmax_transform, columns=['workclass', 'education_level',
                                                                                'marital-status', 'occupation',
                                                                                'relationship', 'race', 'sex', 
                                                                                'native-country'])
    income = income_raw.map({'<=50K': 0, '>50K': 1})
    encoded = list(features_final.columns)
    print('{} total features after one-hot encoding.'.format(len(encoded)))

    X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size=0.2, random_state=0)
    print('Training set has {} samples.'.format(X_train.shape[0]))
    print('Testing set has {} samples.'.format(X_test.shape[0]))

    TP = np.sum(income)
    FP = income.count() - TP
    TN = 0
    FN = 0
    accuracy = float(TP) / (TP+FP)
    recall = float(TP) / (TP+FN)
    precision = accuracy
    fscore = (1 + 0.5**2)*(precision*recall) / (0.5**2*precision+recall)
    print('Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}'.format(accuracy, fscore))

    clf_A = RandomForestClassifier(random_state=8)
    clf_B = KNeighborsClassifier()
    clf_C = SVC(random_state=8)

    samples_100 = len(y_train)
    samples_10 = int(samples_100/10)
    samples_1 = int(samples_10/10)

    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)
    
    clf = RandomForestClassifier(random_state=8)

    parameters_rand = {"max_depth": [3, None],
                       "n_estimators": list(range(10, 200)),
                       "max_features": list(range(1, X_test.shape[1]+1)),
                       "min_samples_split": list(range(2, 11)),
                       "min_samples_leaf": list(range(1, 11)),
                       "bootstrap": [True, False],
                       "criterion": ["gini", "entropy"]}


    parameters_k = {'n_neighbors': list(range(1, 5)), 
                    'leaf_size': list(range(10, 100, 10)),
                    'weights': ['uniform', 'distance']}

    parameters_svc = {'degree': list(range(3, 5)), 
                      'C': [1, 10, 100, 1000], 
                      'kernel': ['poly'], 
                      'gamma': [0.01, 0.001, 0.0001]}

    scorer = make_scorer(fbeta_score, beta=0.5)

    grid_obj = GridSearchCV(estimator=clf, param_grid=parameters_rand, scoring=scorer)
    grid_fit = grid_obj.fit(X_train, y_train)
    best_clf = grid_fit.best_estimator_

    predictions = (clf.fit(X_train, y_train).predict(X_test))
    best_predictions = best_clf.predict(X_test)

    print('Unoptimized model\n-------')
    print('Accuracy score on testing data: {:.4f}'.format(accuracy_score(y_test, predictions)))
    print('F-score on testing data: {:.4f}'.format(fbeta_score(y_test, predictions, beta=0.5)))
    print('\nOptimized Model\n-------')
    print('Final accuracy score on the tesitn data: {:.4f}'.format(accuracy_score(y_test, best_predictions)))
    print('Final F-score on the testing data: {:.4f}'.format(fbeta_score(y_test, best_predictions, beta=0.5)))


