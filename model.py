# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import time
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import plot_roc_curve, f1_score, roc_curve, roc_auc_score,auc, accuracy_score, classification_report, confusion_matrix


dataset = pd.read_csv('stroke_df_local.csv', index_col=0)
dataset.drop(dataset[(dataset['gender']=='Other')].index, inplace=True)
# dataset.drop(columns=['patient','residence'], inplace=True)
# dataset.drop(dataset.columns[[0]], axis=1)

important_features = ['gender', 'age', 'bmi', 'avg_glucose_level', 'hypertension',
       'heart_disease', 'ever_married', 'work_type', 'smoking']
# features = pd.DataFrame(features, columns = ['gender', 'age', 'bmi', 'avg_glucose_level', 'hypertension',
#        'heart_disease', 'ever_married', 'work_type', 'smoking'])


X = dataset[important_features]
y = dataset['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state=42, stratify=y)

# X.to_csv('wtf.csv')

NUMERICAL_FEATURES = ['age', 'bmi', 'avg_glucose_level',]
CATEGORICAL_FEATURES = [
    'gender',
    'work_type',
    'hypertension',
    'heart_disease',
    'ever_married',
#     'residence',
    'smoking']
CLASSIFIER = XGBClassifier(n_estimators=200, max_depth=1, verbosity=0, scale_pos_weight=45)

def get_model_pipeline() -> Pipeline:
    """return user score predcition model pipeline"""

    num_preprocessor = Pipeline([
#                 ("std_scaler", StandardScaler()),
                ('robust_scaler', RobustScaler())
#             ("pca", PCA(0.95))
        ])
    

    preprocessor = ColumnTransformer(
        [
            ('num_features', num_preprocessor, NUMERICAL_FEATURES),
            ('categ_features', OneHotEncoder(), CATEGORICAL_FEATURES),
        ], 
        remainder='drop'
    )
    
    model_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ('knn_imputer', KNNImputer(n_neighbors=4, weights='uniform')),
                ('current_model', CLASSIFIER)
            ],
        )

    return model_pipeline

model = get_model_pipeline()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

CM = confusion_matrix(y_test, y_pred)

current_time = time.strftime("%Y%m%d-%H%M%S")
MODEL_FILEPATH = f"model.pkl"
REPORT_FILEPATH = f"report.csv"

pd.DataFrame(report).round(3).transpose().to_csv(REPORT_FILEPATH)

# save model
joblib.dump(model, MODEL_FILEPATH)


# # Saving model to disk
# pickle.dump(pipe_deoloy, open('model.pkl','wb'))

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''