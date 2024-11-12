import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from datapreprocess import DataPreprocessor


def load_model(model_path, model_type):
    if model_type == "xgboost":
        model = XGBClassifier()
        model.load_model(model_path)
    elif model_type == "lightgbm":
        model = joblib.load(model_path)
    else:
        raise ValueError("Invalid model type. Choose 'xgboost' or 'lightgbm'.")
    print(f'Model loaded from {model_path}')
    return model

def ensemble_predict_proba(X, models):
    total_proba = np.zeros(X.shape[0])
    for model in models:
        total_proba += model.predict_proba(X)[:, 1]
    return total_proba / len(models)
    

if __name__ == "__main__":
    model0 = load_model('test_model.bin', 'xgboost')
    model1 = load_model('test_model.pkl', 'lightgbm')
    datapath = 'dataset.xlsx'
    data = pd.read_excel(datapath)
    features = [
        'L1', 'L2', 'L3', 'L4', 'L5', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
        'B1', 'B2', 'B3', 'B4', 'Z1', 'Z2', 'Z3', 'ANNUAL_INCOME_AMT', 
        'TOTAL_POLICY_DISPREM', 'TOTAL_POLICY_PREM', 'CHANNEL_BANK', 'CHANNEL_SEC', 
        'INS_AMT', 'AGE', 'OCCUPATION_DESC_missing', 'ANNUAL_INCOME_AMT_missing', 
        'MAILING_ZIP_CODE_missing', 'NW_SEGMENT_DESC_missing', 
        'EDUCATION_DESC_ENCODED', 'OCCUPATION_DESC_ENCODED', 'GENDER_CODE_ENCODED',
        'MAILING_ZIP_CODE_ENCODED', 'NW_SEGMENT_DESC_ENCODED'
    ]
    target_column = 'CHANNEL_LIFE'
    datapreprocessor = DataPreprocessor()
    # X = datapreprocessor.fit(data, features, target_column, save_target_encoding_path='target_encoding')
    X = datapreprocessor.transform(data, features, target_encoding_path='target_encoding.pkl')
    avg_proba = ensemble_predict_proba(X, [model0, model1])
    
    avg = (avg_proba > 0.5).astype(int)
    print(avg)
    
    
    # single_data = X.iloc[[0]]
    # xgboost_prediction_proba = model.predict_proba(X)[:, 1]
    # xgboost_prediction = model.predict(X)
    # print(xgboost_prediction)
    # lightgbm_prediction_proba  = model.predict_proba(single_data)[:, 1][0]
    # lightgbm_prediction = int(lightgbm_prediction_proba > 0.5)