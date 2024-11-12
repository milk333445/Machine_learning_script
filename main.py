import pandas as pd
import matplotlib.pyplot as plt

from Trainer import Trainer


def main():
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
    
    X = data
    y = data[target_column]
    
    param_space = {
        'learning_rate': [0.1],
        'max_depth': [7],
        'n_estimators': [10, 20],
        'min_child_weight': [5],
    }
    
    trainer = Trainer(
        X, 
        y, 
        param_space, 
        model_type='lightgbm',
        features=features, 
        target_column=target_column, 
        n_split=5, 
        ros=True, 
        minority_percentage=20)
    results = trainer.train_and_evaluate_kfold()
    
    print('logloss: {}, f1: {}'.format(results['logloss'], results['f1']))
    print('recall: {}, precision: {}'.format(results['recall'], results['precision']))
    print('best params loss: ', results['best_params_loss'])
    print('best params f1: ', results['best_params_f1'])
    print('best params recall: ', results['best_params_recall'])
    print('best params precision: ', results['best_params_precision'])
    print('feature importances: ', results['best_f1_feature_importance'])
    
    # ----------------- single train test split -----------------
    
    best = {'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 10, 'min_child_weight': 5}
    results = Trainer.train_and_evaluate(X, y, best, features, target_column, model_type='lightgbm', train_test_split_ratio=0.1, ros=True, minority_percentage=20, save_path="test_model")
    print(results)
    
    # draw feature importance
    # plt.figure(figsize=(10, 8))
    # plt.barh(results['best_f1_feature_importance'].index, results['best_f1_feature_importance']['importance'], align='center')
    # plt.xlabel('Importance')
    # plt.ylabel('Features')
    # plt.title('Feature Importance')
    # plt.gca().invert_yaxis()  
    # plt.tight_layout()
    # plt.show()
    
    
if __name__ == '__main__':
    main()