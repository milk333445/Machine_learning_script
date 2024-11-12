import itertools
import pandas as pd
import joblib
import numpy as np
from collections import Counter

from sklearn.metrics import f1_score, recall_score, precision_score, log_loss
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from datapreprocess import DataPreprocessor


class Trainer:
    def __init__(self, X, y, param_space, features=None, model_type='xgboost', target_column=None, n_split=5, ros=True, minority_percentage=20):
        self.X = X
        self.y = y
        self.features = features
        self.target_column = target_column
        self.n_split = n_split
        self.ros = ros
        self.minority_percentage = minority_percentage
        self.param_space = param_space
        self.param_combinations, self.param_names = self.generate_param_combinations(param_space)
        self.model_type = model_type
        self.dataprocessor = DataPreprocessor() 

    def generate_param_combinations(self, param_space):
        param_names = list(param_space.keys())
        param_values = [param_space[param] for param in param_names]
        param_combinations = list(itertools.product(*param_values))
        return param_combinations, param_names

    def train_and_evaluate_kfold(self):
        params = []
        scores_loss = []
        scores_f1 = []
        scores_recall = []
        scores_precision = []
        scores_f1threshold = []
        feature_importance_all = []
        
        for param_set in self.param_combinations:
            param_dict = {self.param_names[i]: param_set[i] for i in range(len(param_set))}
            print('Training with params:', param_dict)
            
            score_folds_loss = []
            score_folds_f1 = []
            score_folds_recall = []
            score_folds_precision = []
            score_folds_f1threshold = []
            feature_importances = np.zeros(len(self.features))
            
            kf = KFold(n_splits=self.n_split, shuffle=True, random_state=42)
            for train_index, val_index in kf.split(self.X):
                X_train, X_val = self.X.iloc[train_index].copy(), self.X.iloc[val_index].copy()
                y_train, y_val = self.y.iloc[train_index].copy(), self.y.iloc[val_index].copy()
                
                X_train = self.dataprocessor.fit(X_train, self.features, self.target_column)
                X_val = self.dataprocessor.transform(X_val, self.features)

                if self.ros:
                    X_train, y_train = self.dataprocessor.fit_resample(X_train, y_train, self.minority_percentage)
                    print('Resampled dataset shape:', Counter(y_train))
                
                if self.model_type == 'xgboost':
                    model = XGBClassifier(**param_dict, random_state=42)
                elif self.model_type == 'lightgbm':
                    if 'max_depth' in param_dict and 'num_leaves' not in param_dict:
                        param_dict['num_leaves'] = 2 ** param_dict['max_depth']
                    model = LGBMClassifier(**param_dict, random_state=42)
                else:
                    raise ValueError("Invalid model type. Choose 'xgboost' or 'lightgbm'.")
                
                model.fit(X_train, y_train)
                
                feature_importances += model.feature_importances_   
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                thresholds = np.linspace(0, 1, 100)
                best_f1 = 0
                best_threshold = 0
                for threshold in thresholds:
                    y_pred = (y_pred_proba > threshold).astype(int)
                    score = f1_score(y_val, y_pred)
                    if score > best_f1:
                        best_f1 = score
                        best_threshold = threshold
                score_folds_f1threshold.append(best_threshold)
                
                score_logloss = log_loss(y_val, y_pred_proba)
                score_f1 = f1_score(y_val, y_pred_proba > 0.5)
                score_recall = recall_score(y_val, y_pred_proba > 0.5)
                score_precision = precision_score(y_val, y_pred_proba > 0.5, zero_division=1)
                
                score_folds_loss.append(score_logloss)
                score_folds_f1.append(score_f1)
                score_folds_recall.append(score_recall)
                score_folds_precision.append(score_precision)

            params.append(param_dict)
            scores_loss.append(np.mean(score_folds_loss))
            scores_f1.append(np.mean(score_folds_f1))
            scores_recall.append(np.mean(score_folds_recall))
            scores_precision.append(np.mean(score_folds_precision))
            scores_f1threshold.append(np.mean(score_folds_f1threshold))
            feature_importance_all.append(feature_importances / self.n_split)

            print('Params train finished')
            print('logloss: {}, f1: {}'.format(np.mean(score_folds_loss), np.mean(score_folds_f1)))
            print('recall: {}, precision: {}'.format(np.mean(score_folds_recall), np.mean(score_folds_precision)))
            print('f1 threshold:', np.mean(score_folds_f1threshold))
        
        best_idx_loss = np.argmin(scores_loss)
        best_idx_f1 = np.argmax(scores_f1)
        best_idx_f1_threshold = np.argmax(scores_f1threshold)
        
        feature_importance_df = pd.DataFrame(
            feature_importance_all[best_idx_f1],
            index=self.features,
            columns=['importance']
        ).sort_values(by='importance', ascending=False)
        
        return {
            'logloss': scores_loss[best_idx_loss],
            'f1': scores_f1[best_idx_f1],
            'recall': scores_recall[best_idx_f1],
            'precision': scores_precision[best_idx_f1],
            'f1_threshold': scores_f1threshold[best_idx_f1_threshold],
            'best_params_loss': params[best_idx_loss],
            'best_params_f1': params[best_idx_f1],
            'best_params_recall': params[np.argmax(scores_recall)],
            'best_params_precision': params[np.argmax(scores_precision)],
            'best_params_f1_threshold': params[best_idx_f1_threshold],
            'best_f1_feature_importance': feature_importance_df
        }
        
    @staticmethod
    def find_best_f1_threshold(y_true, y_pred_proba, thresholds=np.linspace(0, 1, 100)):
        best_f1 = 0
        best_threshold = 0
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            score = f1_score(y_true, y_pred)
            if score > best_f1:
                best_f1 = score
                best_threshold = threshold
        return best_threshold, best_f1

    @staticmethod
    def train_and_evaluate(X, y, param_dict, features, target_column, model_type='xgboost', train_test_split_ratio=0.2, ros=True, minority_percentage=20, save_path=None):
        dataprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_split_ratio, random_state=42)
        X_train = dataprocessor.fit(X_train, features, target_column, save_target_encoding_path='target_encoding')
        X_test = dataprocessor.transform(X_test, features)
        
        if ros:
            X_train, y_train = dataprocessor.fit_resample(X_train, y_train, minority_percentage)
            print('Resampled dataset shape:', Counter(y_train))
        
        if model_type == 'lightgbm' and 'max_depth' in param_dict and 'num_leaves' not in param_dict:
            param_dict['num_leaves'] = 2 ** param_dict['max_depth']
        
        if model_type == 'xgboost':
            model = XGBClassifier(**param_dict, random_state=42)
        elif model_type == 'lightgbm':
            model = LGBMClassifier(**param_dict, random_state=42)
        else:
            raise ValueError("Invalid model type. Choose 'xgboost' or 'lightgbm'.")
        
        model.fit(X_train, y_train)
        
        if save_path is not None:
            if model_type == 'xgboost':
                save_path = save_path + '.bin'
                model.save_model(save_path)
            elif model_type == 'lightgbm':
                save_path = save_path + '.pkl'
                joblib.dump(model, save_path)
                
        
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame(
            feature_importances, index=features, columns=['importance']
        ).sort_values(by='importance', ascending=False)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        best_threshold, best_f1 = Trainer.find_best_f1_threshold(y_test, y_pred_proba)
        
        score_logloss = log_loss(y_test, y_pred_proba)
        score_f1 = f1_score(y_test, y_pred_proba > 0.5)
        score_recall = recall_score(y_test, y_pred_proba > 0.5)
        score_precision = precision_score(y_test, y_pred_proba > 0.5, zero_division=1)
        
        results = {
            'logloss': score_logloss,
            'f1': score_f1,
            'recall': score_recall,
            'precision': score_precision,
            'best_threshold': best_threshold,
            'feature_importance': feature_importance_df
        }
        
        return results