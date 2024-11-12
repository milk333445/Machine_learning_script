import pandas as pd
import joblib

from imblearn.over_sampling import RandomOverSampler


class DataPreprocessor:
    def __init__(self):
        self.education_mapping = {'Unknown': 0, '國中': 1, '高中職': 2, '大專': 3, '碩士': 4, '博士': 5, '其他': 6}
        self.gender_mapping = {'M': 1, 'F': 0, 'O': -1}
        self.nw_segment_mapping = {'潛力': 4, '價值Plus': 3, '價值一般': 2, '價值新戶': 1, '無貼標': 0}
        self.occupation_target_encoding = None
        self.zip_code_target_encoding = None
        self.features = [
        'L1', 'L2', 'L3', 'L4', 'L5', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
        'B1', 'B2', 'B3', 'B4', 'Z1', 'Z2', 'Z3', 'ANNUAL_INCOME_AMT', 
        'TOTAL_POLICY_DISPREM', 'TOTAL_POLICY_PREM', 'CHANNEL_BANK', 'CHANNEL_SEC', 
        'INS_AMT', 'AGE', 'OCCUPATION_DESC_missing', 'ANNUAL_INCOME_AMT_missing', 
        'MAILING_ZIP_CODE_missing', 'NW_SEGMENT_DESC_missing', 
        'EDUCATION_DESC_ENCODED', 'OCCUPATION_DESC_ENCODED', 'GENDER_CODE_ENCODED',
        'MAILING_ZIP_CODE_ENCODED', 'NW_SEGMENT_DESC_ENCODED'
        ]
        self.target_column = 'CHANNEL_LIFE'

    def fit(self, data, features=None, target_column=None, save_target_encoding_path=None):
        if features is None:
            features = self.features
        if target_column is None:
            target_column = self.target_column
        missing_values = data.isnull().sum()
        for column in missing_values[missing_values > 0].index:
            data[f'{column}_missing'] = data[column].isnull().astype(int)
        
        data['EDUCATION_DESC_ENCODED'] = data['EDUCATION_DESC'].map(self.education_mapping).fillna(-1)

        self.occupation_target_encoding = data.groupby('OCCUPATION_DESC')[target_column].mean()
        data['OCCUPATION_DESC_ENCODED'] = data['OCCUPATION_DESC'].map(self.occupation_target_encoding).fillna(-1)

        data['GENDER_CODE_ENCODED'] = data['GENDER_CODE'].map(self.gender_mapping).fillna(-1)

        self.zip_code_target_encoding = data.groupby('MAILING_ZIP_CODE')[target_column].mean()
        data['MAILING_ZIP_CODE_ENCODED'] = data['MAILING_ZIP_CODE'].map(self.zip_code_target_encoding).fillna(-1)

        data['NW_SEGMENT_DESC_ENCODED'] = data['NW_SEGMENT_DESC'].map(self.nw_segment_mapping).fillna(-1)

        X_train = data[features]
        
        if save_target_encoding_path is not None:
            save_target_encoding_path = save_target_encoding_path + '.pkl'
            joblib.dump(
                {
                    'occupation_target_encoding': self.occupation_target_encoding,
                    'zip_code_target_encoding': self.zip_code_target_encoding,
                }, save_target_encoding_path
            )
            print(f'Target encoding saved to {save_target_encoding_path}')
            
        return X_train
    
    def load_encoding(self, load_path):
        encodings = joblib.load(load_path)
        self.occupation_target_encoding = encodings['occupation_target_encoding']
        self.zip_code_target_encoding = encodings['zip_code_target_encoding']
        print(f'Loaded target encoding from {load_path}')
    
    def transform(self, data, features=None, target_encoding_path=None):
        """
        Test data
        """
        if target_encoding_path is not None:
            self.load_encoding(target_encoding_path)
        if features is None:
            features = self.features
        missing_values = data.isnull().sum()
        for column in missing_values[missing_values > 0].index:
            data[f'{column}_missing'] = data[column].isnull().astype(int)
        
        data['EDUCATION_DESC_ENCODED'] = data['EDUCATION_DESC'].map(self.education_mapping).fillna(-1)

        data['OCCUPATION_DESC_ENCODED'] = data['OCCUPATION_DESC'].map(self.occupation_target_encoding).fillna(-1)

        data['GENDER_CODE_ENCODED'] = data['GENDER_CODE'].map(self.gender_mapping).fillna(-1)

        data['MAILING_ZIP_CODE_ENCODED'] = data['MAILING_ZIP_CODE'].map(self.zip_code_target_encoding).fillna(-1)

        data['NW_SEGMENT_DESC_ENCODED'] = data['NW_SEGMENT_DESC'].map(self.nw_segment_mapping).fillna(-1)
        
        X_test = data[features]
        return X_test
    
    def fit_resample(self, X, y, minority_percentage=100):
        majority_class_count = y.value_counts().max()
        minority_class_count = y.value_counts().min()
        
        target_minority_count = int(majority_class_count * (minority_percentage / 100))
        if target_minority_count < minority_class_count:
            print('Minority class count is already greater than target minority count')
            return X, y
        
        ros = RandomOverSampler(sampling_strategy={y.value_counts().idxmin(): target_minority_count}, random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        return X_resampled, y_resampled

    
    
if __name__ == '__main__':
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
    print(len(features))
    target = 'CHANNEL_LIFE'
    
    preprocessor = DataPreprocessor()
    X_train = preprocessor.fit(data, features, target)
    y_train = data[target]
    print(y_train.value_counts())
    X_resampled, y_resampled = preprocessor.fit_resample(X_train, y_train, minority_percentage=20)
    
    # X_resampled, y_resampled = preprocessor.fit_resample(X, y, minority_percentage=20)
    
    # print(X_resampled.shape, y_resampled.shape)
    # print(y_resampled.value_counts())
    
        
        