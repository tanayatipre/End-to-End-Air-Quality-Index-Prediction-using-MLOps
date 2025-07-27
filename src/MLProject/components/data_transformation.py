import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer # Import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from MLProject import logger
from MLProject.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self) -> ColumnTransformer:
        '''
        This function is responsible for data transformation.
        It returns a ColumnTransformer object.
        '''
        numerical_cols = self.config.numerical_cols
        categorical_cols = self.config.categorical_cols
        columns_to_log_transform = self.config.columns_to_log_transform
        
        # Identify numerical columns that need log transformation *within* the numerical pipeline
        numerical_cols_for_scaling = [col for col in numerical_cols if col not in columns_to_log_transform]
        numerical_cols_for_log_and_scaling = [col for col in numerical_cols if col in columns_to_log_transform]

        # Pipeline for numerical columns that are ONLY scaled
        numerical_transformer_only_scaled = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline for numerical columns that are LOG-TRANSFORMED and scaled
        numerical_transformer_log_scaled = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), # Impute before log transform
            ('log1p', FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)), # Apply log1p
            ('scaler', StandardScaler())
        ])
        
        # Define preprocessing steps for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Impute categorical
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_log_transform_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log1p', FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)),
            ('scaler', StandardScaler())
        ])

        numerical_standard_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Separate numerical_cols into those that need log transform and those that don't
        num_cols_to_log = [col for col in numerical_cols if col in columns_to_log_transform]
        num_cols_no_log = [col for col in numerical_cols if col not in columns_to_log_transform]
        
        logger.info(f"CT Config - numerical_cols (from params): {numerical_cols}")
        logger.info(f"CT Config - columns_to_log_transform (from params): {columns_to_log_transform}")
        logger.info(f"CT Config - num_cols_to_log (derived for CT): {num_cols_to_log}")
        logger.info(f"CT Config - num_cols_no_log (derived for CT): {num_cols_no_log}")
        logger.info(f"CT Config - categorical_cols (from params): {categorical_cols}")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num_log', numerical_log_transform_pipeline, num_cols_to_log),
                ('num_std', numerical_standard_pipeline, num_cols_no_log),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='passthrough' 
        )
        
        return preprocessor

    def initiate_data_transformation(self):
        try:
            data = pd.read_csv(self.config.data_path)
            logger.info(f"Original data loaded. Shape: {data.shape}")

            # 1. Drop rows with missing values in 'AQI' and 'AQI_Bucket'
            initial_rows = data.shape[0]
            data.dropna(subset=['AQI', 'AQI_Bucket'], inplace=True)
            logger.info(f"Dropped {initial_rows - data.shape[0]} rows with NaN in AQI or AQI_Bucket. New shape: {data.shape}")

            # Define target column
            target_column_name = self.config.target_column
            
            # Feature Engineering for Date 
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data['Year'] = data['Date'].dt.year
                data['Month'] = data['Date'].dt.month
                data['Day'] = data['Date'].dt.day
                data['DayOfWeek'] = data['Date'].dt.dayofweek
                data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
                logger.info("Date features engineered.")

            
            # Create X and y AFTER date engineering, but BEFORE any other drops or log transforms
            # that ColumnTransformer should handle.
            X = data.drop(columns=[target_column_name, 'AQI_Bucket'], errors='ignore') 
            y = data[target_column_name]

            # 'Xylene' and the original 'Date' column.
            columns_to_drop_from_X = [col for col in self.config.columns_to_drop_after_feature_eng if col != 'AQI_Bucket']

            # Drop specified columns from X (like Xylene, and the original Date if listed)
            X = X.drop(columns=[col for col in columns_to_drop_from_X if col in X.columns], errors='ignore')
            logger.info(f"Columns explicitly dropped from features (X) before CT: {columns_to_drop_from_X}. New X shape: {X.shape}")

            # Now perform train-test split on the prepared X and y
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=42
            )
            logger.info(f"Data split into train ({X_train.shape}) and test ({X_test.shape}) sets.")
            
            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()
            
            logger.info(f"X_train columns before ColumnTransformer fit_transform: {list(X_train.columns)}")                                                                              

            # Fit and transform X_train
            X_train_transformed = preprocessor_obj.fit_transform(X_train)

            X_test_transformed = preprocessor_obj.transform(X_test)
            logger.info("ColumnTransformer fitted on X_train and transformed X_train, X_test.")

            if self.config.target_column in self.config.columns_to_log_transform:
                logger.info(f"Applying log1p transformation to target column '{self.config.target_column}' in training and test sets.")
                y_train = np.log1p(y_train)
                y_test = np.log1p(y_test)

            else:
                logger.info(f"Target column '{self.config.target_column}' is NOT configured for log transformation.")

            try:
                # This works for sklearn >= 1.0
                transformed_feature_names = preprocessor_obj.get_feature_names_out()
            except AttributeError:
                logger.warning("get_feature_names_out() not available. Transformed DataFrame column names might be generic.")
                transformed_feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

            # Convert transformed arrays back to DataFrames with proper column names
            X_train_df = pd.DataFrame(X_train_transformed, columns=transformed_feature_names, index=X_train.index)
            X_test_df = pd.DataFrame(X_test_transformed, columns=transformed_feature_names, index=X_test.index)

            # Join X_train_df with y_train and X_test_df with y_test to create final train/test CSVs
            # Ensure y_train/y_test are Series for concat. If they are already series, .to_frame() is fine.
            train_df = pd.concat([X_train_df, y_train.to_frame(name=target_column_name)], axis=1) # explicit name
            test_df = pd.concat([X_test_df, y_test.to_frame(name=target_column_name)], axis=1) # explicit name
            
            # Save the processed data
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info(f"Transformed train data saved to {self.config.train_data_path}. Shape: {train_df.shape}")
            logger.info(f"Transformed test data saved to {self.config.test_data_path}. Shape: {test_df.shape}")

            # Save the preprocessor object
            joblib.dump(preprocessor_obj, os.path.join(self.config.root_dir, self.config.preprocessor_name))
            logger.info(f"Preprocessor object saved to {self.config.root_dir}/{self.config.preprocessor_name}")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test
            )

        except Exception as e:
            logger.exception(f"An error occurred during data transformation: {e}")
            raise e