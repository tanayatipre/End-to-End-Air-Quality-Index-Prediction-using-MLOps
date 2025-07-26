import os
import pandas as pd
from MLProject import logger
from MLProject.entity.config_entity import DataValidationConfig
from box import ConfigBox

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = True

            data = pd.read_csv(self.config.csv_file_path)
            logger.info(f"Data loaded from {self.config.csv_file_path}. Data columns: {list(data.columns)}")
            
            all_cols = set(data.columns)
            logger.info(f"Set of columns from loaded data: {all_cols}") 

            # Debugging the schema part
            logger.info(f"Raw self.config.all_schema type: {type(self.config.all_schema)}") 
            logger.info(f"Raw self.config.all_schema content: {self.config.all_schema}") 

            if isinstance(self.config.all_schema, ConfigBox) or isinstance(self.config.all_schema, dict):
                all_schema = set(self.config.all_schema.keys())
            else:
                logger.error(f"self.config.all_schema is not a dict or ConfigBox: {type(self.config.all_schema)}")
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                # Raising an error here stops further processing with an invalid schema
                raise TypeError("Schema configuration is not in expected dictionary format.")
            
            logger.info(f"Set of columns from schema: {all_schema}") 


            # Check for missing columns in data compared to schema
            missing_in_data = all_schema - all_cols
            if missing_in_data:
                logger.error(f"Validation Error: Columns missing in dataset (from schema): {missing_in_data}")
                validation_status = False
            
            extra_in_data = all_cols - all_schema 
            if extra_in_data:
                logger.warning(f'Validation Warning: Extra columns found in dataset not in schema: {extra_in_data}')
                # You might choose to set validation_status = False here if extra columns are critical errors


            for col, dtype in self.config.all_schema.items():
                if col in all_cols:

                    pass
                else:
                    logger.error(f"Validation Error: Schema column '{col}' not found in dataset. (Detailed check)")
                    validation_status = False 

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            
            if validation_status:
                logger.info("Data validation completed: Schema validation successful.") # Improved log message
            else:
                logger.error("Data validation completed: Schema validation FAILED. Check logs for details.") # Improved log message

            return validation_status
            
        except Exception as e:
            logger.exception(f"An error occurred during data validation: {e}") # Changed to logger.exception for full traceback
            raise e