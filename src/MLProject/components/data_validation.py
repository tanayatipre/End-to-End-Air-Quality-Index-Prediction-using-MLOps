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
            logger.info(f"Set of columns from loaded data: {all_cols}") # Debug print

            # Debugging the schema part
            logger.info(f"Raw self.config.all_schema type: {type(self.config.all_schema)}") # Debug print
            logger.info(f"Raw self.config.all_schema content: {self.config.all_schema}") # Debug print

            # Ensure all_schema is correctly extracted as a set of keys
            # Added a type check for robustness, though ConfigBox should handle keys()
            if isinstance(self.config.all_schema, ConfigBox) or isinstance(self.config.all_schema, dict):
                all_schema = set(self.config.all_schema.keys())
            else:
                logger.error(f"self.config.all_schema is not a dict or ConfigBox: {type(self.config.all_schema)}")
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                # Raising an error here stops further processing with an invalid schema
                raise TypeError("Schema configuration is not in expected dictionary format.")
            
            logger.info(f"Set of columns from schema: {all_schema}") # Debug print


            # Check for missing columns in data compared to schema
            missing_in_data = all_schema - all_cols
            if missing_in_data:
                logger.error(f"Validation Error: Columns missing in dataset (from schema): {missing_in_data}")
                validation_status = False
            
            # Check for extra columns in data not in schema
            # FIX: Changed assignment operator '=' to set difference operator '-'
            extra_in_data = all_cols - all_schema # CORRECTED LINE
            if extra_in_data:
                logger.warning(f'Validation Warning: Extra columns found in dataset not in schema: {extra_in_data}')
                # You might choose to set validation_status = False here if extra columns are critical errors

            # Check if all schema columns are present and data types match (optional, but robust)
            # This loop implicitly re-checks missing_in_data and ensures type validation if you add it.
            for col, dtype in self.config.all_schema.items():
                if col in all_cols:
                    # Optional: Add type validation here if schema.yaml includes exact types
                    # E.g., if str(data[col].dtype) != dtype:
                    #     logger.error(f"Validation Error: Column '{col}' type mismatch. Expected {dtype}, got {data[col].dtype}")
                    #     validation_status = False
                    pass
                else:
                    # This case should largely be caught by missing_in_data check above, but provides specific log if needed
                    logger.error(f"Validation Error: Schema column '{col}' not found in dataset. (Detailed check)")
                    validation_status = False # Ensure status is set to False if this condition is met

            # Write the final validation status after all checks are done
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