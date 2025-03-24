import os
import sys
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging

class ModelTrainerConfig:
    trained_model_file_path = os.path.join("D:/artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            params = {
                "Linear Regression":{},
                "Decision Tree": {'criterion': ['squared_error', 'poisson']},
                "Random Forest": {'n_estimators': [10, 50, 100]},
                "Gradient Boosting": {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]},
                "XGBRegressor": {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]},
                "CatBoosting Regressor": {'depth': [6, 10], 'iterations': [50, 100]},
                "AdaBoost Regressor": {'learning_rate': [0.1, 0.01], 'n_estimators': [50, 100]},
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with Test R²: {model_report[best_model_name]:.4f}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            return model_report[best_model_name]
        except Exception as e:
            raise CustomException(e, sys)
