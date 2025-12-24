import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Applying SMOTE to the training data")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42),
                # ================================================================= #
                # THE FIX IS HERE: 'use_label_encoder' HAS BEEN REMOVED             #
                # ================================================================= #
                "XGBClassifier": XGBClassifier(random_state=42, eval_metric='logloss'),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False, random_state=42),
                "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 7, 9, 11, 13]
                },
                "Random Forest": {
                    'n_estimators': [10, 25, 50, 75, 100, 125, 150],
                    'max_depth': [3, 5, 7, 9, 11, 13],
                    'max_leaf_nodes': [2, 3, 5, 7, 9]
                },
                "Gradient Boosting": {
                    'loss': ['log_loss', 'exponential'],
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'n_estimators': [25, 50, 100, 150, 200, 250],
                    'criterion': ['friedman_mse', 'squared_error'],
                    'max_depth': [3, 5, 7, 9, 11]
                },
                "Logistic Regression": {},
                "XGBClassifier": {
                    'learning_rate': [0.1, 0.05, 0.01],
                    'n_estimators': [50, 100, 150, 200]
                },
                "CatBoosting Classifier": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Classifier": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [50, 100, 150, 200]
                }
            }

            logging.info("Starting model evaluation with RandomizedSearchCV")
            model_report: dict = evaluate_models(
                X_train=X_train_resampled, y_train=y_train_resampled,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Model evaluation report: {model_report}")
            logging.info(f"Best Model Found: {best_model_name} with Accuracy: {best_model_score}")

            if best_model_score < 0.7:
                raise CustomException(f"No model found with accuracy > 95%. Best model was {best_model_name} with score {best_model_score}", sys)
            
            logging.info(f"Saving the best model: {best_model_name}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)