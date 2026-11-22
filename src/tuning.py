
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def get_parameter_grids():

    """
    Defines hyperparameter combinations for all six models.
    """

    parameter_grids = {

        "Decision Tree": {
            "classifier__max_depth": [None, 5, 10, 15],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4]
        },

        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        },

        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        },

        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        },

        "Bagging": {
            "classifier__n_estimators": [25, 50, 100],
            "classifier__max_samples": [0.7, 1.0],
            "classifier__max_features": [0.7, 1.0]
        },

        "Stacking": {
            "classifier__final_estimator__C": [0.1, 1, 10],
            "classifier__passthrough": [False, True]
        }

    }

    return parameter_grids


def tune_models(models, X_train, y_train):

    """
    Performs GridSearchCV for every model.
    Selects the best hyperparameters using macro F1-score.
    """

    parameter_grids = get_parameter_grids()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    tuned_models = {}
    tuning_results = {}

    for name, model in models.items():

        print(f"\nTuning {name}...")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=parameter_grids[name],
            scoring="f1_macro",
            cv=cv,
            n_jobs=1
        )

        grid_search.fit(X_train, y_train)

        tuned_models[name] = grid_search.best_estimator_

        tuning_results[name] = {
            "Best CV F1": grid_search.best_score_,
            "Best Parameters": grid_search.best_params_
        }

        print("Best CV F1:", round(grid_search.best_score_, 4))
        print("Best Parameters:", grid_search.best_params_)

    return tuned_models, tuning_results
