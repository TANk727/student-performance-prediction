
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def create_preprocessor(scale_numeric=False):
    """
    Handles categorical and numerical features.
    """

    numeric_transformer = (
        StandardScaler() if scale_numeric else "passthrough"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include="object")
            ),
            (
                "numerical",
                numeric_transformer,
                make_column_selector(dtype_exclude="object")
            )
        ]
    )

    return preprocessor


def build_models():
    """
    Creates six machine learning models.
    """

    models = {

        # 1. Decision Tree
        "Decision Tree": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("classifier", DecisionTreeClassifier(
                random_state=42
            ))
        ]),

        # 2. Random Forest
        "Random Forest": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))
        ]),

        # 3. K-Nearest Neighbors
        "KNN": Pipeline([
            ("preprocessor", create_preprocessor(scale_numeric=True)),
            ("classifier", KNeighborsClassifier(
                n_neighbors=5
            ))
        ]),

        # 4. Logistic Regression
        "Logistic Regression": Pipeline([
            ("preprocessor", create_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(
                max_iter=1000
            ))
        ]),

        # 5. Bagging
        "Bagging": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("classifier", BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=50,
                random_state=42
            ))
        ]),

        # 6. Stacking
        "Stacking": Pipeline([
            ("preprocessor", create_preprocessor(scale_numeric=True)),
            ("classifier", StackingClassifier(
                estimators=[
                    (
                        "decision_tree",
                        DecisionTreeClassifier(random_state=42)
                    ),
                    (
                        "random_forest",
                        RandomForestClassifier(
                            n_estimators=50,
                            random_state=42
                        )
                    ),
                    (
                        "knn",
                        KNeighborsClassifier(n_neighbors=5)
                    )
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5
            ))
        ])

    }

    return models
