
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on testing data.
    """

    predictions = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1-Score": f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
    }

    return results


def show_detailed_evaluation(model, X_test, y_test):
    """
    Displays confusion matrix and classification report.
    """

    predictions = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))


def cross_validate_model(model, X_train, y_train):
    """
    Performs 5-fold stratified cross-validation.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=[
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro"
        ]
    )

    return {
        "CV Accuracy": scores["test_accuracy"].mean(),
        "CV Precision": scores["test_precision_macro"].mean(),
        "CV Recall": scores["test_recall_macro"].mean(),
        "CV F1-Score": scores["test_f1_macro"].mean()
    }


def compare_models(results):
    """
    Converts model results into a comparison table.
    """

    return pd.DataFrame(results).T
