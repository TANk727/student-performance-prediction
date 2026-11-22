
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_training import build_models

from src.evaluation import (
    evaluate_model,
    show_detailed_evaluation,
    cross_validate_model,
    compare_models
)

from src.visualizations import (
    plot_feature_importance,
    plot_model_comparison
)

from src.model_persistence import save_models
from src.tuning import tune_models


# LOAD DATA
data = load_data("data/student_data.csv")

if data is None:
    raise SystemExit("Dataset could not be loaded.")


# PREPROCESS
X, y = preprocess_data(data)

print("Dataset Shape:", X.shape)
print("\nTarget Distribution:")
print(y.value_counts())


# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# CREATE DEFAULT MODELS
models = build_models()


# EVALUATE DEFAULT MODELS
print("\n========== DEFAULT MODEL EVALUATION ==========")

comparison_results = {}

for name, model in models.items():

    print(f"\nEvaluating default {name}...")

    cv_results = cross_validate_model(
        model,
        X_train,
        y_train
    )

    model.fit(X_train, y_train)

    test_results = evaluate_model(
        model,
        X_test,
        y_test
    )

    comparison_results[f"Default - {name}"] = {
        "CV F1-Score": cv_results["CV F1-Score"],
        "Test F1-Score": test_results["F1-Score"]
    }


# HYPERPARAMETER TUNING
print("\n========== HYPERPARAMETER TUNING ==========")

tuned_models, tuning_results = tune_models(
    models,
    X_train,
    y_train
)


# EVALUATE TUNED MODELS
print("\n========== TUNED MODEL EVALUATION ==========")

for name, model in tuned_models.items():

    print(f"\nEvaluating tuned {name}...")

    test_results = evaluate_model(
        model,
        X_test,
        y_test
    )

    comparison_results[f"Tuned - {name}"] = {
        "CV F1-Score": tuning_results[name]["Best CV F1"],
        "Test F1-Score": test_results["F1-Score"]
    }


# COMPARE DEFAULT AND TUNED MODELS
comparison = compare_models(comparison_results)

print("\n========== DEFAULT VS TUNED COMPARISON ==========")
print(comparison.round(3))


# DETAILED EVALUATION OF TUNED MODELS
for name, model in tuned_models.items():

    print(f"\n========== TUNED {name} ==========")

    show_detailed_evaluation(
        model,
        X_test,
        y_test
    )


# FEATURE IMPORTANCE
plot_feature_importance(
    tuned_models["Random Forest"],
    "Random Forest"
)

plot_feature_importance(
    tuned_models["Logistic Regression"],
    "Logistic Regression"
)


# VISUALIZE COMPARISON
plot_model_comparison(comparison)


# SAVE TUNED MODELS
save_models(
    tuned_models,
    "models/tuned_student_models.joblib"
)


print("\nAll training and evaluation tasks completed successfully!")
