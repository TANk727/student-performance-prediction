
import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, model_name, top_n=15):
    """
    Visualizes important features for supported models.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
        title = f"{model_name} - Feature Importance"

    elif hasattr(classifier, "coef_"):
        importance = classifier.coef_.flatten()
        title = f"{model_name} - Feature Coefficients"

    else:
        print(f"Feature importance unavailable for {model_name}")
        return

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df["Absolute"] = importance_df["Importance"].abs()

    importance_df = importance_df.nlargest(
        top_n, "Absolute"
    )

    importance_df = importance_df.sort_values("Absolute")

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.title(title)
    plt.xlabel("Importance / Coefficient")
    plt.ylabel("Features")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)

    filename = model_name.lower().replace(" ", "_")

    plt.savefig(
        f"outputs/{filename}_feature_importance.png"
    )

    plt.show()
    plt.close()


def plot_model_comparison(comparison):
    """
    Compares default and tuned model performance.
    """

    comparison[
        ["CV F1-Score", "Test F1-Score"]
    ].plot(
        kind="bar",
        figsize=(13, 7)
    )

    plt.title("Default vs Tuned Model Comparison")
    plt.xlabel("Models")
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    plt.legend([
        "Cross-Validation F1",
        "Test F1"
    ])

    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)

    plt.savefig(
        "outputs/default_vs_tuned_comparison.png"
    )

    plt.show()
    plt.close()
