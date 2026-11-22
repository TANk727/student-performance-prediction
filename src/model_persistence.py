
import os
import joblib


def save_models(models, file_path="models/student_models.joblib"):
    """
    Saves all trained models into one Joblib file.
    """

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save models
    joblib.dump(models, file_path)

    print(f"\nModels successfully saved to: {file_path}")


def load_models(file_path="models/student_models.joblib"):
    """
    Loads previously saved models.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Model file not found: {file_path}. Train the models first."
        )

    models = joblib.load(file_path)

    print("Models successfully loaded!")

    return models
