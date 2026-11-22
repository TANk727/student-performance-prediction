
def preprocess_data(data):
    """
    Prepares the student dataset for machine learning.
    Separates input features (X) and target variable (y).
    """

    # Create a copy to protect the original dataset
    data = data.copy()

    # Create Pass/Fail target
    # 1 = Pass, 0 = Fail
    data["Pass"] = (data["G3"] >= 10).astype(int)

    # Separate features and target
    X = data.drop(columns=["G3", "Pass"])
    y = data["Pass"]

    return X, y
