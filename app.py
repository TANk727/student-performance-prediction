
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from src.model_persistence import load_models
from src.preprocessing import preprocess_data


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# LOAD DATA AND MODELS
# ==========================================

@st.cache_data
def load_dataset():
    return pd.read_csv("data/student_data.csv", sep=";")


@st.cache_resource
def load_trained_models():
    return load_models("models/tuned_student_models.joblib")


data = load_dataset()
models = load_trained_models()


# ==========================================
# FRIENDLY FIELD NAMES
# ==========================================

field_labels = {
    "school": "School",
    "sex": "Gender",
    "age": "Student Age",
    "address": "Home Location",
    "famsize": "Family Size",
    "Pstatus": "Parents' Living Status",
    "Medu": "Mother's Education Level",
    "Fedu": "Father's Education Level",
    "Mjob": "Mother's Occupation",
    "Fjob": "Father's Occupation",
    "reason": "Reason for Choosing School",
    "guardian": "Primary Guardian",
    "traveltime": "Travel Time to School",
    "studytime": "Weekly Study Time",
    "failures": "Past Class Failures",
    "schoolsup": "Extra School Support",
    "famsup": "Family Educational Support",
    "paid": "Extra Paid Classes",
    "activities": "Extracurricular Activities",
    "nursery": "Attended Nursery School",
    "higher": "Wants Higher Education",
    "internet": "Internet Access at Home",
    "romantic": "In a Romantic Relationship",
    "famrel": "Family Relationship Quality",
    "freetime": "Free Time After School",
    "goout": "Going Out with Friends",
    "Dalc": "Workday Alcohol Consumption",
    "Walc": "Weekend Alcohol Consumption",
    "health": "Current Health Status",
    "absences": "School Absences",
    "G1": "First Period Grade",
    "G2": "Second Period Grade"
}


# ==========================================
# FRIENDLY VALUE LABELS
# ==========================================

value_labels = {
    "school": {
        "GP": "Gabriel Pereira School",
        "MS": "Mousinho da Silveira School"
    },
    "sex": {"F": "Female", "M": "Male"},
    "address": {"U": "Urban Area", "R": "Rural Area"},
    "famsize": {
        "LE3": "3 or Fewer Family Members",
        "GT3": "More Than 3 Family Members"
    },
    "Pstatus": {
        "T": "Parents Living Together",
        "A": "Parents Living Apart"
    },
    "Medu": {
        0: "No Formal Education",
        1: "Primary Education",
        2: "5th to 9th Grade",
        3: "Secondary Education",
        4: "Higher Education"
    },
    "Fedu": {
        0: "No Formal Education",
        1: "Primary Education",
        2: "5th to 9th Grade",
        3: "Secondary Education",
        4: "Higher Education"
    },
    "Mjob": {
        "teacher": "Teacher",
        "health": "Healthcare",
        "services": "Services",
        "at_home": "Works at Home",
        "other": "Other"
    },
    "Fjob": {
        "teacher": "Teacher",
        "health": "Healthcare",
        "services": "Services",
        "at_home": "Works at Home",
        "other": "Other"
    },
    "reason": {
        "home": "Close to Home",
        "reputation": "School Reputation",
        "course": "Preferred Course",
        "other": "Other Reason"
    },
    "guardian": {
        "mother": "Mother",
        "father": "Father",
        "other": "Other"
    },
    "traveltime": {
        1: "Less Than 15 Minutes",
        2: "15 to 30 Minutes",
        3: "30 Minutes to 1 Hour",
        4: "More Than 1 Hour"
    },
    "studytime": {
        1: "Less Than 2 Hours",
        2: "2 to 5 Hours",
        3: "5 to 10 Hours",
        4: "More Than 10 Hours"
    },
    "failures": {
        0: "No Previous Failures",
        1: "1 Previous Failure",
        2: "2 Previous Failures",
        3: "3 or More Previous Failures"
    },
    "schoolsup": {"yes": "Yes", "no": "No"},
    "famsup": {"yes": "Yes", "no": "No"},
    "paid": {"yes": "Yes", "no": "No"},
    "activities": {"yes": "Yes", "no": "No"},
    "nursery": {"yes": "Yes", "no": "No"},
    "higher": {"yes": "Yes", "no": "No"},
    "internet": {"yes": "Yes", "no": "No"},
    "romantic": {"yes": "Yes", "no": "No"},
    "famrel": {
        1: "Very Poor",
        2: "Poor",
        3: "Average",
        4: "Good",
        5: "Excellent"
    },
    "freetime": {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    },
    "goout": {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    },
    "Dalc": {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    },
    "Walc": {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    },
    "health": {
        1: "Very Poor",
        2: "Poor",
        3: "Average",
        4: "Good",
        5: "Very Good"
    }
}


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🎓 Student Analytics")

st.sidebar.caption(
    "Student Performance Prediction System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Exploratory Data Analysis",
        "Predict Performance",
        "Model Performance",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Project | Python & Scikit-learn"
)


# ==========================================
# OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.title("Student Performance Dashboard")

    st.write(
        "Explore student academic information, "
        "understand dataset patterns and predict "
        "student performance using machine learning."
    )

    st.divider()

    total_students = len(data)
    average_grade = data["G3"].mean()
    passing_students = (data["G3"] >= 10).sum()
    failing_students = (data["G3"] < 10).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", total_students)
    col2.metric("Average Final Grade", f"{average_grade:.2f}/20")
    col3.metric("Passing Students", passing_students)
    col4.metric("Failing Students", failing_students)

    st.divider()

    st.subheader("Pass vs Fail Distribution")

    pass_counts = pd.Series({
        "Pass": passing_students,
        "Fail": failing_students
    })

    col1, col2 = st.columns([2, 1])

    with col1:

        fig, ax = plt.subplots(figsize=(7, 4))

        pass_counts.plot(kind="bar", ax=ax)

        ax.set_ylabel("Number of Students")
        ax.set_xlabel("Final Result")
        ax.set_title("Student Pass/Fail Distribution")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        st.markdown("### Dataset Information")

        st.write("**Dataset:** Student Performance (Math)")
        st.write(f"**Students:** {total_students}")
        st.write(f"**Features:** {data.drop(columns=['G3']).shape[1]}")
        st.write("**Target:** Pass / Fail")
        st.write("**Passing Threshold:** Final grade ≥ 10")

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )


# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================

elif page == "Exploratory Data Analysis":

    st.title("Exploratory Data Analysis")

    st.write(
        "Explore relationships and patterns "
        "within the student performance dataset."
    )

    st.divider()

    st.subheader("Final Grade Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(data["G3"], bins=20, edgecolor="black")

    ax.set_xlabel("Final Grade")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Final Grades")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Study Time vs Final Grade")

    study_grade = data.groupby("studytime")["G3"].mean()

    study_labels = {
        1: "Less than 2 hours",
        2: "2 to 5 hours",
        3: "5 to 10 hours",
        4: "More than 10 hours"
    }

    study_grade.index = [
        study_labels.get(x, str(x))
        for x in study_grade.index
    ]

    fig, ax = plt.subplots(figsize=(9, 4))

    study_grade.plot(kind="bar", ax=ax)

    ax.set_xlabel("Weekly Study Time")
    ax.set_ylabel("Average Final Grade")
    ax.set_title("Study Time vs Average Final Grade")
    ax.tick_params(axis="x", rotation=15)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("School Absences vs Final Grade")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(
        data["absences"],
        data["G3"],
        alpha=0.6
    )

    ax.set_xlabel("Number of Absences")
    ax.set_ylabel("Final Grade")
    ax.set_title("Absences vs Final Grade")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Numerical Feature Correlation")

    numeric_data = data.select_dtypes(include="number")
    correlation = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(12, 8))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(correlation.columns, rotation=90)
    ax.set_yticklabels(correlation.columns)

    fig.colorbar(image, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Explore Individual Features")

    selected_feature = st.selectbox(
        "Select a feature",
        data.columns
    )

    display_name = field_labels.get(
        selected_feature,
        selected_feature
    )

    st.write(f"Distribution of {display_name}")

    if data[selected_feature].dtype == "object":

        st.bar_chart(data[selected_feature].value_counts())

    else:

        st.bar_chart(
            data[selected_feature].value_counts().sort_index()
        )


# ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "Predict Performance":

    st.title("Student Performance Prediction")

    st.write(
        "Enter student information to predict "
        "whether the student is likely to pass or fail."
    )

    st.info(
        "The prediction uses previous academic grades "
        "and other student information."
    )

    st.divider()

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        list(models.keys())
    )

    selected_model = models[selected_model_name]

    feature_data = data.drop(columns=["G3"])

    categorical_columns = feature_data.select_dtypes(
        include=["object"]
    ).columns

    user_input = {}

    with st.form("student_prediction_form"):

        st.subheader("Student Information")

        columns = st.columns(2)

        for index, feature in enumerate(feature_data.columns):

            label = field_labels.get(
                feature,
                feature.replace("_", " ").title()
            )

            with columns[index % 2]:

                if feature in categorical_columns:

                    options = sorted(
                        feature_data[feature].dropna().unique().tolist()
                    )

                    labels = value_labels.get(feature, {})

                    user_input[feature] = st.selectbox(
                        label,
                        options,
                        format_func=lambda value, labels=labels:
                            labels.get(value, str(value))
                    )

                else:

                    minimum = float(feature_data[feature].min())
                    maximum = float(feature_data[feature].max())
                    default = float(feature_data[feature].median())

                    user_input[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1.0
                    )

        submitted = st.form_submit_button(
            "Predict Student Performance",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_input],
            columns=feature_data.columns
        )

        prediction = selected_model.predict(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("The model predicts: PASS")
        else:
            st.error("The model predicts: FAIL")

        if hasattr(selected_model, "predict_proba"):

            probabilities = selected_model.predict_proba(input_df)[0]

            fail_probability = probabilities[0] * 100
            pass_probability = probabilities[1] * 100

            col1, col2 = st.columns(2)

            col1.metric(
                "Fail Probability",
                f"{fail_probability:.2f}%"
            )

            col2.metric(
                "Pass Probability",
                f"{pass_probability:.2f}%"
            )

            st.subheader("Prediction Probability")

            st.progress(
                float(probabilities[1]),
                text="Pass Probability"
            )

            st.progress(
                float(probabilities[0]),
                text="Fail Probability"
            )

        st.caption(
            "Predictions are estimates from the selected model, "
            "not guarantees of actual academic outcomes."
        )


# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.title("Model Performance Analysis")

    st.write(
        "Compare the performance of all six tuned models "
        "using an independent test dataset."
    )

    st.info(
        "These results use the same 80/20 stratified split "
        "used during model training. No models are retrained here."
    )

    # Recreate the original test split
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Evaluate saved models
    performance_results = {}

    predictions = {}

    for name, model in models.items():

        prediction = model.predict(X_test)

        predictions[name] = prediction

        performance_results[name] = {
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(
                y_test, prediction,
                average="macro",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test, prediction,
                average="macro",
                zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, prediction,
                average="macro",
                zero_division=0
            ),
            "Fail Recall": recall_score(
                y_test, prediction,
                pos_label=0,
                zero_division=0
            )
        }

    performance_df = pd.DataFrame(
        performance_results
    ).T

    st.divider()

    # Best model
    best_model_name = performance_df["F1-Score"].idxmax()

    best_f1 = performance_df.loc[
        best_model_name, "F1-Score"
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric("Models Evaluated", len(models))
    col2.metric("Best Model (Macro F1)", best_model_name)
    col3.metric("Best Test F1", f"{best_f1:.3f}")

    st.divider()

    # Comparison table
    st.subheader("Model Comparison")

    st.dataframe(
        performance_df.style.format("{:.3f}").highlight_max(
            axis=0
        ),
        use_container_width=True
    )

    st.caption(
        "Fail Recall measures the proportion of actual failing "
        "students correctly identified by the model."
    )

    st.divider()

    # Performance chart
    st.subheader("Performance Metrics Comparison")

    selected_metrics = st.multiselect(
        "Choose metrics to compare",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Fail Recall"
        ],
        default=["Accuracy", "F1-Score", "Fail Recall"]
    )

    if selected_metrics:

        st.bar_chart(
            performance_df[selected_metrics],
            use_container_width=True
        )

    st.divider()

    # Individual confusion matrix
    st.subheader("Confusion Matrix")

    selected_model_name = st.selectbox(
        "Choose a model to inspect",
        list(models.keys()),
        key="performance_model"
    )

    selected_predictions = predictions[selected_model_name]

    matrix = confusion_matrix(
        y_test,
        selected_predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Fail", "Pass"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(f"{selected_model_name} - Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
        **How to read the confusion matrix:**

        - Top-left: Actual Fail correctly predicted as Fail.
        - Top-right: Actual Fail incorrectly predicted as Pass.
        - Bottom-left: Actual Pass incorrectly predicted as Fail.
        - Bottom-right: Actual Pass correctly predicted as Pass.
        """
    )

    st.divider()

    # Classification report
    st.subheader("Detailed Classification Report")

    report = classification_report(
        y_test,
        selected_predictions,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.style.format("{:.3f}"),
        use_container_width=True
    )


# ==========================================
# MODEL INFORMATION PAGE
# ==========================================

elif page == "Model Information":

    st.title("Machine Learning Models")

    st.write(
        "This project uses six supervised machine learning "
        "algorithms to predict student pass/fail outcomes."
    )

    model_descriptions = {
        "Decision Tree": (
            "Uses decision rules based on feature values "
            "to classify student outcomes."
        ),
        "Random Forest": (
            "Combines multiple decision trees to improve "
            "prediction stability and generalization."
        ),
        "KNN": (
            "Predicts outcomes by comparing students with "
            "similar feature values."
        ),
        "Logistic Regression": (
            "Estimates the probability of a student passing "
            "or failing using a classification model."
        ),
        "Bagging": (
            "Trains multiple models on bootstrap samples "
            "and combines their predictions."
        ),
        "Stacking": (
            "Combines predictions from different base models "
            "using a final meta-model."
        )
    }

    for name, description in model_descriptions.items():

        with st.expander(name):
            st.write(description)

    st.divider()

    st.subheader("Training Information")

    st.write("**Dataset:** UCI Student Performance Dataset")
    st.write("**Target:** Pass / Fail")
    st.write("**Passing Condition:** G3 ≥ 10")
    st.write("**Evaluation:** Stratified 5-Fold Cross-Validation")
    st.write("**Hyperparameter Tuning:** GridSearchCV")
    st.write("**Tuning Metric:** Macro F1-Score")
    st.write("**Model Storage:** Joblib")
    st.write("**Application Framework:** Streamlit")


# ==========================================
# FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "Student Performance Prediction System"
)

st.sidebar.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)
