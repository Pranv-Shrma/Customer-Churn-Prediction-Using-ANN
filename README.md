# Customer Churn Prediction Using Artificial Neural Networks

## Overview

This project predicts customer churn for a banking institution using a neural network model. The model is trained using customer data, including credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, activity status, and estimated salary. A Streamlit application is provided for interactive predictions.

## Project Structure

The project consists of the following files:

*   `app.py`: The main Streamlit application file.
*   `experiment.ipynb`: A Jupyter Notebook containing the model training and preprocessing steps.
*   `label_encoder_gender.pkl`: A pickled LabelEncoder for the 'Gender' feature.
*   `onehot_encoder_geo.pkl`: A pickled OneHotEncoder for the 'Geography' feature.
*   `scaler.pkl`: A pickled StandardScaler for scaling numerical features.
*   `model.h5`: A trained Keras model for churn prediction.
*   `prediction.ipynb`:A Jupyter Notebook used for prediction

## Data

The dataset used for training the model is `Churn_Modelling.csv`. The data includes customer information and a binary `Exited` column indicating churn status (1 for churned, 0 for not churned).

## Dependencies

The project requires the following Python libraries:

*   streamlit
*   numpy
*   tensorflow
*   scikit-learn
*   pandas
*   pickle

To install these dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install dependencies:**

    ```bash
    pip install streamlit numpy tensorflow scikit-learn pandas
    ```

## Model Training

The model training and preprocessing steps are detailed in the `experiment.ipynb` Jupyter Notebook. Key steps include:

1.  **Data Loading:** The dataset is loaded using pandas.
2.  **Preprocessing:**
    *   Irrelevant columns (`RowNumber`, `CustomerId`, `Surname`) are dropped.
    *   Categorical features are encoded:
        *   `Gender` is encoded using LabelEncoder.
        *   `Geography` is encoded using OneHotEncoder.
    *   The data is split into independent (X) and dependent (Y) features.
    *   The data is split into training and testing sets.
    *   Numerical features are scaled using StandardScaler.
3.  **Model Building:**
    *   A sequential neural network model is built using TensorFlow/Keras.
    *   The model consists of three dense layers with ReLU and sigmoid activation functions.
    *   The model is compiled with the Adam optimizer and binary cross-entropy loss.
4.  **Training and Validation:**
    *   The model is trained on the training data and validated on the testing data.
    *   Early stopping is used to prevent overfitting.
    *   TensorBoard is used for training visualization.
5.  **Model Saving:**
    *   The trained model, encoders, and scaler are saved as `.h5` and `.pkl` files.

## Usage

### Streamlit Application

1.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

2.  **Interact with the app:**

    *   The app will open in your web browser.
    *   Enter customer information in the provided input fields.
    *   The app will display the churn probability and a prediction.

### Prediction using `prediction.ipynb`

1.  **Open the `prediction.ipynb` Jupyter Notebook.**
2.  **Run the notebook cells:** The notebook loads the trained model, scaler, and encoders. It then uses example input data to make a churn prediction. You can modify the `input_data` dictionary to test with different customer profiles.

## Configuration

The Streamlit application uses the trained model (`model.h5`) and preprocessing objects (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl`). Ensure these files are in the same directory as `app.py`.

## Final Use Case

A bank wants to proactively identify customers who are likely to churn. By using this project, the bank can:

1.  **Input Customer Data:** Bank employees can input customer details into the Streamlit application.
2.  **Predict Churn Probability:** The application provides a churn probability score, indicating the likelihood of a customer churning.
3.  **Take Proactive Measures:** Based on the churn probability, the bank can take proactive measures, such as offering personalized services or discounts, to retain high-risk customers.
4.  **Improve Retention Rates:** By identifying and addressing potential churn risks early on, the bank can improve customer retention rates and reduce financial losses associated with customer attrition.
