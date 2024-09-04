from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pickle


app = Flask(__name__)
CORS(app)

# Global variables to store dataset and model
dataset = None
model = None

@app.route('/upload', methods=['POST'])
def upload_file():
    global dataset
    file = request.files['file']
    if file:
        try:
            dataset = pd.read_csv(file)
            columns = dataset.columns.tolist()
            return jsonify({"columns": columns})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    return jsonify({"error": "No file uploaded"}), 400

@app.route('/set_target', methods=['POST'])
def set_target():
    global dataset, model
    data = request.json
    target_column = data.get('target')

    if dataset is not None and target_column in dataset.columns:
        return jsonify({"message": "Target column set successfully"})
    return jsonify({"error": "Invalid target column"}), 400
@app.route('/train_model', methods=['POST'])
def train_model():
    global dataset, model
    data = request.json
    target_column = data['target']
    
    # Check if dataset is loaded and target column exists
    if dataset is not None and target_column in dataset.columns:
        print(f"Dataset columns: {dataset.columns}")

        # Identify categorical and numerical columns
        categorical_columns = [col for col in dataset.columns if dataset[col].nunique() < 10 and col != target_column]
        numerical_columns = [col for col in dataset.columns if dataset[col].dtype in ['int64', 'float64'] and col != target_column]
        
        # Preprocessing for categorical data: one-hot encoding
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        # Preprocessing for numerical data: scaling
        numerical_transformer = StandardScaler()
        
        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ])
        
        # Define multiple classifiers to test
        classifiers = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'KNN': KNeighborsClassifier(),
            'LogisticRegression': LogisticRegression(),
        }

        best_accuracy = 0
        best_model = None
        best_model_name = ""

        # Split the dataset into features and target
        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Iterate through each classifier
        for model_name, clf in classifiers.items():
            # Create a pipeline that first preprocesses the data and then applies the classifier
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])
            
            # Train the model using the pipeline
            pipeline.fit(X_train, y_train)
            
            # Make predictions and calculate accuracy
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{model_name} accuracy: {accuracy}")
            
            # Store the best model based on accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pipeline
                best_model_name = model_name

        if best_model:
            # Store the best model
            model = best_model

            # Save the best model as a .pkl file
            pkl_filename = f"best_model_{best_model_name}_{best_accuracy:.2f}.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(best_model, file)
            
            return jsonify({"best_model": best_model_name, "accuracy": best_accuracy, "model_file": pkl_filename})
        else:
            return jsonify({"error": "No model could be trained."}), 500

    
    return jsonify({"error": "Invalid target column or dataset not loaded"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    global model
    data = request.json
    if model:
        df = pd.DataFrame(data['features'])
        predictions = model.predict(df)
        return jsonify({"predictions": predictions.tolist()})
    return jsonify({"error": "Model not trained"}), 400

if __name__ == '__main__':
    app.run(debug=True)
