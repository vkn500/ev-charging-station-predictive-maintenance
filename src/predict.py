import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the models and preprocessor
maintenance_model = joblib.load("../models/maintenance_model.pkl")
fault_model = joblib.load("../models/fault_model.pkl")
efficiency_model = joblib.load("../models/efficiency_model.pkl")
preprocessor = joblib.load("../models/preprocessor.pkl")

# Initialize LabelEncoder
le = LabelEncoder()
le.classes_ = ['Software Update', 'Connector Cleaning', 'Cable Inspection', 'Battery Check']

# Load new data
new_data = pd.read_csv("path_to_new_data.csv")

# Preprocess new data
X_new = preprocessor.transform(new_data)

# Make predictions
maintenance_predictions = maintenance_model.predict(X_new)
fault_predictions = fault_model.predict(X_new)
efficiency_predictions = efficiency_model.predict(X_new)

# Encode maintenance_needed
maintenance_predictions_cat = le.inverse_transform(maintenance_predictions)

# Display predictions
print("Maintenance Predictions:", maintenance_predictions_cat)
print("Fault Predictions:", fault_predictions)
print("Efficiency Predictions:", efficiency_predictions)
