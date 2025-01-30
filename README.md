# EV Charging Station Maintenance System Prediction

This project aims to enhance the reliability and efficiency of EV charging stations by predicting their maintenance needs. By leveraging machine learning techniques, the system can forecast potential faults and schedule maintenance proactively, thereby minimizing downtime and ensuring a seamless charging experience for users.

## Project Structure

- `data/`
  - `charging_station_data`: Contains historical data on charging station usage and maintenance records.
- `notebooks/`
  - `model.ipynb`: Jupyter Notebook containing the data analysis, preprocessing, and model training code.
  - `version check.ipynb`: Jupyter Notebook to check the versions of the libraries used in the project.
  - `visualization.ipynb`: Jupyter Notebook to query from data.
- `models/`
  - `model_fault_probability.pkl`: Trained model for predicting faults probability.
  - `model_maintenance_needed.pkl`: Trained model for predicting type of maintenance needed.
  - `model_next_maintenance_days.pkl`: Trained model for predicting next maintenance in days.
  - `scaler.pkl`: Preprocessing pipeline used for data transformation.
- `src/`
  - `predict.py`: Script for making predictions using the trained model.
  - `img/`: Optional Image files for this project.
- `LICENSE`: Contains the licensing information for the project.
- `README.md`: Project documentation.
- `requirements.txt`: List of dependencies required to run the project.
- `streamlit_app.py`: Script to run the Streamlit web application for the project.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/vkn500/ev-charging-maintenance-prediction.git
    cd ev-charging-maintenance-prediction
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the data:
    - Ensure the `charging_station_data.csv` file is placed in the `data/` directory.

2. Run the Jupyter Notebook:
    - Open `notebooks/models.ipynb` and execute the cells to preprocess the data and train the model.

## Deployment

To deploy the prediction model, follow these steps:

1. Save the trained model:
    - Ensure the model is saved to a file after training in the Jupyter Notebook.

2. Create a web service:
    - Use a web framework like Flask, FastAPI, or Streamlit to create an API endpoint for making predictions.

3. Deploy the web service:
    - Deploy the web service to a cloud platform such as AWS, Azure, or Heroku.

4. Make predictions via the API:
    - Send HTTP requests to the API endpoint with the necessary input data to receive predictions.

## Results

The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Detailed results and visualizations can be found in the Jupyter Notebook.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the GNU General Public License v3.0. See the `LICENSE` file for more details.
