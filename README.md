
##  CTG Fetal Health Status Prediction Model

This project aims to develop a machine learning solution which automatically predicts **fetal health status (Normal, Suspect, Pathologic)** based on **Cardiotocography (CTG)** data using `RandomForestClassifier`.

The final product is an interactive **web app** that allows users to get **real-time predictions** by inputting key metrics.

### File Structure

`main.py`:  **Main script**. Generates the final model files in the `models/` directory.

`src/data.py`: **Data processing module**. Responsible for loading data and cleaning data from the source file.

`src/model.py`: **Model training module**. Contains the `model_training` function. Trains and evaluates the models.

`src/predictor.py`: **Predictor module**. Defines the FetalHealthPredictor class & loads the saved model & provides  `.predict()` interface.

`app.py`: **Web Application**. The final user interface.

### How to use

Clone the Repository
```
git clone <url>
cd fetal-health-datathon
```
Create and Activate a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```
Install Dependencies

```
pip install -r requirements.txt
```

RUN
```
python3 main.py
streamlit run app.py
```
### Predictor Interface

Create an instance of the predictor.
```
predictor = FetalHealthPredictor()
```
Prepare a dictionary with all required features as input
```
new_data = {
    'LB': 120,
    'AC': 2,
    'FM': 0,
    'UC': 4,
    'DS': 0,
    'DP': 0,
    'ASTV': 50,
    'MSTV': 1.5,
    'ALTV': 20,
    'Nzeros': 0
}
```
Call the `.predict()` method to get the result
```
result = predictor.predict(new_data)
print(result)
```

[SUPPORT US!!](https://www.youtube.com/watch?v=dQw4w9WgXcQ "SUPPORT US!!")
