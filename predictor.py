import joblib
import pandas as pd

class FetalHealthPredictor:
    def __init__(self):
        try:
            self.model = joblib.load('models/XGBCLassifier_pipeline.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
        except FileNotFoundError:
            self.model = None
            self.scaler = None
    
    def predict(self, input_data: dict):
        if self.model is None:
            print("https://www.youtube.com/watch?v=dQw4w9WgXcQ") 
        input_df = pd.DataFrame([input_data])
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)[0]
        
        prediction_final = int(prediction) + 1
        
        return {"PREDICTION": prediction_final}
