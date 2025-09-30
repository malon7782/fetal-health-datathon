import time

class FetalHealthPredictor:
    def __init__(self, model_path: str = None, scaler_path: str = None):
        time.sleep(1) 
        self.model = "Mock Model" 
        self.inverse_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}

    def predict(self, raw_input: dict):
        print(f" Receive input: {raw_input}")
        
        time.sleep(0.5)

        mock_result = {
            "prediction_code": 2,
            "prediction_label": self.inverse_mapping[2], 
            "confidence": {
                "Normal": 0.25,
                "Suspect": 0.65, 
                "Pathologic": 0.10
            }
        }
        
        if raw_input.get('DP', 0) > 3:
            mock_result = {
                "prediction_code": 3,
                "prediction_label": self.inverse_mapping[3],
                "confidence": {"Normal": 0.1, "Suspect": 0.2, "Pathologic": 0.7}
            }
        
        print(f"{mock_result}")
        return mock_result
