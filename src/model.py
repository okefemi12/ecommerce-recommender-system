import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

from src.config import MODEL_PATH, HISTORY_LENGTH

class ModelService:
    interpreter = None
    input_details = None
    output_details = None
    
    @classmethod
    def load(cls):
        """Load the model when app starts"""
        try:
            cls.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            cls.interpreter.allocate_tensors()
            cls.input_details = cls.interpreter.get_input_details()
            cls.output_details = cls.interpreter.get_output_details()
            print("Model loaded!")
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    @classmethod
    def predict(cls, history):
        """Make a prediction"""
        # Fix history length
        if len(history) > HISTORY_LENGTH:
            history = history[-HISTORY_LENGTH:]
        elif len(history) < HISTORY_LENGTH:
            history = [0] * (HISTORY_LENGTH - len(history)) + history
        
        # Prepare input
        input_data = np.array(history, dtype=np.int32).reshape(1, HISTORY_LENGTH)
        
        # Run model
        input_index = cls.input_details[0]['index']
        cls.interpreter.set_tensor(input_index, input_data)
        cls.interpreter.invoke()
        
        # Get result
        output_index = cls.output_details[0]['index']
        q_values = cls.interpreter.get_tensor(output_index)[0]
        best_action = int(np.argmax(q_values))
        
        return best_action + 1, q_values.tolist()