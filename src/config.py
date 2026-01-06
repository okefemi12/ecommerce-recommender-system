import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model_data", "hybrid_dqn_model.tflite")

HISTORY_LENGTH = 20