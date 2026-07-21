import os
import pickle
from typing import Tuple

# Global singletons to cache loaded models
_vectorizer = None
_model = None

def get_local_ml_model():
    global _vectorizer, _model
    if _vectorizer is not None and _model is not None:
        return _vectorizer, _model

    # Resolves models directory path relative to this file's directory (backend/core)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    model_path = os.path.join(models_dir, "scam_xgb_model.pkl")

    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        try:
            with open(vectorizer_path, "rb") as f:
                _vectorizer = pickle.load(f)
            with open(model_path, "rb") as f:
                _model = pickle.load(f)
        except Exception as e:
            print(f"Error loading local ML model: {e}")
            
    return _vectorizer, _model

def predict_local_scam_probability(text: str) -> float:
    """
    Returns the threat probability (0.0 to 100.0) of a message being a scam
    based on the locally trained XGBoost + TF-IDF model.
    """
    vectorizer, model = get_local_ml_model()
    if vectorizer is None or model is None:
        return 0.0
        
    try:
        vec = vectorizer.transform([text])
        # predict_proba returns [prob_class_0, prob_class_1]
        probs = model.predict_proba(vec)[0]
        # Class 1 is fraud/scam
        return float(probs[1]) * 100.0
    except Exception as e:
        print(f"Error predicting probability: {e}")
        return 0.0
