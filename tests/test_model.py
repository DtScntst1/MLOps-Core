import os
import pytest
from src.train import train_model

def test_model_training_outputs():
    """Test if the training script successfully creates model and report files."""
    # Ensure directories are clean or run training
    train_model()
    
    assert os.path.exists("models/model.pkl"), "Model file was not created!"
    assert os.path.exists("reports/metrics.md"), "Metrics report was not created!"
    assert os.path.exists("reports/confusion_matrix.png"), "Confusion matrix plot was not created!"
