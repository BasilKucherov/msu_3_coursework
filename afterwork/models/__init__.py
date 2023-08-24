from .ds_cnn import *

available_models = [
    'ds_cnn',
]

def create_model(model_name, num_classes, in_channels):
    if model_name == "ds_cnn":
        model = DSCNN(num_classes=num_classes, in_channels=in_channels)
    return model
