# =============================================================================
#               GRADIO WEB APP FOR PNEUMONIA CLASSIFICATION
# =============================================================================

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Config 
MODEL_PATH = './saved_models/best_model_pneumonia.pth' 
DEVICE = torch.device("cpu")
class_names = ['NORMAL', 'PNEUMONIA']

# LOAD THE CHAMPION MODEL (EFFICIENTNET-B0) 
model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

# Load the saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval() # Set model to evaluation mode

print("EfficientNet-B0 model loaded successfully.")

# DEFINE THE PREDICTION FUNCTION
def predict(input_image: Image.Image):
    """
    Takes a user-uploaded PIL image, preprocesses it, and returns the model's prediction probabilities.
    """
    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image and add a batch dimension
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    # Run prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Format the output into a dictionary of class names and confidences
    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return confidences

# CREATE AND LAUNCH THE GRADIO INTERFACE
print("Launching Gradio interface...")

# Define some example images from your dataset for users to try
example_images = [
    'chest_xray/test/NORMAL/IM-0011-0001-0002.jpeg',
    'chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg'
]

iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction Results"),
    title="🤖 AI-Powered Pneumonia Detector",
    description="This application uses an EfficientNet-B0 model to classify chest X-ray images as NORMAL or PNEUMONIA. Upload an image or use one of the examples below to see it in action.",
    examples=example_images,
    theme=gr.themes.Soft() # A nice, modern theme
)

# Launch the web application
if __name__ == "__main__":
    iface.launch()
