# AI-Pneumonia-Classifier
In this Project, I Compare 3 models and their accuracy on my chosen dataset.



## 🚀 Live Demo
You can test the best-performing model (EfficientNet-B0) live in your browser! This web application is hosted on Hugging Face Spaces.
➡️ Click here to access the Gradio Web App (این لینک را بعد از ساخت Space در Hugging Face جایگزین کن)

## 📋 Project Overview
In this project, I tackled the important challenge of detecting pneumonia from chest X-ray images using deep learning. The goal was to develop a reliable classification model by exploring and comparing different modern neural network architectures.
The project follows a complete research cycle: from data analysis and addressing class imbalance to training multiple models, performing a comparative analysis, and finally, deploying the best-performing model as an interactive web application.
Key Features:
 * Comparative Analysis: Trained and evaluated three different architectures: ResNet-18 (as a baseline), EfficientNet-B0 (a modern, efficient CNN), and Vision Transformer (ViT).
 * Class Imbalance Handling: Implemented a WeightedLoss function to address the inherent bias in the dataset, significantly improving the model's fairness and performance.
 * Model Checkpointing: The training script automatically saves the model with the best validation accuracy.
 * Interactive Demo: A user-friendly web app built with Gradio for live inference and demonstration of the final model.
   
## 🛠️ Tech Stack
 * Programming Language: Python 3.9
 * Core Libraries: PyTorch, Torchvision
 * Data Handling & Analysis: Pandas, NumPy, Scikit-learn
 * Visualization: Matplotlib, Seaborn
 * Web App: Gradio
   
## 📊 Results & Analysis
My initial experiments with a standard ResNet-18 model revealed a significant bias towards the majority class (Pneumonia), resulting in a low recall of only 35% for the NORMAL class. To address this, I implemented a weighted loss strategy which successfully improved the model's balance.
To further enhance performance, I trained two more advanced architectures: EfficientNet-B0 and a Vision Transformer (ViT-B_16). The final test results are summarized below:

|     Model Architecture     | Test Accuracy | Test Recall (NORMAL) | Test Recall (PNEUMONIA) | F1-Score (Weighted Avg) |
------------------------------------------------------------------------------------------------------------------------.
|    ResNet-18 (Weighted)    |     80.0%     |         0.46         |          1.00           |           0.77          |
------------------------------------------------------------------------------------------------------------------------.
| EfficientNet-B0 (Weighted) |     85.1%     |         0.65         |          0.97           |           0.85          |
------------------------------------------------------------------------------------------------------------------------.
|  Vision Transformer (ViT)  |     70.0%     |         0.26         |          0.96           |           0.65          |
------------------------------------------------------------------------------------------------------------------------

## Conclusion
The EfficientNet-B0 model emerged as the clear winner. It not only achieved the highest overall accuracy but, more importantly, provided the best balance between correctly identifying healthy patients (65% recall) and not missing sick ones (97% recall). The Vision Transformer, likely due to the limited size of the dataset, struggled with training stability and did not generalize as well as the CNN-based models.
🖥️ The Web Application
Here is a screenshot of the final Gradio application in action.
<img width="2929" height="1714" alt="web-app-pic" src="https://github.com/user-attachments/assets/5631d5d4-8269-4f67-86e0-90971f4f475b" />

## ⚙️ Setup & Usage
To replicate this project on your local machine, follow these steps:
1. Clone the repository:
git clone https://github.com/Amir-Shokrzadeh/AI-Pneumonia-Classifier.git
cd AI-Pneumonia-Classifier

2. Create a Conda environment and install dependencies:
# Create and activate the environment
conda create -n pneumonia-env python=3.9
conda activate pneumonia-env

# Install all required libraries
pip install -r requirements.txt

## 3. Download the Dataset:
Download the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle and place the chest_xray folder in the root of the project directory.
### web link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
### 5. Run the Research Notebooks (Optional):
To see the full research and training process, you can explore the Jupyter notebooks located in the /notebooks directory.
### 6. Launch the Web App:
To start the interactive demo, run the following command in your terminal:
python app.py

Then, open your browser and navigate to the local URL provided (e.g., http://127.0.0.1:7860).
🚀 Future Work
 * Experiment with more advanced data augmentation techniques using the Albumentations library.
 * Implement Test-Time Augmentation (TTA) to potentially improve inference accuracy.
 * Explore other modern architectures like ConvNeXt.
