# Explainable and Uncertainty-Aware Brain Tumor Classification

## Project Overview
This project focuses on brain tumor classification using MRI images. It uses deep learning techniques and explainable AI methods to provide visual explanations and uncertainty estimation for model predictions.

## Technologies Used
- Python
- PyTorch
- Swin Transformer
- Vision Transformer (ViT)
- Grad-CAM
- SHAP
- Monte Carlo Dropout

## Tumor Classes
- Glioma
- Meningioma
- No Tumor
- Pituitary

## Project Execution
1. Install Python 3.10 or 3.11.
2. Install the required packages:
   `pip install -r requirements.txt`
3. Download the Brain Tumor MRI Dataset from Kaggle.
4. Place the dataset inside the `dataset` folder.
5. Run the training file:
   `python train.py`
6. Evaluate the model:
   `python evaluate.py`
7. Predict an MRI image:
   `python predict.py --image "image_path" --shap`

## Results
The output folder contains:
- Training curves
- Confusion matrix
- Test report
- Grad-CAM and SHAP explanations

## Disclaimer
This project is developed for academic and research purposes only. It is not intended for clinical diagnosis.
