# Explainable & Uncertainty-Aware Brain Tumor Classification (Swin + ViT)

1. Python 3.10/3.11 install pannunga, VS Code la intha folder open pannunga.
2. Terminal:  python -m venv venv  ->  venv\Scripts\activate  ->  pip install -r requirements.txt
3. Dataset (Kaggle "Brain Tumor MRI Dataset" - Masoud Nickparvar) download panni:
   dataset/Training/{glioma,meningioma,notumor,pituitary}
   dataset/Testing/{glioma,meningioma,notumor,pituitary}
4. python train.py
5. python evaluate.py
6. python predict.py --image dataset/Testing/glioma/<any>.jpg --shap

Results -> outputs/ (training_curves.png, confusion_matrix.png, test_report.txt, explanation.png)
