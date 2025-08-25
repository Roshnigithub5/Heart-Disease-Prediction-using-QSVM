# Heart Disease Prediction using QSVM ❤️⚛️

### 📌 Overview

This project predicts the likelihood of heart disease using Quantum Support Vector Machine (QSVM) and compares its performance with a classical Support Vector Machine (SVM).
The implementation leverages Qiskit Machine Learning for quantum kernels and scikit-learn for classical ML techniques.


---

### 🎯 Objective

To build a predictive model for heart disease detection using QSVM and evaluate its accuracy against a classical SVM baseline.


---

### 🛠️ Tech Stack

Language: Python

Quantum Framework: Qiskit Machine Learning

Machine Learning Library: scikit-learn

Data Handling: pandas, NumPy

Visualization: matplotlib, seaborn

Dataset: Heart Disease Dataset



---

### ⚙️ Implementation

Data Preprocessing: StandardScaler applied; PCA used to reduce 13 features → 4 components

Classical Model: Support Vector Machine (SVM) with RBF kernel

Quantum Model: Quantum Support Vector Machine (QSVM) using:

ZZFeatureMap for feature embedding

FidelityStatevectorKernel for quantum similarity computation


Train/Test Split: 70% training, 30% testing



---

### 📊 Results

Model	Accuracy

Classical SVM	~97%
QSVM	~95%


Visual Outputs

Confusion Matrices (SVM & QSVM)

Quantum Kernel Gram Heatmap

Accuracy Comparison Bar Chart



---

### 🚀 Future Scope

Deploy QSVM on real quantum hardware instead of a simulator

Experiment with advanced feature maps and hyperparameter tuning

Test performance on larger and more complex healthcare datasets
