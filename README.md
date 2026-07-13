# 🍊 Orange Disease Classification

This project focuses on identifying different diseases in oranges using Deep Learning and Computer Vision. The model is trained to classify images into four categories: **Black Spot, Canker, Fresh, and Greening**.

The goal of this project is to automate the process of detecting orange diseases from images, making it easier to identify infected fruits quickly and accurately.

---

## About the Project

Fruit diseases can reduce both the quality and quantity of agricultural produce. Manually inspecting every fruit is time-consuming and can be inconsistent. In this project, I developed a CNN-based image classification model that can recognize different orange diseases from images.

The project includes image preprocessing, model training, evaluation, and prediction using TensorFlow and Keras.

---

## Dataset

The dataset is organized into training and testing folders with four classes:

```
train/
├── blackspot
├── canker
├── fresh
└── grenning

test/
├── blackspot
├── canker
├── fresh
└── grenning
```

Each folder contains images belonging to its respective disease category.

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Project Workflow

- Load the dataset
- Preprocess the images
- Build a CNN model
- Train the model
- Evaluate its performance
- Predict the disease class for new images

---

## Model Evaluation

The model was evaluated using standard classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-Score

These metrics helped measure how well the model performs on unseen test images.

---

## How to Run

1. Clone this repository.

```bash
git clone https://github.com/BHEEMREDDYAVINASHREDDY/Orange-Disease-Classification.git
```

2. Move into the project directory.

```bash
cd Orange-Disease-Classification
```

3. Install the required libraries.

```bash
pip install -r requirements.txt
```

4. Open the notebook.

```bash
jupyter notebook
```

5. Run `orange_classifier.ipynb`.

---

## Project Structure

```
Orange-Disease-Classification/
│
├── train/
├── test/
├── orange_classifier.ipynb
├── orange_report.pdf
├── requirements.txt
└── README.md
```

---

## Future Improvements

Some improvements that can be made in the future include:

- Using Transfer Learning models like MobileNetV2 or EfficientNet
- Increasing the dataset size
- Applying data augmentation
- Deploying the model as a web application using Streamlit or Flask
- Building a mobile application for real-time disease detection

---

## Author

**Bheemreddy Avinash Reddy**

B.Tech in Computer Science Engineering

Interested in Machine Learning, Data Science, and Artificial Intelligence.

GitHub: https://github.com/BHEEMREDDYAVINASHREDDY

---

Thank you for checking out this project. If you have any suggestions or feedback, feel free to open an issue or connect with me.
