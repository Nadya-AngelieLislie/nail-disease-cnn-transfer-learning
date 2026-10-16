# Nail Disease Image Classification

> Baseline CNN, tuned CNN and EfficientNetB0 transfer learning on an imbalanced nail-photo dataset.

## Overview

- Individual Deep Learning final-exam project (UAS, dataset B): classify fingernail photos into 5 classes.
- Compares a baseline CNN, a modified CNN and transfer learning with a pre-trained EfficientNetB0.

## Data / Scope

- 5 classes: Acral Lentiginous Melanoma, Healthy Nail, Onychogryphosis, Blue Finger and Pitting.
- The raw data was badly imbalanced (about 600 images for the largest classes, 343 for healthy nails and only 21 for melanoma), so it was cleaned and augmented to a balanced set.
- Final splits: 2,200 training, 470 validation and 475 test images (224x224).

## Workflow

- Inspected sample images and class counts, then cleaned and augmented the minority classes.
- Baseline CNN with ReLU, trained for 20 epochs.
- Modified CNN (extra layers and batch normalization) trained for 30 epochs.
- EfficientNetB0 transfer learning, then fine-tuning with the top layers unfrozen at a learning rate of 1e-5.
- Compared all models on validation curves and on the test set with accuracy, precision, recall, F1 and confusion matrices.

## Results

- Test accuracy: baseline CNN 81.1%, modified CNN 85.3%, EfficientNet transfer learning 98.95%.
- The baseline struggled most with Pitting (precision 0.60); EfficientNet separated all five classes with near-perfect recall.
- Validation loss dropped steadily with no divergence, so fine-tuning did not overfit.

## Tech Stack

Python, TensorFlow/Keras, EfficientNetB0, scikit-learn, Matplotlib

## Repository Structure

```
notebook/    Full notebook with code, outputs and written interpretation
figures/     Key charts exported from the notebook
requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook/UAS_DL_Nail_Disease_CNN.ipynb
```

The notebook was written in Google Colab; the dataset is not included, so upload it to the Colab session (or edit the path in the first data-loading cell).

## Team

Individual project by Nadya Angelie Lislie (Deep Learning course, BINUS University).

## Notes

- Student ID numbers were removed from the notebook text.
- The nail image dataset (`3B.zip`) is course material and is not included; this is a course exercise, not a medical tool.

## Author

Nadya Angelie Lislie, BINUS University (Computer Science and Statistics). [LinkedIn](https://www.linkedin.com/in/nadya-angelie-lislie/) | [GitHub](https://github.com/Nadya-AngelieLislie)

## Experiment Tracking

`src/experiment_tracking.py` provides an optional MLflow wrapper (`track_run`) for logging parameters, metrics and artifacts; it does nothing when MLflow is not installed. Run `mlflow ui` after logging runs to compare experiments.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff` and `pytest` repository-health checks on every push; run them locally with `make lint test`.
