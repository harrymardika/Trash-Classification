# Trash Classification: Recyclable vs. Other Waste (EfficientNetV2M)

An image classifier that sorts waste photos into **recyclable** and **other** waste, built with transfer learning on EfficientNetV2M in TensorFlow/Keras and exported to TensorFlow Lite for mobile and edge devices.

## Results

| Metric | Value |
|---|---|
| Training accuracy | 97.06% |
| Validation accuracy | 93.00% |
| Training loss / validation loss | 0.0758 / 0.3641 |
| Epochs | 34 (stopped once both accuracies reached 93%) |

![Training and validation accuracy](https://github.com/harrymardika/Trash-Classification/assets/130530985/701a6c3c-1721-4a2c-9347-88366cff5831)
![Training and validation loss](https://github.com/harrymardika/Trash-Classification/assets/130530985/b839c9f6-1cf2-4b1a-80ff-648408e6f8f7)
![Prediction example](https://github.com/harrymardika/Trash-Classification/assets/130530985/1ae45972-b09a-43c1-9673-309db8a444a2)

## Dataset

20,228 images in two classes: `other` (10,229) and `recycle` (9,999). To shorten training, 6,000 images per class were used (12,000 total), split 80/20 into 9,600 training and 2,400 validation images.

## Approach

1. **Preprocessing:** resize every image to 200x200 with OpenCV (`INTER_AREA`), copying files in parallel.
2. **Augmentation (training):** rescaling, rotation (20°), horizontal flip, width/height shifts, zoom, and shear (0.2).
3. **Model:** ImageNet-pretrained EfficientNetV2M as feature extractor, then `Dropout(0.2) → Conv2D(64, 3x3) → MaxPooling → Dropout → Flatten → Dropout → Dense(32) → Dropout → Dense(2, softmax)`.
4. **Training:** Adam (lr 0.0001), categorical cross-entropy, batch 50, up to 100 epochs; early stopping on validation accuracy (patience 10), best-model checkpoint, and a callback that stops at 93% train and validation accuracy.
5. **Deployment:** the best model (`best_model.h5`) is converted to TensorFlow Lite (`best_model.tflite`).

## Tech Stack

Python, Google Colab, TensorFlow/Keras (EfficientNetV2M, TFLite converter), OpenCV, NumPy, Matplotlib.

## Project Structure

```
Trash-Classification/
├── model_trash_classification.ipynb   # Notebook with outputs
└── model_trash_classification.py      # Script exported from Colab
```

The dataset and trained model files are not stored in the repository.

## Getting Started

```bash
git clone https://github.com/harrymardika/Trash-Classification.git
cd Trash-Classification
pip install tensorflow opencv-python numpy matplotlib jupyter
```

Put images in `other/` and `recycle/` folders. The notebook was written for Colab and reads data from `/content/drive/MyDrive/assets/trash-data`; to run locally, remove the `drive.mount` and `files.upload` cells and update `base_dir`.

## Limitations

- Accuracy is measured on the validation set, which also drives early stopping; a separate test set would be more reliable.
- Validation loss (0.36) is far above training loss (0.08), suggesting some overfitting.

## Author

**Harry Mardika** · [GitHub](https://github.com/harrymardika)
