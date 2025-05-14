
# 🐱 Cats vs Dogs Classifier with PyTorch

This project is a beginner-friendly image classification task using PyTorch and a CNN (Convolutional Neural Network) to distinguish between cats and dogs. It's not about solving a real-world problem — it's about **understanding how machine learning models learn from data**.

---

## 🎯 What is this?

This notebook walks through the full supervised learning pipeline:

- loading and preprocessing an image dataset (Microsoft Cats vs Dogs from Kaggle),
- building a custom PyTorch Dataset and DataLoader,
- designing and training a simple CNN from scratch,
- evaluating model performance (loss and accuracy),
- understanding model outputs (logits and predictions),
- and learning the theory behind **classification, targets, and labels**.

---

## 🤔 Why this project?

As a beginner in machine learning and computer vision, I wanted to **really understand**:

- how a model learns to tell two classes apart (even without knowing what a "cat" or "dog" is),
- how supervised learning works (labels + loss → learning),
- why it's important that the number of classes matches the data,
- and how accuracy improves with epochs and optimization.

This project is a playground to experiment, ask "what if..." questions, and build true understanding — not just run code.

---

## 📚 What I learned

- ✅ How image classification works in PyTorch  
- ✅ How to implement a custom `Dataset` and use `transforms`  
- ✅ How CNN layers (Conv2d + ReLU + MaxPool) extract features from images  
- ✅ How to interpret model outputs (logits) and convert them into predictions  
- ✅ How loss functions (like `CrossEntropyLoss`) guide learning  
- ✅ Why labels and output size must match  
- ✅ What happens when you confuse the model with wrong or missing labels

---

## 🧪 Dataset

- Source: [Kaggle Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset)
- Structure:  
  - `PetImages/Cat/*.jpg` → label `0`  
  - `PetImages/Dog/*.jpg` → label `1`  
- Used basic data filtering to remove corrupt images

---

## 🧠 Model architecture

```python
SimpleCNN(
  Conv2d(3, 16, kernel_size=3, padding=1),
  ReLU(),
  MaxPool2d(2),
  Conv2d(16, 32, kernel_size=3, padding=1),
  ReLU(),
  MaxPool2d(2),
  Flatten(),
  Linear(32 * 32 * 32, 64),
  ReLU(),
  Linear(64, 2)
)
```

---

## 📊 Training results (example)

```
Epoch 1/5, Loss: 44.51, Accuracy: 49.15%
Epoch 2/5, Loss: 43.11, Accuracy: 51.40%
Epoch 3/5, Loss: 42.38, Accuracy: 57.20%
Epoch 4/5, Loss: 41.91, Accuracy: 59.55%
Epoch 5/5, Loss: 39.71, Accuracy: 64.75%
```

---

## 💡 Key Takeaways

- A model doesn’t “know” what a cat is. It just sees patterns.
- You must **know the target (label)** when training — the model doesn’t guess it.
- Adding an extra class that doesn’t exist in the data **confuses the model**.
- Accuracy is not everything — understanding the process is more important.

---

## 🚀 Next steps

You can extend this project by:
- Adding a validation split
- Applying data augmentation (flips, rotations, etc.)
- Using a pretrained model like ResNet18
- Deploying it in a web demo

---

## 📦 How to run this notebook

This was developed in a Kaggle notebook, using PyTorch and GPU acceleration.  
To run locally, you'll need:

- Python 3.10+
- PyTorch
- torchvision
- Jupyter or VSCode with Jupyter extension

---

## ✍️ Author note

I created this project not to impress, but to understand.  
If you're just starting out too — feel free to fork this, break it, modify it, and learn.

---

## 🐾 Bonus: Why cats vs dogs?

Because it’s the "Hello World" of computer vision. 🐶🐱  
It’s simple, visual, and easy to verify predictions with your own eyes.
