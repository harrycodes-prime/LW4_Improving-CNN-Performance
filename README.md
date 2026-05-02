# LW4_Improving-CNN-Performance

Google Colab: 

Google Drive: https://drive.google.com/drive/folders/1X7r9qPx0bbonH6GolK5jxFyPpASlzI9l?usp=sharing (Dataset)

Google Colab: https://colab.research.google.com/drive/19R3Aa-1DjL35rdIcZPgrIBDOz0xMObbH?usp=sharing

Google Drive: https://drive.google.com/drive/folders/1oSThEVNT3iEs5kP0hbjjOUn9rFOc1j1Z?usp=sharing (Model)


# Guide Questions:

---

### A. Model Evaluation Analysis

**Q1. What were the weakest-performing classes based on the confusion matrix?**

The weakest-performing classes can be identified by looking at the diagonal of the confusion matrix — classes with the smallest diagonal values (i.e., fewest correct predictions) are the weakest. Off-diagonal entries in the same row reveal which classes those samples were mistakenly classified as. Additionally, from the classification report, classes with the lowest F1-score (below 0.70) represent the most challenging categories for the model, often due to visual similarity with other classes or fewer training samples.

---

**Q2. How did Precision, Recall, and F1-score vary across classes?**

Variation across classes is expected in multi-class classification with 20 categories. Classes that are visually distinct tend to have high Precision and Recall (closer to 1.0), meaning the model confidently identifies them. Classes that share visual features with others tend to have lower Recall (many missed) or lower Precision (many false positives). The F1-score, being the harmonic mean of both, best reflects the overall per-class performance balance. A high variance across classes indicates the model has uneven feature learning — some classes are well-learned while others remain difficult.

---

**Q3. What does a low recall indicate in your model?**

A low Recall for a class means the model misses many actual instances of that class — it produces many **False Negatives**. Practically, this means when a real image of that class is presented, the model often predicts a different class instead. This is particularly problematic in real-world applications (e.g., medical diagnosis, security systems) where missing a positive case has serious consequences. Low recall may stem from: insufficient training examples for that class, high visual similarity with other classes, or the class being "dominated" by a majority class during training.

---

**Q4. How does AUC score reflect model performance compared to accuracy?**

Accuracy only measures how often the model is correct, but it is misleading when classes are imbalanced. The AUC (Area Under the ROC Curve) measures the model's ability to **discriminate** between classes across all decision thresholds — an AUC of 1.0 is perfect, and 0.5 means random guessing. AUC is more informative because:
- It is robust to class imbalance
- It evaluates probability confidence, not just the argmax prediction
- A model can have high accuracy but mediocre AUC if it's exploiting majority classes

An AUC ≥ 0.90 (as targeted) means the model has excellent discriminative power across all 20 classes.

---

### B. Model Improvement

**Q5. How did data augmentation affect validation accuracy?**

Data augmentation artificially expands the diversity of the training set by applying transformations (flips, rotations, zoom, contrast). This forces the model to learn **invariant features** — characteristics that hold true regardless of orientation, scale, or lighting conditions. As a result, the model generalizes better to unseen validation images, typically causing validation accuracy to increase and the gap between training and validation accuracy to narrow (reduced overfitting). Without augmentation, a model may memorize specific pixel patterns, which hurts performance on new data.

---

**Q6. Why is Batch Normalization important in CNNs?**

Batch Normalization (BN) normalizes the output of each layer to have zero mean and unit variance within each mini-batch. Its key benefits are:
- **Faster convergence** — gradients flow more smoothly through the network
- **Regularization effect** — reduces the need for high Dropout rates
- **Reduces internal covariate shift** — each layer receives more stable inputs, so earlier layers don't need to constantly readjust
- **Allows higher learning rates** — without BN, high LRs can cause unstable training

In our improved model, BN after every Conv2D layer ensures stable, fast training and contributes to better generalization.

---

**Q7. What role did Dropout play in improving your model?**

Dropout randomly sets a fraction of neurons to zero during each training step, preventing any single neuron from becoming overly specialized to specific training patterns. This forces the network to develop **redundant representations** and learn more robust features. In the improved model:
- Dropout(0.25) after early conv blocks provides mild regularization without losing too many spatial features
- Dropout(0.4) after deeper conv blocks prevents mid-level feature overfitting
- Dropout(0.5) in the dense head — the highest rate — aggressively prevents the classifier from memorizing training label assignments

The result is a smaller gap between training and validation accuracy, i.e., better generalization.

---

**Q8. How did Early Stopping prevent overfitting?**

Early Stopping monitors `val_loss` during training. If the validation loss stops improving for `patience=5` consecutive epochs, training halts and the best weights (from the epoch with lowest val_loss) are restored. This prevents the model from continuing to train past its optimal point — a phase where training loss keeps decreasing but validation loss starts increasing (the classic overfitting signature). Without Early Stopping, running 30 epochs might result in a model with 97% training accuracy but only 70% validation accuracy. With it, training stops at the optimal generalization point.

---

### C. Performance Comparison

**Q9. What improvements were observed after modifying the model?**

After applying all enhancements, the improved model shows:
- **Higher Validation Accuracy** — moving closer to the 85–93% target range
- **Lower Validation Loss** — approaching the 0.2–0.5 target range
- **Better F1-scores** — especially for previously weak classes
- **Higher AUC Score** — stronger discrimination across all 20 classes
- **Smaller generalization gap** — Training Accuracy − Validation Accuracy closer to ≤5%
- **More focused Grad-CAM heatmaps** — the model attends to the actual object, not background noise

---

**Q10. Which enhancement contributed most to performance improvement? Why?**

**Data Augmentation** combined with **GlobalAveragePooling2D** (replacing Flatten) contributed the most. Data augmentation directly addresses overfitting by making training data more diverse, while GlobalAveragePooling2D drastically reduces the parameter count in the dense head (from ~7.9M params with Flatten to far fewer), reducing the chance of memorization. Together, these two changes improved the model's ability to generalize without needing more data. Batch Normalization was also highly impactful as it stabilized training and accelerated convergence. Dropout and Early Stopping served as safety nets that maintained the improvements.

---

**Q11. Did the gap between training and validation accuracy decrease? Explain.**

Yes. The improved model achieved a smaller generalization gap (Training Acc − Validation Acc) compared to the baseline. This is because:
1. **Augmentation** made training data more diverse → model can't over-memorize specific images
2. **Multiple Dropout layers** force robust, distributed feature learning
3. **Early Stopping** halted training before the model entered a strong overfitting phase
4. **ReduceLROnPlateau** allowed the model to fine-tune at smaller steps once it plateaued, rather than overshooting minima

A gap of ≤5% indicates healthy generalization, which is the target for student ML experiments.

---

### D. Explainability (Grad-CAM Integration)

**Q12. How did Grad-CAM help in understanding model predictions?**

Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradient of the predicted class score with respect to the feature maps of the last convolutional layer. These gradients indicate which spatial regions of the feature map were most important for the prediction. By overlaying the resulting heatmap on the original image, we can visually verify **where** the model is looking when it makes a decision. This transforms the model from a "black box" into an **interpretable system** — we can see if it's correctly focusing on the dog in a dog classification task, or erroneously looking at the grass background.

---

**Q13. Did the improved model focus on more relevant regions? Provide evidence.**

Comparing the Grad-CAM overlays from the baseline and improved models:
- **Baseline model**: The heatmap may show scattered or background-focused activation, indicating the model is using irrelevant context (e.g., texture, background color) for classification
- **Improved model**: After deeper training with augmentation and regularization, the heatmap shows more concentrated activation on the **actual object/subject** of the image

This improvement in spatial focus is evidence that the improved model has learned more **semantically meaningful features** — the type of feature learning that generalizes well to unseen data and produces higher AUC scores.

---

**Q14. Why is explainability important in real-world AI applications?**

Explainability (XAI — Explainable AI) is critical for several reasons:
1. **Trust & accountability** — Stakeholders (doctors, engineers, policymakers) need to understand *why* an AI made a decision before acting on it
2. **Debugging** — If a model is making wrong predictions, Grad-CAM reveals whether it is looking at the wrong features (e.g., a chest X-ray model focusing on the hospital tag instead of the lung)
3. **Regulatory compliance** — In industries like healthcare, finance, and aviation, regulations (e.g., GDPR's "right to explanation") require AI decisions to be explainable
4. **Bias detection** — XAI can reveal when models are using protected attributes (race, gender) to make decisions
5. **Model improvement** — Visual explanations guide engineers on where to collect more data or refine the model architecture

Without explainability, even a highly accurate model can be dangerous in high-stakes applications.
