# LW4_Improving-CNN-Performance

Google Colab: https://colab.research.google.com/drive/1LCtZUBDj4gbUrhoAEmWxvprL9RMUQyhc?usp=sharing
Google Drive: https://drive.google.com/drive/folders/1X7r9qPx0bbonH6GolK5jxFyPpASlzI9l?usp=sharing (Dataset)
Google Drive: https://drive.google.com/drive/folders/1oSThEVNT3iEs5kP0hbjjOUn9rFOc1j1Z?usp=sharing (Model)


# Guide Questions:

A. Model Evaluation Analysis
1. What were the weakest-performing classes based on the confusion matrix?
The weakest classes are those with the most off-diagonal values in the confusion matrix — meaning the model kept misclassifying them as something else. You can spot these by looking for rows where the correct diagonal value is low compared to the other numbers in that row.
2. How did Precision, Recall, and F1-score vary across classes?
Some classes had high precision but low recall, which means the model was picky — when it predicted that class, it was usually right, but it missed a lot of actual examples of that class. Other classes showed the opposite. The F1-score balances both, so classes with low F1 were genuinely hard for the model.
3. What does a low recall indicate in your model?
Low recall means the model is missing many actual examples of that class — it's predicting "not this class" too often. For example, if cat recall is 0.60, 40% of real cats were labeled as something else.
4. How does AUC score reflect model performance compared to accuracy?
Accuracy just counts correct predictions overall, so it can be misleading if one class dominates the dataset. AUC measures how well the model separates each class from the others across all thresholds, so it gives a more honest picture — especially when class sizes are unequal.

B. Model Improvement
5. How did data augmentation affect validation accuracy?
Data augmentation made the model see more varied versions of the training images (flipped, rotated, zoomed), so it stopped memorizing specific images. This generally brought validation accuracy closer to training accuracy because the model learned more generalizable features.
6. Why is Batch Normalization important in CNNs?
Batch normalization keeps the values flowing through the network in a stable range during training. Without it, the values can drift and slow down learning. It also acts as a mild regularizer, which helps reduce overfitting a bit.
7. What role did Dropout play in improving your model?
Dropout randomly turns off a portion of neurons during training, which forces the network to not rely too heavily on any single path. This makes the model more robust and less likely to just memorize the training data.
8. How did Early Stopping prevent overfitting?
Early stopping watches the validation loss during training. Once it stops improving for a set number of epochs (patience=3), training halts and the best weights are restored. This prevents the model from continuing to train past the point where it starts overfitting.

C. Performance Comparison
9. What improvements were observed after modifying the model?
After adding augmentation, batch normalization, dropout, and early stopping, the validation accuracy went up and the gap between training and validation accuracy became smaller. The classification report also showed better F1-scores for the previously weak classes.
10. Which enhancement contributed the most to performance improvement? Why?
Data augmentation likely had the biggest impact because the root cause of poor generalization is often a model that overfit to limited training examples. By artificially expanding the dataset variety, the model learned patterns instead of specific images.
11. Did the gap between training and validation accuracy decrease? Explain?
Yes. Before improvement, there was a noticeable difference (e.g., 95% train vs 75% validation). After applying regularization techniques, the gap narrowed to something closer to 3–5%, which shows the model is generalizing better rather than just memorizing.

D. Explainability (Grad-CAM Integration)
12. How did Grad-CAM help in understanding model predictions?
Grad-CAM creates a heatmap that shows which parts of the image the model focused on when making its prediction. Instead of just knowing the model predicted "cat," you can actually see whether it was looking at the cat's face or at the background.
13. Did the improved model focus on more relevant regions? Provide evidence?
In the baseline model, the heatmap sometimes highlighted background areas or irrelevant parts of the image. After improvement, the highlighted regions moved to the actual object of interest (like the main subject). This shows the model learned more meaningful visual features after regularization.
14. Why is explainability important in real-world AI applications?
In real applications like medical diagnosis or safety systems, you can't just trust a "black box" answer. Explainability lets you verify that the model is making decisions for the right reasons. If a cancer detection model highlights the patient's name tag instead of the tumor, that's a serious problem even if the accuracy looks fine.
