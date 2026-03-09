import os
import cv2
import numpy as np

DATASET_PATH = "dataset/train"

images = []
labels = []

for class_name in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):
        img_path = os.path.join(class_path, image_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (128, 128))

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize
        img = img / 255.0

        images.append(img)
        labels.append(class_name)

X_images = np.array(images)
y = np.array(labels)

print("Images shape:", X_images.shape)
print("Labels shape:", y.shape)
from skimage.feature import hog

hog_features = []

for img in X_images:
    feature = hog(img,
                  pixels_per_cell=(8,8),
                  cells_per_block=(2,2),
                  feature_vector=True)
    hog_features.append(feature)

X = np.array(hog_features)

print("HOG Feature shape:", X.shape)
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=1000)  # select top 1000 features
X_selected = selector.fit_transform(X, y)

print("After Feature Selection:", X_selected.shape)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_selected, y)

print("After LDA:", X_lda.shape)
from collections import Counter

print("Class distribution before SMOTE:")
print(Counter(y))
from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_lda, y)

print("After SMOTE shape:", X_balanced.shape)
print("Class distribution after SMOTE:")
print(Counter(y_balanced))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    random_state=42
)

bagging.fit(X_train, y_train)
y_pred_bag = bagging.predict(X_test)

print("Bagging Classification Report:")
print(classification_report(y_test, y_pred_bag))
from sklearn.ensemble import AdaBoostClassifier

boosting = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

boosting.fit(X_train, y_train)
y_pred_boost = boosting.predict(X_test)

print("Boosting Classification Report:")
print(classification_report(y_test, y_pred_boost))  