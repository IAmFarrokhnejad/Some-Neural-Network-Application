#Purpose of the work: Classification of wines(with generalization to ensure the accuracy of classification).(classification using an MLP Neural Network)
#Dataset includes a selection of Italian wines from the same region, made from three different grape varieties.
#Attributes included:Alcohol, Malic acid, Ash, Alkalinity of ash, Magnesium, Total phenols, Flavanoids, Nonflavanoid phenols, Proanthocyanidins, Color intensity, Hue, OD280/OD315 of diluted wines, Proline

#Credits: Morteza Farrokhnejad, Ali Farrokhnejad
#Based on the original work by Prof. Dr. Ahmet Rizaner


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(path):
    data = np.genfromtxt(path, delimiter=',')
    X = data[:-3, :].T  # Input features (13 dimensions)
    T = data[-3:, :].T   # One-hot encoded targets
    
    # Convert one-hot to class labels
    y = T.argmax(axis=1)
    
    # Standardize features
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), y

def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(50, 30),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=50,
        random_state=42,
        verbose=True
    )

def evaluate_model(model, X, y_true, title):
    y_pred = model.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()
    
    accuracy = np.mean(y_true == y_pred)
    print(f"Accuracy ({title}): {accuracy*100:.2f}%")
    return accuracy

# Load and prepare data
X, y = load_and_prepare_data('path to the dataset goes here') # Specify the dataset path 

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Create and train model
model = create_model()
model.fit(X_train, y_train)

# Evaluate performance
train_acc = evaluate_model(model, X_train, y_train, "Training Set")
test_acc = evaluate_model(model, X_test, y_test, "Testing Set")

# Plot learning curve
plt.figure(figsize=(8, 4))
plt.plot(model.loss_curve_, label='Training Loss')
if hasattr(model, 'validation_scores_'):
    plt.plot(model.validation_scores_, label='Validation Accuracy')
plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.legend()
plt.show()