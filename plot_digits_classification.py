# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Function to split data into train, development, and test subsets
def split_train_dev_test(X, y, test_size, dev_size):
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Further split the train data into train and development subsets
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=dev_size, shuffle=False)
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Function to predict and evaluate a model
def predict_and_eval(model, X_test, y_test):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)
    
    # Visualization of the first 4 test samples and show their predicted digit value in the title
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    
    # Print classification report
    print(f"Classification report for classifier {model}:\n"
          f"{metrics.classification_report(y_test, predicted)}\n")
    
    # Plot the confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

###############################################################################
# Digits dataset
# --------------

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into train, development, and test subsets
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(
    data, digits.target, test_size=0.2, dev_size=0.1
)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Evaluate the model on the development set
predict_and_eval(clf, X_dev, y_dev)

# Evaluate the model on the test set
predict_and_eval(clf, X_test, y_test)
