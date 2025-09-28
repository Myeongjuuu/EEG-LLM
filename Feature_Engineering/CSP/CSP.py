import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.decoding import CSP

# Function to load the dataset
def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)  # Load training data
    val_df = pd.read_csv(val_path)      # Load validation data
    test_df = pd.read_csv(test_path)    # Load test data

    return train_df, val_df, test_df

# Function to preprocess the data and reshape it
def preprocess_data(data):
    channels = data.columns[:-1]  # Exclude the last column which is the label
    X = data[channels].values     # Features (EEG data)
    y = data['label'].values      # Labels

    # Define number of channels and time points
    n_channels = len(channels)
    n_times = 1000  # Number of time points per sample

    # Calculate the number of epochs
    n_epochs = len(y) // n_times

    # Reshape the data into epochs (n_epochs x n_channels x n_times)
    X_reshaped = X.reshape((n_epochs, n_channels, n_times))
    y_reshaped = y[::n_times]  # Keep one label per epoch

    return X_reshaped, y_reshaped

# Function to plot CSP-transformed data
def plot_csp_transformed_data(X_transformed, y, label1, label2):
    plt.figure(figsize=(8, 6))
    
    # Plot data for class 1
    plt.scatter(X_transformed[y == 1, 0], X_transformed[y == 1, 1], label=f'Label {label1}', alpha=0.7)
    
    # Plot data for class 2
    plt.scatter(X_transformed[y == 0, 0], X_transformed[y == 0, 1], label=f'Label {label2}', alpha=0.7)
    
    plt.xlabel('CSP Component 1')
    plt.ylabel('CSP Component 2')
    plt.title(f'CSP Visualization for Label {label1} vs Rest')
    plt.legend()
    plt.show()

# Function to apply CSP and plot the transformed data for each label comparison
def apply_csp_and_plot(X_train, y_train, X_test, y_test, output_dir, label1, label2):
    # Apply CSP (Common Spatial Pattern) to extract features
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    csp.fit(X_train, y_train)
    X_test_transformed = csp.transform(X_test)
    
    # Binary labeling for the current comparison (label1 vs rest)
    y_test_binary = np.where(y_test == label1, 1, np.where(y_test == label2, 0, -1))
    valid_test_indices = y_test_binary != -1
    X_test_binary = X_test_transformed[valid_test_indices]
    y_test_binary = y_test_binary[valid_test_indices]

    # Visualize the CSP-transformed data
    plot_csp_transformed_data(X_test_binary, y_test_binary, label1, label2)

# File paths for each label comparison
# Label 1 vs Rest
train_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 1 vs Rest/train.csv"
val_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 1 vs Rest/val.csv"
test_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 1 vs Rest/test.csv"

# Output directory for saving features
output_dir = "C:/Users/windows/Desktop/CSP/Dataset/Label 1 vs Rest/CSP_Features"

# Load and preprocess the data
train_data, val_data, test_data = load_data(train_path, val_path, test_path)
X_train_epochs, y_train_labels = preprocess_data(train_data)
X_val_epochs, y_val_labels = preprocess_data(val_data)
X_test_epochs, y_test_labels = preprocess_data(test_data)

# Apply CSP and visualize for Label 1 vs Rest
apply_csp_and_plot(X_train_epochs, y_train_labels, X_test_epochs, y_test_labels, output_dir, label1=1, label2=0)

# Label 2 vs Rest
train_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 2 vs Rest/train.csv"
val_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 2 vs Rest/val.csv"
test_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 2 vs Rest/test.csv"

output_dir = "C:/Users/windows/Desktop/CSP/Dataset/Label 2 vs Rest/CSP_Features"

train_data, val_data, test_data = load_data(train_path, val_path, test_path)
X_train_epochs, y_train_labels = preprocess_data(train_data)
X_val_epochs, y_val_labels = preprocess_data(val_data)
X_test_epochs, y_test_labels = preprocess_data(test_data)

# Apply CSP and visualize for Label 2 vs Rest
apply_csp_and_plot(X_train_epochs, y_train_labels, X_test_epochs, y_test_labels, output_dir, label1=2, label2=0)

# Label 3 vs Rest
train_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 3 vs Rest/train.csv"
val_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 3 vs Rest/val.csv"
test_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 3 vs Rest/test.csv"

output_dir = "C:/Users/windows/Desktop/CSP/Dataset/Label 3 vs Rest/CSP_Features"

train_data, val_data, test_data = load_data(train_path, val_path, test_path)
X_train_epochs, y_train_labels = preprocess_data(train_data)
X_val_epochs, y_val_labels = preprocess_data(val_data)
X_test_epochs, y_test_labels = preprocess_data(test_data)

# Apply CSP and visualize for Label 3 vs Rest
apply_csp_and_plot(X_train_epochs, y_train_labels, X_test_epochs, y_test_labels, output_dir, label1=3, label2=0)

# Label 4 vs Rest
train_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 4 vs Rest/train.csv"
val_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 4 vs Rest/val.csv"
test_path = "C:/Users/windows/Desktop/CSP/Dataset/Label 4 vs Rest/test.csv"

output_dir = "C:/Users/windows/Desktop/CSP/Dataset/Label 4 vs Rest/CSP_Features"

train_data, val_data, test_data = load_data(train_path, val_path, test_path)
X_train_epochs, y_train_labels = preprocess_data(train_data)
X_val_epochs, y_val_labels = preprocess_data(val_data)
X_test_epochs, y_test_labels = preprocess_data(test_data)

# Apply CSP and visualize for Label 4 vs Rest
apply_csp_and_plot(X_train_epochs, y_train_labels, X_test_epochs, y_test_labels, output_dir, label1=4, label2=0)
