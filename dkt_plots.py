import matplotlib.pyplot as plt
import numpy as np


def plot_history(history_file):
    """
    Plots the training loss and test AUC from the history file.
    """
    # Read the history file
    training_loss = []
    test_auc = []
    test_acc = []

    with open(history_file, 'r') as f:
        for line in f:
            epoch_data = line.strip().split('\t')
            training_loss.append(float(epoch_data[0]))
            test_auc.append(float(epoch_data[1]))
            test_acc.append(float(epoch_data[2]))

    # Plot Training Loss and Test AUC
    epochs = np.arange(1, len(training_loss) + 1)

    plt.figure(figsize=(12, 6))

    # Subplot for training loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, training_loss, label='Training Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')
    plt.grid(True)

    # Subplot for test AUC
    plt.subplot(1, 3, 2)
    plt.plot(epochs, test_auc, label='Test AUC', color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Test AUC')
    plt.title('Test AUC over Epochs')
    plt.grid(True)

    # Subplot for test AUC
    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_acc, label='Test ACC', color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Test ACC')
    plt.title('Test ACC over Epochs')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_predictions(preds_file):
    """
    Plots the predicted probabilities and observed student recall values.
    """
    # Read the predictions file
    probs_recall = []
    student_recalled = []

    with open(preds_file, 'r') as f:
        # Skip the header
        next(f)
        for line in f:
            data = line.strip().split('\t')
            probs_recall.append(float(data[1]))
            student_recalled.append(int(data[2]))

    # Convert to numpy arrays for easier handling
    probs_recall = np.array(probs_recall)
    student_recalled = np.array(student_recalled)

    # Plot Predicted Probabilities vs Student Recall
    plt.figure(figsize=(8, 6))
    plt.scatter(probs_recall, student_recalled, alpha=0.6, color='b', label='Predictions')
    plt.xlabel('Predicted Probability of Recall')
    plt.ylabel('Student Recalled (1 = Yes, 0 = No)')
    plt.title('Predicted Recall Probability vs Actual Recall')
    plt.grid(True)
    plt.show()

def load_dkt_xes3g5m_k_cv(dkt_folder="dkt_xes3g5m_k_cv", fold=1):
    prefix = f"{dkt_folder}/5_fold_35/fold{fold}/dataset"
    history_file = prefix +'.txt.history'
    preds_file = prefix +'.txt.preds'
    return history_file, preds_file

def load_dkt_xes3g5m_90_10(dkt_folder="dkt_xes3g5m_90_10"):
    prefix = f"{dkt_folder}/dataset"
    history_file = prefix +'.txt.history'
    preds_file = prefix +'.txt.preds'
    return history_file, preds_file


if __name__ == "__main__":
    # Set your file paths here
    # history_file, preds_file = load_dkt_xes3g5m_k_cv(fold=1)
    history_file, preds_file = load_dkt_xes3g5m_90_10()

    # Plot the history (Training Loss and Test AUC)
    plot_history(history_file)

    # Plot the predictions (Predicted Probability vs Actual Recall)
    plot_predictions(preds_file)
