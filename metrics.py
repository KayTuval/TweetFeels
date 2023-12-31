import numpy as np
from sklearn.metrics import f1_score


def f1_score_func(preds, labels):
    """
    Calculate the weighted F1 score for predictions.

    Args:
        preds: The predicted labels.
        labels: The true labels.

    Returns:
        The weighted F1 score.
    """
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_per_class(preds, labels, label_dict):
    """
    Print the accuracy per class for predictions.

    Args:
        preds: The predicted labels.
        labels: The true labels.
        label_dict: A dictionary mapping labels to integers.

    """
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')
