import numpy as np

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_regression(y_true, y_pred):
    """
    Calculate regression evaluation metrics.

    Returns:
        Dictionary containing regression metrics.
    """

    r2 = r2_score(y_true, y_pred)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    # Avoid division by zero in MAPE
    non_zero = np.asarray(y_true) != 0

    if non_zero.any():

        mape = np.mean(
            np.abs(
                (
                    np.asarray(y_true)[non_zero]
                    - np.asarray(y_pred)[non_zero]
                )
                /
                np.asarray(y_true)[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    return {
        "R2": float(r2),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


def evaluate_classification(
    y_true,
    y_pred,
    y_probability=None
):
    """
    Calculate classification evaluation metrics.

    Parameters:
        y_true: Actual labels
        y_pred: Predicted labels
        y_probability: Probability for positive class,
                       if available.

    Returns:
        Dictionary containing classification metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1)
    }

    # ROC-AUC requires probability scores
    if y_probability is not None:

        try:

            if y_probability.ndim == 1:

                auc = roc_auc_score(
                    y_true,
                    y_probability
                )

            else:

                auc = roc_auc_score(
                    y_true,
                    y_probability,
                    multi_class="ovr"
                )

            metrics["ROC-AUC"] = float(auc)

        except ValueError:

            metrics["ROC-AUC"] = None

    return metrics