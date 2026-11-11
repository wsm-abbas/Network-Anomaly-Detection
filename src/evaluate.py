"""Small helper so both models are scored the same way."""
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def score(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": name,
        "precision": round(precision_score(y_true, y_pred), 3),
        "recall": round(recall_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        # share of normal traffic that was wrongly flagged
        "false_alarm_rate": round(fp / (fp + tn), 4),
        "false_alarms": int(fp),
        "missed_attacks": int(fn),
    }
