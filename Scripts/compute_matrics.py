def compute_metrics(p):
    from seqeval.metrics import precision_score, recall_score, f1_score
    preds = p.predictions.argmax(axis=2)
    true_labels = [
        [id2label[l] for l in label if l != -100] for label in p.label_ids
    ]
    true_preds = [
        [id2label[pred] for pred, lab in zip(pred_row, label) if lab != -100]
        for pred_row, label in zip(preds, p.label_ids)
    ]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }