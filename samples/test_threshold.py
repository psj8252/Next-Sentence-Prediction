import csv
from random import randint, shuffle

import nsp
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _get_metrics(true_labels, pred_labels):
    """
    :param true_labels: (iterable) correct label.
    :param pred_labels: (iterable) predicted label by model.
    :return: (accuracy, precision, recall, f1) (tuple of float)
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="binary", zero_division=0
    )
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    """
    Data file format: (id, context, query, reply) tsv
    It is same as training data file format
    """
    with open("validation.tsv", "r") as f:
        rdr = csv.reader(f, delimiter="\t")
        header = next(rdr)

        # Load data file
        data = list(rdr)

        # Make validation dataset
        random_replies = [row[-1] for row in data]
        shuffle(random_replies)

        labels = [randint(0, 1) for _ in range(len(data))]
        labeled_data = []
        for datum, random_reply, label in zip(data, random_replies, labels):
            if label == 0:
                datum[-1] = random_reply
            datum.append(label)
            labeled_data.append(datum)

    inputs = [(c, q, r) for _, c, q, r, _ in labeled_data]

    # Load inferencer & Predict
    inferencer = nsp.manager.InferManager("infer_config.yml", str.split, "cuda")
    _, preds = inferencer.inference_texts(inputs)
    preds = np.array(preds)

    # Check result by threshold
    threshold = 0.3
    while threshold < 0.71:
        acc, prec, recl, f1 = _get_metrics(labels, preds >= threshold)
        print(f"Threshold: {threshold:.2f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Recl: {recl:.4f}, f1: {f1:.4f}")
        threshold += 0.01

    output_data = [datum + [pred] for datum, pred in zip(labeled_data, preds)]

    with open("nsp_labeled.tsv", "w") as f:
        wtr = csv.writer(f, delimiter="\t")
        wtr.writerow(header + ["prediction"])
        wtr.writerows(output_data)
