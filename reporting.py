import numpy as np

from collections import defaultdict
from itertools import chain

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from prettytable import PrettyTable

from constants import LABELS


class StatsManager:

    def __init__(self):
        self.reports = []
        self.y_pred = []
        self.y_true = []

    def transform(self, data):
        t_data = []
        for l in data:
            ele = []
            for e in l:
                ele.append(LABELS[e])
            t_data.append(ele)
        return t_data

    def append_report(self, y_true, y_pred):

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_)
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

        report = classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
            output_dict=True
        )

        self.reports.append(report)

    def summarize(self):
        summary = defaultdict(lambda: defaultdict(list))

        for report in self.reports:
            for key in report.keys():
                for metric in report[key].keys():
                    summary[key][metric].append(report[key][metric])

        report = defaultdict(dict)

        for key in summary.keys():
            metrics = defaultdict()
            for metric in summary[key].keys():
                metrics[metric] = [np.mean(summary[key][metric]), np.std(summary[key][metric])]
            report[key] = metrics
        return report, summary


def pretty_print_report(report):
    table = PrettyTable(["", "Precision", "Recall", "F1-Score", "Support"])
    for obj in report:
        if obj in LABELS.keys():
            precision = report[obj]["precision"]
            recall = report[obj]["recall"]
            f1 = report[obj]["f1-score"]
            sup = report[obj]["support"]
            table.add_row([obj, "{:03.2f} ({:03.2f})".format(precision[0], precision[1]),
                           "{:03.2f} ({:03.2f})".format(recall[0], recall[1]),
                           "{:03.2f} ({:03.2f})".format(f1[0], f1[1]),
                           "{:03.2f} ({:03.2f})".format(sup[0], sup[1])])

    print(table)
