import torch
import numpy as np


class Estimator():
    def __init__(self, criterion, num_classes, thresholds=None):
        self.criterion = criterion
        self.num_classes = num_classes
        self.thresholds = [-0.5 + i for i in range(num_classes)] if not thresholds else thresholds
        self.prediction = None
        self.arrival = None
        self.error = float('inf')
        self.mae = 0.
        self.rmse = 0.

        self.reset()  # intitialization

    def update(self, predictions, targets):
        targets = targets.data.cpu()
        predictions = predictions.data.cpu()
        predictions = self.to_prediction(predictions)

        # update metrics
        self.num_samples += len(predictions)
        self.correct += (predictions == targets).sum().item()
        for i, p in enumerate(predictions):
            self.conf_mat[int(targets[i])][int(p.item())] += 1

    def get_accuracy(self, digits=-1):
        acc = self.correct / self.num_samples
        acc = acc if digits == -1 else round(acc, digits)
        return acc

    def get_kappa(self, digits=-1):
        kappa = quadratic_weighted_kappa(self.conf_mat)
        kappa = kappa if digits == -1 else round(kappa, digits)
        return kappa

    def update_error(self, predictions, targets):
        targets = targets.data.cpu()
        predictions = predictions.data.cpu()

        # update metrics
        self.num_samples += len(predictions)
        if self.prediction is None:
            self.prediction = predictions.cpu().detach().numpy()
        else:
            self.prediction = np.concatenate((self.prediction, predictions.cpu().detach().numpy()), axis=0)

        if self.arrival is None:
            self.arrival = targets.cpu().detach().numpy()
        else:
            self.arrival = np.concatenate((self.arrival, targets.cpu().detach().numpy()), axis=0)

    def get_error(self):
        targets = self.arrival.reshape(-1, 1)
        predictions = self.prediction.reshape(-1, 1)
        self.mae = np.mean(np.abs(targets - predictions)).item()
        self.rmse = np.sqrt(np.mean((targets - predictions)**2)).item()
        self.error = [self.mae, self.rmse]

    def reset(self):
        self.correct = 0
        self.num_samples = 0
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=int)
        self.prediction = None
        self.arrival = None
        self.error = float('inf')
        self.mae = 0.
        self.rmse = 0.

    def to_prediction(self, predictions):
        if self.criterion in ['cross_entropy', 'focal_loss', 'kappa_loss']:
            predictions = torch.tensor(
                [torch.argmax(p) for p in predictions]
            ).long()
        elif self.criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:
            predictions = torch.tensor(
                [self.classify(p.item()) for p in predictions]
            ).float()
        else:
            raise NotImplementedError('Not implemented criterion.')

        return predictions

    def classify(self, predict):
        thresholds = self.thresholds
        predict = max(predict, thresholds[0])
        for i in reversed(range(len(thresholds))):
            if predict >= thresholds[i]:
                return i


def quadratic_weighted_kappa(conf_mat):
    assert conf_mat.shape[0] == conf_mat.shape[1]
    cate_num = conf_mat.shape[0]

    # Quadratic weighted matrix
    weighted_matrix = np.zeros((cate_num, cate_num))
    for i in range(cate_num):
        for j in range(cate_num):
            weighted_matrix[i][j] = 1 - float(((i - j)**2) / ((cate_num - 1)**2))

    # Expected matrix
    ground_truth_count = np.sum(conf_mat, axis=1)
    pred_count = np.sum(conf_mat, axis=0)
    expected_matrix = np.outer(ground_truth_count, pred_count)

    # Normalization
    conf_mat = conf_mat / conf_mat.sum()
    expected_matrix = expected_matrix / expected_matrix.sum()

    observed = (conf_mat * weighted_matrix).sum()
    expected = (expected_matrix * weighted_matrix).sum()
    return (observed - expected) / (1 - expected)
