"""
Metrics
"""

class Precision:
    def __init__(self):
        self.count = 0
        self.total = 1e-6
    def update(self, probs, labels):
        mask = labels != -1
        preds = probs > 0.5
        self.count += ((preds == labels) & (preds == 1) & mask).sum()
        self.total += ((preds == 1) & mask).sum()
    def compute(self):
        return float(self.count / self.total)
    def reset(self):
        self.count = 0
        self.total = 1e-6

class Recall:
    def __init__(self):
        self.count = 0
        self.total = 1e-6
    def update(self, probs, labels):
        mask = labels != -1
        preds = probs > 0.5
        self.count += ((preds == labels) & (preds == 1) & mask).sum()
        self.total += ((labels == 1) & mask).sum()
    def compute(self):
        return float(self.count / self.total)
    def reset(self):
        self.count = 0
        self.total = 1e-6

class HardIoU:
    def __init__(self):
        self.count = 0
        self.total = 1e-6
    def update(self, probs, labels):
        mask = labels != -1
        preds = probs > 0.5
        self.count += ((preds == labels) & (preds == 1) & mask).sum()
        self.total += (((preds == 1) | (labels == 1)) & mask).sum()
    def compute(self):
        return float(self.count / self.total)
    def reset(self):
        self.count = 0
        self.total = 1e-6