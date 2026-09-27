import math

class Node:
    def __init__(self, depth, feature = None, threshold = None, prediction = None):
        self.depth = depth
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None
        self.prediction = prediction

    @staticmethod
    def count_classes(y):
        count = {}
        for current_class in y:
            if current_class not in count:
                count[current_class] = 0
            count[current_class] += 1
        return count

    @staticmethod
    def gini(y):
        if len(y) == 0:
            return 0.0

        count = Node.count_classes(y)
        total = len(y)
        impurity = 1.0

        for occurrences in count.values():
            probability = occurrences / total
            impurity -= probability ** 2

        return impurity

    def gini_split(self, X, y, feature, threshold):
        if len(y) == 0:
            return 0.0

        _, y_left, _, y_right = self.split(X, y, feature, threshold)
        return (
            (len(y_left) / len(y)) * self.gini(y_left)
            + (len(y_right) / len(y)) * self.gini(y_right)
        )

    @staticmethod
    def entropy(y):
        if len(y) == 0:
            return 0.0

        count = Node.count_classes(y)
        total = len(y)
        entropy_value = 0.0

        for occurrences in count.values():
            probability = occurrences / total
            entropy_value -= probability * math.log2(probability)

        return entropy_value

    @staticmethod
    def classification_error(y):
        if len(y) == 0:
            return 0.0

        count = Node.count_classes(y)
        total = len(y)
        majority_class_count = max(count.values())

        return 1 - (majority_class_count / total)

    def split(self, X, y, feature, threshold):
        X_left = []
        X_right = []
        y_left = []
        y_right = []

        for x, label in zip(X, y):
            if x[feature] < threshold:
                X_left.append(x)
                y_left.append(label)
            else:
                X_right.append(x)
                y_right.append(label)

        return X_left, y_left, X_right, y_right

    def find_best_split(self, X, y):
        if len(X) == 0 or len(y) == 0:
            return None

        best_gini = float("inf")
        number_of_features = len(X[0])

        for feature in range(number_of_features):
            values = sorted({x[feature] for x in X})

            for index in range(len(values) - 1):
                threshold = (values[index] + values[index + 1]) / 2
                split_gini = self.gini_split(X, y, feature, threshold)

                if split_gini < best_gini:
                    best_gini = split_gini
                    self.feature = feature
                    self.threshold = threshold

        return best_gini
