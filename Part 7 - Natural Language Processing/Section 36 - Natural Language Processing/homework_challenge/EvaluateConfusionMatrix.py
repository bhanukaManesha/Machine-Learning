class EvaluateConfusionMatrix():
    """
    proides a set of functions to find the accuracy, precison,recall and f1_score of a given confusion matrix
    """

    def __init__(self,confusion_matrix):
        self.FN = confusion_matrix[1][0]
        self.FP = confusion_matrix[0][1]
        self.TN = confusion_matrix[0][0]
        self.TP = confusion_matrix[1][1]
    def get_accuracy(self):
        """
        returns the accuracy of the confusion matrix by calcualting the total correct predictions and getting the average
        :return: average
        """
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def get_precision(self):
        """
        returns the precision of the confusion matrix
        :return: precision
        """
        return self.TP / (self.TP + self.FP)

    def get_recall(self):
        """
        returns the recall of the confusion matrix
        :return:
        """
        return self.TP / (self.TP + self.FN)

    def get_f1_score(self):
        """
        calculates the precision and recall and then returns the f1_score of the matrix
        :return:
        """
        Precision = self.get_precision()
        Recall = self.get_recall()
        return 2 * Precision * Recall / (Precision + Recall)


