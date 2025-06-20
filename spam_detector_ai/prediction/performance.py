class ModelAccuracy:
    NAIVE_BAYES = 0.75
    RANDOM_FOREST = 0.99
    LOGISTIC_REG = 0.97

    @classmethod
    def total_accuracy(cls):
        return sum([cls.NAIVE_BAYES, cls.RANDOM_FOREST, cls.LOGISTIC_REG])
