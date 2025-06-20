import os

from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.classifiers.logistic_regression_classifier import LogisticRegressionSpamClassifier
from spam_detector_ai.classifiers.naive_bayes_classifier import NaiveBayesClassifier
from spam_detector_ai.classifiers.random_forest_classifier import RandomForestSpamClassifier
from spam_detector_ai.loading_and_processing.preprocessor import Preprocessor
from spam_detector_ai.prediction.performance import ModelAccuracy


def get_model_path(model_type):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)

    # define paths
    paths_map = {
        ClassifierType.NAIVE_BAYES: (
            'models/bayes/naive_bayes_model.joblib',
            'models/bayes/naive_bayes_vectoriser.joblib'
        ),
        ClassifierType.RANDOM_FOREST: (
            'models/random_forest/random_forest_model.joblib',
            'models/random_forest/random_forest_vectoriser.joblib'
        ),
        ClassifierType.LOGISTIC_REGRESSION: (
            'models/logistic_regression/logistic_regression_model.joblib',
            'models/logistic_regression/logistic_regression_vectoriser.joblib'
        )
    }

    relative_path_model, relative_path_vectoriser = paths_map.get(model_type)
    if relative_path_model and relative_path_vectoriser:
        absolute_path_model = os.path.join(base_dir, relative_path_model)
        absolute_path_vectoriser = os.path.join(base_dir, relative_path_vectoriser)
        return absolute_path_model, absolute_path_vectoriser
    else:
        # raise if model type is invalid
        raise ValueError(f"Invalid model type: {model_type}")


class SpamDetector:
    def __init__(self, model_type=ClassifierType.NAIVE_BAYES):
        classifier_map = {
            ClassifierType.NAIVE_BAYES.value: NaiveBayesClassifier(),
            ClassifierType.RANDOM_FOREST.value: RandomForestSpamClassifier(),
            ClassifierType.LOGISTIC_REGRESSION.value: LogisticRegressionSpamClassifier(),
        }
        classifier = classifier_map.get(model_type.value)
        if not classifier:
            raise ValueError(f"Invalid model type: {model_type}")

        self.model = classifier
        self.model_type = model_type
        model_path, vectoriser_path = get_model_path(model_type)
        self.model.load_model(model_path, vectoriser_path)
        self.processor = Preprocessor()

    def is_spam(self, message_):
        processed_message = self.processor.preprocess_text(message_)
        vectorized_message = self.model.vectoriser.transform([processed_message]).toarray()
        prediction = self.model.classifier.predict(vectorized_message)
        return prediction[0] == 'spam'

    def test_is_spam(self, message_):
        processed_message = self.processor.preprocess_text(message_)
        vectorized_message = self.model.vectoriser.transform([processed_message]).toarray()
        prediction = self.model.classifier.predict(vectorized_message)
        return prediction[0]


class VotingSpamDetector:
    def __init__(self):
        total_accuracy = ModelAccuracy.total_accuracy()
        self.detectors = [
            (SpamDetector(model_type=ClassifierType.NAIVE_BAYES), ModelAccuracy.NAIVE_BAYES / total_accuracy),
            (SpamDetector(model_type=ClassifierType.RANDOM_FOREST), ModelAccuracy.RANDOM_FOREST / total_accuracy),
            (SpamDetector(model_type=ClassifierType.LOGISTIC_REGRESSION), ModelAccuracy.LOGISTIC_REG / total_accuracy)
        ]
        self.model_names = {
            ClassifierType.NAIVE_BAYES: "Naive Bayes",
            ClassifierType.RANDOM_FOREST: "Random Forest",
            ClassifierType.LOGISTIC_REGRESSION: "Logistic Regression"
        }

    def is_spam(self, message_):

        total_weight = sum(weight for _, weight in self.detectors)
        votes = []
        
        for detector, weight in self.detectors:
            # get result from each model
            is_spam_result = detector.is_spam(message_)
            votes.append((is_spam_result, weight))
        
        # calculate weighted spam score
        weighted_spam_score = sum(vote * weight for vote, weight in votes)
        
        # normalize spam score to be between 0 and 1
        normalized_score = weighted_spam_score / total_weight
        
        # determine final result - spam if score > 0.5
        final_decision = normalized_score > 0.5
        
        return final_decision


if __name__ == "__main__":
    voting_detector = VotingSpamDetector()
