# Thêm đoạn này vào đầu file
project_root = Path(__file__).parent.parent.parent # Có thể cần thêm một .parent nữa nếu custom_models ngang hàng với spam_detector_ai
sys.path.append(str(project_root))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from spam_detector_ai.classifiers.base_classifier import BaseClassifier
from spam_detector_ai.classifiers.classifier_types import ClassifierType
from spam_detector_ai.logger_config import init_logging
from spam_detector_ai.training.train_models import ModelTrainer

# Giả định bạn đã viết CustomLogisticRegression
from spam_detector_ai.custom_models.custom_logistic import CustomLogisticRegression

if __name__ == '__main__':
    logger = init_logging()

    logger.info("Define parameter grid for Custom Logistic Regression")
    param_grid = {
        'lr': [0.001, 0.01, 0.1, 1],
        'max_iter': [100, 300, 500]
    }

    trainer = ModelTrainer(data_path='../data/spam.csv',
                           classifier_type=ClassifierType.LOGISTIC_REGRESSION,
                           logger=logger)

    logger.info("Splitting the data")
    X_train, X_test, y_train, y_test = trainer.split_data_()

    logger.info("Vectorising the data")
    vectoriser = TfidfVectorizer(**BaseClassifier.VECTORIZER_PARAMS)
    X_train_vect = vectoriser.fit_transform(X_train).toarray()
    X_test_vect = vectoriser.transform(X_test).toarray()

    best_acc = 0
    best_params = None
    best_model = None

    for lr in param_grid['lr']:
        for max_iter in param_grid['max_iter']:
            logger.info(f"Training with lr={lr}, max_iter={max_iter}")
            model = CustomLogisticRegression(lr=lr, max_iter=max_iter)
            model.fit(X_train_vect, y_train)

            y_pred = model.predict(X_test_vect)
            acc = accuracy_score(y_test, y_pred)

            logger.info(f"Accuracy: {acc}")

            if acc > best_acc:
                best_acc = acc
                best_params = {'lr': lr, 'max_iter': max_iter}
                best_model = model

    logger.info(f"Best Parameters: {best_params}")
    logger.info("Evaluation with best model:")
    y_pred = best_model.predict(X_test_vect)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
