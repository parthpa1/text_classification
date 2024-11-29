import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

class TextClassifier:
    def __init__(self, train_data_path, test_data_path):
        # Load data
        self.train_data = pd.read_excel(train_data_path)
        self.test_data = pd.read_excel(test_data_path)

        # Initialize LabelEncoder
        self.label_encoder = LabelEncoder()
        self.train_data['encoded_labels'] = self.label_encoder.fit_transform(self.train_data['target_col'])
        self.test_data['encoded_labels'] = self.label_encoder.transform(self.test_data['target_col'])

        # Initialize TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Initialize Naive Bayes classifier
        self.nb_classifier = MultinomialNB()

    def train_base_model(self):
        # Fit and transform training data with TfidfVectorizer
        X_train_tfidf = self.vectorizer.fit_transform(self.train_data['cleaned_datasheet_text'])
        X_test_tfidf = self.vectorizer.transform(self.test_data['cleaned_datasheet_text'])

        # Train the model
        self.nb_classifier.fit(X_train_tfidf, self.train_data['encoded_labels'])

        # Make predictions on test data
        y_pred = self.nb_classifier.predict(X_test_tfidf)

        # Evaluate the model using F1 score
        f1 = f1_score(self.test_data['encoded_labels'], y_pred, average='weighted')
        print(f"Base Model F1 Score: {f1:.2f}")
        print("Base Model Classification Report:")
        print(classification_report(self.test_data['encoded_labels'], y_pred, target_names=self.label_encoder.classes_))

    def hyperparameter_tuning(self):
        # Create a pipeline with TF-IDF Vectorizer and Naive Bayes
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('nb', MultinomialNB())
        ])

        # Hyperparameter grid
        param_grid = {
            'tfidf__stop_words': [None, 'english'],
            'tfidf__max_df': [0.8, 0.9, 1.0],
            'tfidf__min_df': [1, 3, 5],
            'tfidf__ngram_range': [(1, 1), (1, 2)],

            'nb__alpha': [0.1, 0.5, 1.0, 1.5],
            'nb__fit_prior': [True, False]
        }

        # Perform Grid Search with F1 score
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )

        # Fit Grid Search
        grid_search.fit(self.train_data['cleaned_datasheet_text'], self.train_data['encoded_labels'])

        # Best model
        best_model = grid_search.best_estimator_
        print("\nBest Hyperparameters:", grid_search.best_params_)
        print("Best Cross-Validation F1 Score:", grid_search.best_score_)

        # Save the best model
        joblib.dump(best_model, 'best_text_classifier_model.joblib')
        print("Best model saved as 'best_text_classifier_model.joblib'")

        # Predictions on test data
        y_pred = best_model.predict(self.test_data['cleaned_datasheet_text'])

        # Evaluation
        test_f1 = f1_score(self.test_data['encoded_labels'], y_pred, average='weighted')
        print(f"\nTest Set F1 Score after Hyperparameter Tuning: {test_f1:.2f}")
        print("\nClassification Report after Hyperparameter Tuning:")
        print(classification_report(self.test_data['encoded_labels'], y_pred, target_names=self.label_encoder.classes_))


if __name__ == "__main__":
    train_data_path = 'cleantext_train_data.xlsx'
    test_data_path = 'cleantext_test_data.xlsx'

    # Instantiate the TextClassifier class
    classifier = TextClassifier(train_data_path, test_data_path)

    # Train base model and evaluate performance
    classifier.train_base_model()

    # Perform hyperparameter tuning and evaluate performance
    classifier.hyperparameter_tuning()

