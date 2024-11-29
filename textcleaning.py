import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('words')

class TextProcessor:
    def __init__(self, train_path=None, test_path=None):
        """
        Initialize the TextProcessor with file paths for train and test datasets.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.english_words = set(w.lower() for w in nltk.corpus.words.words())

    def load_data(self, path, data_type='train'):
        """
        Load data from an Excel file and return it as a DataFrame.
        """
        data = pd.read_excel(path)
        data = data.dropna().reset_index(drop=True)
        if data_type == 'train':
            self.train_data = data
        elif data_type == 'test':
            self.test_data = data
        return data

    def clean_text(self, text):
        """
        Clean the input text by removing special characters, numbers, stop words, and more.
        """
        # Remove special characters and numbers
        text = re.sub('[^\w\s]', '', text)
        text = re.sub('\d', '', text)

        # Convert to lowercase and remove extra whitespace
        text = text.lower()
        text = re.sub('\s+', ' ', text).strip()

        # Remove URLs, HTML tags, and email addresses
        text = re.sub('<.*?>', '', text)
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub('\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove non-English words
        words = text.split()
        text = ' '.join(word for word in words if word in self.english_words)

        # Remove stop words and apply lemmatization
        text = ' '.join(self.lemmatizer.lemmatize(word) for word in text.split() if word not in self.stop_words)
        return text

    def process_data(self, data, text_column):
        """
        Apply text cleaning to the specified column in the dataset.
        """
        data[f'cleaned_{text_column}'] = data[text_column].apply(self.clean_text)
        return data

    def save_cleaned_data(self, data, filename):
        """
        Save the cleaned DataFrame to an Excel file.
        """
        data.to_excel(filename, index=False)
        print(f"Saved cleaned data to {filename}")


# Example usage
if __name__ == "__main__":
    # Initialize processor with paths for train and test data
    processor = TextProcessor(train_path='text_train_data.xlsx', test_path='text_test_data.xlsx')

    # Load and process train data
    train_data = processor.load_data(processor.train_path, data_type='train')
    processed_train_data = processor.process_data(train_data, 'datasheet_text')
    processor.save_cleaned_data(processed_train_data, 'cleantext_train_data.xlsx')

    # Load and process test data
    test_data = processor.load_data(processor.test_path, data_type='test')
    processed_test_data = processor.process_data(test_data, 'datasheet_text')
    processor.save_cleaned_data(processed_test_data, 'cleantext_test_data.xlsx')

