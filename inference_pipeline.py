import joblib
from textextraction import PDFProcessor
from textcleaning import TextProcessor
from sklearn.preprocessing import LabelEncoder

class InferencePipeline:
    def __init__(self, model_path, label_encoder_path):
        self.pdf_processor = PDFProcessor(max_workers=10, timeout=45)
        self.text_processor = TextProcessor()
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def extract_text_from_pdf(self, pdf_url):
        pdf_content = self.pdf_processor._download_pdf(pdf_url)
        if not pdf_content:
            return None
        for method in self.pdf_processor.extraction_methods:
            text = method(pdf_content)
            if text:
                return text
        return None

    def clean_text(self, text):
        return self.text_processor.clean_text(text)

    def predict(self, pdf_url):
        # Extract text from PDF
        raw_text = self.extract_text_from_pdf(pdf_url)
        if not raw_text:
            return "Failed to extract text from the PDF."

        # Clean the extracted text
        cleaned_text = self.clean_text(raw_text)

        # Predict the label and class probabilities
        tfidf_vector = self.model.named_steps['tfidf'].transform([cleaned_text])
        probabilities = self.model.named_steps['nb'].predict_proba(tfidf_vector)[0]
        predicted_label_index = probabilities.argmax()
        predicted_label = self.label_encoder.inverse_transform([predicted_label_index])[0]

        return predicted_label, probabilities

# Example usage
if __name__ == "__main__":
    model_path = 'best_text_classifier_model.joblib'
    label_encoder_path = 'label_encoder.joblib'
    pipeline = InferencePipeline(model_path, label_encoder_path)
