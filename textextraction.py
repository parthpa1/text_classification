import os
import re
import logging
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import tempfile

# Import libraries with conditional checks
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except ImportError:
    pdfminer_extract_text = None

try:
    import textract
except ImportError:
    textract = None

try:
    import pytesseract
    from PIL import Image
    import pdf2image
except ImportError:
    pytesseract = None
    Image = None
    pdf2image = None

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    def __init__(self, max_workers=10, timeout=30):
        """
        Initialize the PDF Processor with multiple extraction methods
        Args:
            max_workers (int): Maximum number of concurrent workers
            timeout (int): Timeout for requests in seconds
        """
        self.max_workers = max_workers
        self.timeout = timeout

        # Check available methods
        self.extraction_methods = [
            method for method in [
                self._extract_with_pypdf2,
                self._extract_with_pdfminer,
                self._extract_with_textract,
                self._extract_with_ocr
            ] if method is not None
        ]

        if not self.extraction_methods:
            raise RuntimeError("No PDF text extraction libraries are available. Install one to proceed.")

    def _download_pdf(self, url):
        try:
            response = requests.get(url, timeout=self.timeout, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                return response.content
            logger.warning(f"Failed to download PDF from {url}. Status code: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return None

    def _clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        text = ''.join(char for char in text if char.isprintable())
        return text

    def _extract_with_pypdf2(self, pdf_content):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            full_text = "".join(page.extract_text() for page in pdf_reader.pages)
            return self._clean_text(full_text)
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return ""

    def _extract_with_pdfminer(self, pdf_content):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_content)
                temp_pdf.close()
                text = pdfminer_extract_text(temp_pdf.name)
                os.unlink(temp_pdf.name)
                return self._clean_text(text)
        except Exception as e:
            logger.warning(f"PDFMiner extraction failed: {e}")
            return ""

    def _extract_with_textract(self, pdf_content):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(pdf_content)
                temp_pdf.close()
                text = textract.process(temp_pdf.name).decode('utf-8')
                os.unlink(temp_pdf.name)
                return self._clean_text(text)
        except Exception as e:
            logger.warning(f"Textract extraction failed: {e}")
            return ""

    def _extract_with_ocr(self, pdf_content):
        try:
            images = pdf2image.convert_from_bytes(pdf_content)
            full_text = "".join(pytesseract.image_to_string(image) for image in images)
            return self._clean_text(full_text)
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""

    def extract_pdf_text(self, url):
        pdf_content = self._download_pdf(url)
        if not pdf_content:
            return ""
        for method in self.extraction_methods:
            text = method(pdf_content)
            if text:
                return text
        logger.error(f"Failed to extract text from {url}")
        return ""

    def process_excel(self, excel_path, output_path=None):
        df = pd.read_excel(excel_path)
        logger.info(f"Processing {len(df)} datasheet links")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.extract_pdf_text, row['datasheet_link']): idx
                for idx, row in df.iterrows()
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    df.at[idx, 'datasheet_text'] = future.result()
                except Exception as exc:
                    logger.error(f"Failed to process index {idx}: {exc}")
        if output_path:
            df.to_excel(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
        return df


def main():
    processor = PDFProcessor(max_workers=10, timeout=45)
    train_data = processor.process_excel('train_data.xlsx', 'text_train_data.xlsx')
    test_data = processor.process_excel('test_data.xlsx', 'text_test_data.xlsx')


if __name__ == "__main__":
    main()