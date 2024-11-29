import pandas as pd
import numpy as np
import requests


class DataDownloader:
    """Handles downloading files from the internet."""

    @staticmethod
    def download_file(url, output_file):
        """
        Downloads a file from the given URL and saves it locally.

        :param url: URL of the file to download
        :param output_file: Local path where the file will be saved
        """
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, "wb") as file:
                file.write(response.content)
            print(f"File downloaded successfully as {output_file}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")


class DataProcessor:
    """Handles data loading, cleaning, and saving."""

    def __init__(self, file_path, sheet_name, output_file):
        """
        Initialize the DataProcessor with file path, sheet name, and output file.

        :param file_path: Path to the Excel file
        :param sheet_name: Name of the sheet to load
        :param output_file: Path for saving the processed file
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.output_file = output_file
        self.df = None

    def load_data(self):
        """Loads the data from the specified Excel sheet."""
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            print(f"Loaded data with {len(self.df)} rows from sheet '{self.sheet_name}'.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def clean_datasheet_links(self):
        """Cleans and processes the 'datasheet_link' column."""
        if 'datasheet_link' in self.df.columns:
            self.df['datasheet_link'] = self.df['datasheet_link'].replace('-', pd.NA)
            self.df.dropna(subset=['datasheet_link'], inplace=True)
            print(f"After removing rows with missing 'datasheet_link': {len(self.df)} rows.")

            duplicate_count = self.df['datasheet_link'].duplicated().sum()
            print(f"Duplicate datasheet links found: {duplicate_count}")

            self.df.drop_duplicates(subset=['datasheet_link'], inplace=True)
            print(f"After removing duplicates: {len(self.df)} rows.")
        else:
            print("Column 'datasheet_link' not found in the data.")

    def update_links(self):
        """Updates 'datasheet_link' to include 'http:' for specific URLs."""
        if 'datasheet_link' in self.df.columns:
            missing_http_count = len(
                self.df[self.df['datasheet_link'].str.startswith('//mm.digikey.com/', na=False)]
            )
            self.df.loc[
                self.df['datasheet_link'].str.startswith('//mm.digikey.com/', na=False),
                'datasheet_link'
            ] = 'http:' + self.df['datasheet_link']
            print(f"Updated {missing_http_count} rows to include 'http:' in 'datasheet_link'.")
        else:
            print("Column 'datasheet_link' not found in the data.")

    def save_data(self):
        """Saves the processed DataFrame to an Excel file."""
        try:
            self.df.to_excel(self.output_file, index=False)
            print(f"Processed data saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving data: {e}")

    def process(self):
        """Executes the complete processing workflow."""
        self.load_data()
        self.clean_datasheet_links()
        self.update_links()
        self.save_data()


if __name__ == "__main__":
    # URL and file download
    download_url = "https://docs.google.com/spreadsheets/d/1oZCKUqLmKV8bGlmcjjbJby-u8lHMgDQm/export?format=xlsx"
    local_file = "DataSet.xlsx"
    DataDownloader.download_file(download_url, local_file)

    # Process train data
    print("\nProcessing train data...")
    train_processor = DataProcessor(local_file, 'train_data', 'train_data.xlsx')
    train_processor.process()

    print("=" * 50)

    # Process test data
    print("\nProcessing test data...")
    test_processor = DataProcessor(local_file, 'test_data', 'test_data.xlsx')
    test_processor.process()