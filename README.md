# CV Data Summarization

A streamlit application that processes CV documents (PDF, DOC, DOCX) and extracts structured information using Gemini AI. The application analyzes resumes and extracts key information such as personal details, work experience, education history, and more.

## Features

- Process multiple CV documents in various formats (PDF, DOC, DOCX)
- Extract structured data including:
  - Personal information (name, age, date of birth, contact details)
  - Work experience (employer, position, dates, field)
  - Education history (school, degree type, degree name, dates)
- Classify CV by work type
- Save extracted data to CSV for further analysis
- Interactive web interface for easy use

## Requirements

- Python 3.12 or higher
- Gemini API key

## Installation

1. Clone this repository:

   ```
   git clone <repository-url>
   cd resumes-ocr
   ```

2. Install dependencies using uv:

   ```
   uv pip install -r requirements.txt
   ```

   Or install directly from pyproject.toml:

   ```
   uv pip install .
   ```

## Usage

1. Place your CV documents (PDF, DOC, DOCX) in the `cvs/` directory (or specify a different directory in the app)

2. Run the application:

   ```
   streamlit run app.py
   ```

3. In the web interface:
   - Enter your Gemini API key
   - Verify or change the folder path containing CV documents
   - Specify the output CSV file path
   - Click "Process Documents" to start extraction
   - View and download the processed data

## How It Works

The application uses:

- Streamlit for the web interface
- Gemini AI (via pydantic-ai) for document analysis and information extraction
- PDF and DOCX processing libraries to handle different document formats
- Pandas for data management and CSV export

## Project Structure

- `app.py`: Main Streamlit application
- `processor.py`: Core document processing logic with the pydantic-ai agent
- `cvs/`: Default directory for CV documents
- `cv_data.csv`: Default output file for processed data

## Notes

- The application requires a valid Gemini API key to function
- Processing time depends on the number and size of documents
- The application tracks which files have been processed to avoid duplicate work

## Future improvements

- Enhance the streamlit app.
- Edit the async processor to do the same work as the processor for processing multiple documents at the same time.
