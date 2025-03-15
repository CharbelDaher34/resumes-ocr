import streamlit as st
import pandas as pd
import os
from processor import Processor

st.set_page_config(page_title="CV Processor App", layout="wide")

st.title("CV Document Processor")

# Input fields
with st.form("input_form"):
    api_key = st.text_input(
        "Gemini API Key", type="password", help="Enter your Gemini API key"
    )

    folder_path = st.text_input(
        "Folder Path",
        value="./cvs",
        help="Path to the folder containing CV documents (PDF, DOC, DOCX)",
    )

    csv_file_path = st.text_input(
        "CSV Output Path",
        value="cv_data.csv",
        help="Path where the processed data will be saved as CSV",
    )

    submit_button = st.form_submit_button("Process Documents")

# Process documents when the form is submitted
if submit_button:
    if not api_key:
        st.error("Please enter your Gemini API key")
    elif not folder_path:
        st.error("Please enter a folder path")
    elif not csv_file_path:
        st.error("Please enter a CSV file path")
    else:
        # Set the API key
        os.environ["GEMINI_API_KEY"] = api_key

        # Check if folder exists
        if not os.path.exists(folder_path):
            st.error(f"Folder path does not exist: {folder_path}")
        else:
            # Initialize processor
            with st.spinner("Processing documents..."):
                try:
                    processor = Processor(
                        model="gemini-2.0-flash",
                    )

                    # Create progress bar
                    progress_text = st.empty()
                    progress_bar = st.progress(0)

                    # Count total files to process
                    total_files = len(
                        [
                            f
                            for f in os.listdir(folder_path)
                            if f.lower().endswith((".pdf", ".doc", ".docx"))
                        ]
                    )

                    # Get set of already processed files
                    processed_files = set()
                    if os.path.isfile(csv_file_path):
                        try:
                            existing_df = pd.read_csv(csv_file_path)
                            processed_files = set(
                                existing_df["document_path"].astype(str).values
                            )
                        except (pd.errors.EmptyDataError, KeyError):
                            pass

                    # Adjust total for already processed files
                    files_to_process = total_files - len(
                        [f for f in processed_files if f in os.listdir(folder_path)]
                    )

                    if files_to_process == 0:
                        st.info("All files have already been processed.")
                    else:
                        # Process the folder
                        processor.process_folder(folder_path, csv_file_path)
                        st.success(
                            f"Successfully processed documents and saved to {csv_file_path}"
                        )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

                finally:
                    # Clear progress indicators
                    progress_text.empty()
                    progress_bar.empty()

# Display CSV content if it exists
st.header("Processed CV Data")

if os.path.exists(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path)
        if len(df) > 0:
            # Display dataframe
            st.dataframe(df)

            # Add download button
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=os.path.basename(csv_file_path),
                mime="text/csv",
            )

        else:
            st.info("The CSV file exists but contains no data.")
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
else:
    st.info("No processed data available yet. Process documents to see results here.")
