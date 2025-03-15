from enum import Enum
from pydantic_ai import Agent, BinaryContent
import os
from pydantic import BaseModel, Field, field_validator
import io
from datetime import date

# For PDF processing
from pdf2image import convert_from_path
from docx2pdf import convert

# For DOCX processing
import pandas as pd  # Import pandas
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# import json


def get_highest_degree(education_list):
    """
    Determines the highest degree from a list of education data.

    Args:
        education_list: A list of education_Data objects.

    Returns:
        A tuple: (highest_degree_name, school_name) or (None, None) if no degree is found.
    """
    degree_hierarchy = {
        "phd": 5,
        "doctorate": 4,
        "master's": 3,
        "bachelor's": 2,
        "high school": 1,
    }
    highest_degree = None
    highest_degree_name = None
    school_name = None
    school_type = None
    for edu in education_list:
        if edu["degreeType"] is not None:
            degree_level = degree_hierarchy.get(
                edu["degreeType"].value, 0
            )  # Default to 0 if not found
            if highest_degree is None or degree_level > highest_degree:
                highest_degree = degree_level
                highest_degree_name = edu["degreeName"]
                school_name = edu["school_name"]
                school_type = edu["degreeType"]

    return highest_degree_name or "", school_name or "", school_type or ""


def post_data(data, classification_type, document_path):
    """
    Post-process the CV data to validate and correct fields.

    Args:
        data (dict): The CV data extracted from the document
        classification_type (dict): The classification of the document

    Returns:
        dict: The processed CV data
    """
    # Make a copy of the data to avoid modifying the original
    processed_data = data.copy()

    # Validate age
    if processed_data["age"] == 0:
        # Try to estimate age from education first
        if processed_data["education"] and len(processed_data["education"]) > 0:
            # Find the earliest education start date, ignoring None values
            edu_start_dates = [
                edu["start_date"]
                for edu in processed_data["education"]
                if edu.get("start_date")
            ]
            if edu_start_dates:
                earliest_edu_start = min(edu_start_dates)
                # Assuming education starts at age 18
                current_year = date.today().year
                estimated_year_of_birth = earliest_edu_start.year - 18
                processed_data["age"] = current_year - estimated_year_of_birth

        # If still no age and has job data, try to estimate from job
        if (
            processed_data["age"] == 0
            and processed_data["jobs"]
            and len(processed_data["jobs"]) > 0
        ):
            # Find the earliest job start date, ignoring None values
            job_start_dates = [
                job["start_date"]
                for job in processed_data["jobs"]
                if job.get("start_date")
            ]
            if job_start_dates:
                earliest_job_start = min(job_start_dates)
                # Assuming job starts at age 12 (as per requirement)
                current_year = date.today().year
                estimated_year_of_birth = earliest_job_start.year - 22
                processed_data["age"] = current_year - estimated_year_of_birth

    # Calculate years of experience, considering months
    years_of_experience = 0
    total_months_in_jobs = 0  # Keep track of total months for averaging
    num_jobs = len(processed_data["jobs"])

    employers_names = []
    fields_of_experience = []
    for i, job in enumerate(processed_data["jobs"]):
        # Convert start_date string to date object
        start_date_obj = date.fromisoformat(str(job["start_date"]))

        if job["end_date"] is not None:
            # Convert end_date string to date object
            end_date_obj = date.fromisoformat(str(job["end_date"]))
        else:
            end_date_obj = date.today()

        # Special case for the first job
        if i == 0 and start_date_obj == end_date_obj:
            end_date_obj = date.today()

        # Calculate the difference in months
        months_of_experience = (end_date_obj.year - start_date_obj.year) * 12 + (
            end_date_obj.month - start_date_obj.month
        )
        years_of_experience += months_of_experience
        total_months_in_jobs += months_of_experience  # Add to total
        employers_names.append(job["employer_name"])
        if job["field_of_this_job"]:
            fields_of_experience.append(job["field_of_this_job"].lower())

    # Convert total months to years
    years_of_experience = years_of_experience / 12.0
    # Calculate average time in each job (in months)
    if num_jobs > 0:  # Avoid division by zero
        processed_data["avg_time_in_each_job"] = total_months_in_jobs / num_jobs / 12
    else:
        processed_data["avg_time_in_each_job"] = 0  # Or some other default value

    processed_data["employers_names"] = list(set(employers_names))
    processed_data["fields_of_experience"] = list(set(fields_of_experience))
    processed_data["years_of_experience"] = years_of_experience
    processed_data["is_educated"] = (
        len(processed_data["education"]) > 0
    )  # Simple check for any education

    processed_data["document_path"] = document_path.split("/")[-1]
    processed_data["job_type"] = classification_type["type"]
    processed_data["job_confidence"] = classification_type["confidence"]
    # Get highest degree and school name
    highest_degree, school_name, school_type = get_highest_degree(
        processed_data["education"]
    )
    processed_data["highest_degree"] = highest_degree
    processed_data["highest_degree_school"] = school_name
    processed_data["highest_degree_school_type"] = school_type

    # Remove 'jobs' and 'education'
    del processed_data["jobs"]
    del processed_data["education"]

    return processed_data


class WorkType(str, Enum):
    ADMINISTRATIVE_ASSISTANT = "administrative assistant"
    RESTAURANT_MANAGER = "restaurant manager"


class DegreeType(str, Enum):
    BACHELORS = "bachelor's"
    MASTERS = "master's"
    PHD = "phd"
    DOCTORATE = "doctorate"
    HIGH_SCHOOL = "high school"


class classification_model(BaseModel):
    type: WorkType = Field(
        description="The type of work being described in the document"
    )
    confidence: float = Field(description="The confidence score of the classification")


class job_Data(BaseModel):
    employer_name: str = Field(description="The name of the employer")
    position_title: str = Field(description="The title of the position")
    start_date: date = Field(description="The start date of the job")
    end_date: date = Field(description="The end date of the job")
    field_of_this_job: str = Field(description="The field of the job")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_dates(cls, value):
        if isinstance(value, str):
            return date.fromisoformat(str(value))
        return value

    @field_validator("employer_name", mode="before")
    @classmethod
    def parse_employer_name(cls, value):
        try:
            float(value)
            return ""
        except:
            return value


class education_Data(BaseModel):
    school_name: str = Field(description="The name of the school")
    degreeType: DegreeType = Field(description="The degree of the education")
    degreeName: str = Field(description="The name of the degree")
    start_date: date = Field(description="The start date of the education")
    end_date: date = Field(description="The end date of the education")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def parse_dates(cls, value):
        if isinstance(value, str):
            return date.fromisoformat(str(value))
        return value

    @field_validator("school_name", mode="before")
    @classmethod
    def parse_school_name(cls, value):
        try:
            float(value)
            return ""
        except:
            return value


class cv_Data(BaseModel):
    age: int = Field(description="The age of the person")
    jobs: list[job_Data] = Field(description="The jobs of the person")
    education: list[education_Data] = Field(description="The education of the person")
    current_address_area: str = Field(description="The current address of the person")
    contact_number: str = Field(description="The contact number of the person")
    name: str = Field(description="The name of the person")


class Processor:
    def __init__(
        self, model="gemini-2.0-flash", result_type=None, classification_type=None
    ):
        """
        Initialize the document processor.

        Args:
            model (str): The model name to use for PydanticAI
            result_type: The expected result type (Pydantic model)
        """
        self.model = model
        self.result_type = result_type
        self.classification_type = classification_type

    async def process_document(self, document_path, prompt):
        """
        Process a document (PDF or DOCX) with PydanticAI.

        Args:
            document_path (str): Path to the document
            prompt (str): Prompt to send to the model

        Returns:
            Results from processing all pages at once
        """
        file_ext = document_path.lower().split(".")[-1]

        # Convert document to images based on file type
        if file_ext == "pdf":
            images = self._convert_pdf_to_images(document_path)
        elif file_ext in ["doc", "docx"]:
            images = self._convert_doc_to_images(document_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        print(f"Processing document with {len(images)} pages")

        # Process all images at once
        data, type = await self._process_images(images, prompt)
        return data, type

    def _convert_pdf_to_images(self, pdf_path):
        """Convert a PDF file to a list of PIL Images"""
        return convert_from_path(pdf_path)

    def _convert_doc_to_images(self, doc_path):
        """Convert a DOCX file to a list of PIL Images"""
        convert(doc_path, "output.pdf")
        return convert_from_path("output.pdf")

    async def _process_images(self, images, prompt):
        """Process all images at once with PydanticAI"""
        # Convert all PIL images to binary content
        binary_contents = []
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            binary_contents.append(
                BinaryContent(data=img_byte_arr, media_type="image/png")
            )

        # Prepare the payload with prompt and all images
        payload = [prompt]
        payload.extend(binary_contents)

        # Run the agent with all images
        data = await self._agent(payload, self.result_type)

        payload = [
            "Based on the document, classify the most probable position of the person in the document."
        ]
        payload.extend(binary_contents)
        type = await self._agent(payload, self.classification_type)

        return data, type

    async def _agent(self, payload, result_type):
        agent = Agent(model=self.model, result_type=result_type)
        result = await agent.run(payload)
        return result.data.model_dump()

    async def _process_single_file(
        self,
        filename,
        folder_path,
        csv_file_path,
        prompt,
        processed_files,
        csv_lock,
        file_exists_ref,
    ):
        """Process a single file asynchronously"""
        if filename in processed_files:
            print(f"Skipping already processed file: {filename}")
            return

        document_path = os.path.join(folder_path, filename)
        print(f"Processing: {document_path}")

        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                data, classification_type = await self.process_document(
                    document_path, prompt
                )
                processed_data = post_data(data, classification_type, document_path)

                # Ensure all expected fields are present
                required_fields = [
                    "document_path",
                    "name",
                    "age",
                    "current_address_area",
                    "contact_number",
                    "years_of_experience",
                    "avg_time_in_each_job",
                    "employers_names",
                    "fields_of_experience",
                    "is_educated",
                    "highest_degree",
                    "highest_degree_school",
                    "highest_degree_school_type",
                    "job_type",
                    "job_confidence",
                ]

                for field in required_fields:
                    if field not in processed_data:
                        processed_data[field] = ""  # Add empty value for missing fields

                df = pd.DataFrame([processed_data])

                # Use a lock to safely write to the CSV
                async with csv_lock:
                    try:
                        if not file_exists_ref[0] or not os.path.exists(csv_file_path):
                            df.to_csv(
                                csv_file_path,
                                mode="w",  # Write mode for the first file
                                header=True,
                                index=False,
                                encoding="utf-8",
                            )
                        else:
                            df.to_csv(
                                csv_file_path,
                                mode="a",  # Append mode for subsequent files
                                header=False,
                                index=False,
                                encoding="utf-8",
                            )
                        file_exists_ref[0] = True
                        print(f"Successfully saved data for: {filename}")
                    except Exception as csv_error:
                        print(f"Error saving CSV for {filename}: {str(csv_error)}")
                        with open("csv_errors.log", "a") as f:
                            f.write(
                                f"CSV save error for {filename}: {str(csv_error)}\n"
                            )
                            f.write(f"Data: {str(processed_data)}\n\n")
                break  # Exit retry loop on success

            except Exception as e:
                if "503" in str(e):
                    retry_count += 1
                    print(f"Model overloaded, waiting {10 * retry_count} seconds...")
                    await asyncio.sleep(10 * retry_count)
                else:
                    print(f"Error processing {filename}: {str(e)}")
                    with open("errors.log", "a") as f:
                        f.write(f"Failed {filename}: {str(e)}\n")
                    break

    async def process_folder(self, folder_path, csv_file_path, prompt):
        """
        Processes all PDF and DOCX files in a folder concurrently, with up to 4 files at a time.
        Appends results to a CSV and retries with a delay on 503 errors.
        """
        # Get set of already processed files
        processed_files = set()
        if os.path.isfile(csv_file_path):
            try:
                existing_df = pd.read_csv(csv_file_path)
                if "document_path" in existing_df.columns:
                    processed_files = set(
                        existing_df["document_path"].astype(str).values
                    )
                    print(f"Found {len(processed_files)} already processed files")
                else:
                    print("Warning: CSV exists but doesn't have document_path column")
            except (pd.errors.EmptyDataError, KeyError) as e:
                print(f"Error reading existing CSV: {str(e)}")
            except Exception as e:
                print(f"Unexpected error reading CSV: {str(e)}")

        file_exists_ref = [
            os.path.isfile(csv_file_path) and os.path.getsize(csv_file_path) > 0
        ]
        csv_lock = asyncio.Lock()  # Lock for CSV file access

        # Find all valid files
        files_to_process = [
            filename
            for filename in os.listdir(folder_path)
            if filename.lower().endswith((".pdf", ".doc", ".docx"))
            and filename not in processed_files
        ]

        print(f"Found {len(files_to_process)} files to process")

        if not files_to_process:
            print("No new files to process")
            return

        # Process files with concurrency limit of 4
        semaphore = asyncio.Semaphore(4)

        async def process_with_semaphore(filename):
            async with semaphore:
                await self._process_single_file(
                    filename,
                    folder_path,
                    csv_file_path,
                    prompt,
                    processed_files,
                    csv_lock,
                    file_exists_ref,
                )

        # Create tasks for all files
        tasks = [process_with_semaphore(filename) for filename in files_to_process]

        # Run all tasks concurrently
        await asyncio.gather(*tasks)

        print(f"Completed processing {len(files_to_process)} files")


# Example usage:
if __name__ == "__main__":

    # Initialize processor
    processor = Processor(
        model="gemini-2.0-flash",
        result_type=cv_Data,
        classification_type=classification_model,
    )

    # Define prompt
    prompt = f"""You are a professional CV analyzer that excels in extracting data from resumes and CVs.
    You are given a document of a person's CV.
    Analyze this document and extract the data of the person in the document.
    you know that people may write in a bad format, so you should be able to handle it.
    if you could not recognize a field, leave it EMPTY.
    If age is not mentioned, put it 0.
    the data should be in the following FORMAT:
    {cv_Data.model_json_schema()}
    """

    # Process folder asynchronously
    folder_path = "./cvs"  # Replace with the actual path to your folder
    csv_file_path = "cv_data.csv"

    # Run the async function
    asyncio.run(processor.process_folder(folder_path, csv_file_path, prompt))
