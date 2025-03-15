from enum import Enum
from itertools import tee
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
        if processed_data["date_of_birth"] is not None:
            # More accurate age calculation that accounts for month and day
            today = date.today()
            born = date.fromisoformat(str(processed_data["date_of_birth"]))
            processed_data["age"] = today.year - born.year
        else:
            # Try to estimate age from education first
            if processed_data["education"] and len(processed_data["education"]) > 0:
                # Find the earliest education start date, ignoring None values
                edu_start_dates = [
                    date.fromisoformat(str(edu["start_date"]))
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
                        date.fromisoformat(str(job["start_date"]))
                        for job in processed_data["jobs"]
                        if job.get("start_date")
                    ]
                    if job_start_dates:
                        earliest_job_start = min(job_start_dates)
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
        if job["start_date"] is not None:
            start_date_obj = date.fromisoformat(str(job["start_date"]))
        else:
            continue

        if job["end_date"] is not None:
            # Convert end_date string to date object
            end_date_obj = date.fromisoformat(str(job["end_date"]))
        else:
            continue

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
    start_date: date | None = Field(description="The start date of the job")
    end_date: date | None = Field(description="The end date of the job")
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
    start_date: date | None = Field(description="The start date of the education")
    end_date: date | None = Field(description="The end date of the education")

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
    date_of_birth: date | None = Field(description="The date of birth of the person")
    jobs: list[job_Data] = Field(description="The jobs of the person")
    education: list[education_Data] = Field(description="The education of the person")
    current_address_area: str = Field(description="The current address of the person")
    contact_number: str = Field(description="The contact number of the person")
    name: str = Field(description="The name of the person")

    @field_validator("date_of_birth", mode="before")
    @classmethod
    def parse_date_of_birth(cls, value):
        if isinstance(value, str):
            return date.fromisoformat(str(value))
        return value


class VisualCVFilter(BaseModel):
    cv_quality_score: float = Field(
        ...,
        description="Score between 0 and 1, it represents the quality of the CV structure relative to the ATS format",
    )


class Processor:

    def __init__(
        self,
        model="gemini-2.0-flash",
        result_type=cv_Data,
        classification_type=classification_model,
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
        self.prompt = f"""You are a highly intelligent CV analyzer that extracts structured data from resumes and CVs, even when the formatting is inconsistent or unstructured. Your task is to analyze the document and extract all relevant details while following these rules:
        
General Guidelines:
- Handle poor formatting and unstructured text intelligently.
- Extract only meaningful and relevant details.
- If a field is unrecognizable or missing, leave it EMPTY.
- If age is not explicitly mentioned, set it to 0.
- For employer names, extract only the company name, excluding location or additional details.
- Standardize extracted data where possible (e.g., formatting phone numbers, normalizing dates).
- If the date of birth is not explicitly mentioned, set it to None.
The data should be in the following FORMAT:
        {cv_Data.model_json_schema()}
        """

    def process_document(self, document_path):
        """
        Process a document (PDF, DOCX, or image) with PydanticAI.

        Args:
            document_path (str): Path to the document

        Returns:
            Results from processing all pages at once
        """
        file_ext = document_path.lower().split(".")[-1]

        # Convert document to images based on file type
        if file_ext == "pdf":
            images = self._convert_pdf_to_images(document_path)
        elif file_ext in ["doc", "docx"]:
            images = self._convert_doc_to_images(document_path)
        elif file_ext in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif"]:
            # Process image files directly
            from PIL import Image

            print("processing image file")
            images = [Image.open(document_path)]
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        print(f"Processing document with {len(images)} pages")

        # Process all images at once
        data, type = self._process_images(images)
        return data, type

    def _convert_pdf_to_images(self, pdf_path):
        """Convert a PDF file to a list of PIL Images"""
        return convert_from_path(pdf_path)

    def _convert_doc_to_images(self, doc_path):
        """Convert a DOCX file to a list of PIL Images"""
        convert(doc_path, "output.pdf")
        return convert_from_path("output.pdf")

    def _process_images(self, images):
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
        # New prompt to aid the agent in evaluating the visual quality of a CV
        visual_filter_prompt = """
       You are an AI-powered evaluator for CV documents, assessing both **visual presentation** and **content quality** based on industry standards.  

### **Assessment Criteria:**  
1. **Visual Quality (50%)** – Analyze the CV's formatting and structure:  
   - **Clarity & Readability**: Is the text legible? Are fonts professional and consistent?  
   - **Layout & Organization**: Is the document well-structured with clear sections (e.g., Summary, Experience, Education, Skills)?  
   - **Visual Noise**: Penalize excessive decorative elements, clutter, misalignment, poor scanning quality, or unnecessary colors.  
   - **Use of Space**: Are margins, spacing, and alignment optimized for readability?  

2. **Content Quality (50%)** – Assess the depth and relevance of the provided information:  
   - **Completeness**: Does the CV contain all essential sections? (Summary, Work Experience, Skills, Education, etc.)  
   - **Detail & Specificity**: Are job descriptions detailed with quantifiable achievements? Avoid vague descriptions.  
   - **Keyword Relevance**: Are industry-relevant keywords and skills present?  
   - **Conciseness**: Is the content well-written, avoiding unnecessary fluff or redundancy?  

### **Scoring System:**  
Return a JSON object with two keys:  
- `"cv_quality_score"`: The final **weighted score** (average of visual and content scores).  
"""
        visual_filter_payload = [visual_filter_prompt]
        visual_filter_payload.extend(binary_contents)
        visual_filter = self._agent(visual_filter_payload, VisualCVFilter)
        print(f"CV quality score: {visual_filter['cv_quality_score']}")
        if visual_filter["cv_quality_score"] < 0.5:
            print("CV is visually bad, skipping...")
            return None, None
        print(f"Extracting CV data from {len(images)} page(s)...")
        payload = [self.prompt]
        payload.extend(binary_contents)

        # Run the agent with all images
        data = self._agent(payload, self.result_type)
        print(f"Successfully extracted CV data for {data.get('name', 'Unknown')}")

        print("Classifying document type...")
        payload = [
            "Based on the document, classify the most probable position of the person in the document."
        ]
        payload.extend(binary_contents)
        type = self._agent(payload, self.classification_type)
        print(
            f"Document classified as: {type.get('type', 'Unknown')} (confidence: {type.get('confidence', 0):.2f})"
        )

        return data, type

    def _agent(self, payload, result_type):
        agent = Agent(model=self.model, result_type=result_type)
        return agent.run_sync(payload).data.model_dump()

    def process_folder(self, folder_path, csv_file_path):
        """
        Processes all PDF and DOCX files in a folder, appends results to a CSV,
        and retries with a delay on 503 errors.
        """
        fieldnames = [
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

        # Get set of already processed files
        processed_files = set()
        if os.path.isfile(csv_file_path):
            try:
                existing_df = pd.read_csv(csv_file_path)
                processed_files = set(existing_df["document_path"].astype(str).values)
                print(
                    f"Found {len(processed_files)} already processed files in {csv_file_path}"
                )
            except (pd.errors.EmptyDataError, KeyError):
                print(
                    f"CSV file {csv_file_path} exists but is empty or has invalid format"
                )
                pass

        file_exists = os.path.isfile(csv_file_path)

        # Count total files to process
        total_files = sum(
            1
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(
                (".pdf", ".doc", ".docx", ".jpg", ".png", ".jpeg")
            )
        )
        skipped_files = 0
        processed_files_count = 0
        error_files = 0

        print(f"Starting to process {total_files} documents from {folder_path}")

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(
                (".pdf", ".doc", ".docx", ".jpg", ".png", ".jpeg")
            ):
                # Skip already processed files
                if filename in processed_files:
                    print(f"Skipping already processed file: {filename}")
                    skipped_files += 1
                    continue

                document_path = os.path.join(folder_path, filename)
                print(
                    f"\n[{processed_files_count + 1}/{total_files - skipped_files}] Processing: {filename}"
                )

                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        data, classification_type = self.process_document(document_path)
                        if data is None and classification_type is None:
                            print(f"Skipping {filename} due to visual quality issues")
                            skipped_files += 1
                            # Move the file to the skipped folder
                            skipped_folder = os.path.join(folder_path, "skipped")
                            os.makedirs(skipped_folder, exist_ok=True)
                            new_file_path = os.path.join(skipped_folder, filename)
                            os.rename(document_path, new_file_path)
                            continue
                        processed_data = post_data(
                            data, classification_type, document_path
                        )
                        df = pd.DataFrame([processed_data])
                        df.to_csv(
                            csv_file_path,
                            mode="a",
                            header=not file_exists,
                            index=False,
                            encoding="utf-8",
                        )
                        file_exists = True
                        processed_files_count += 1
                        print(f"✓ Successfully processed and saved data for {filename}")
                        break  # Exit retry loop on success

                    except Exception as e:
                        if "503" in str(e):
                            retry_count += 1
                            print(
                                f"⚠ Model overloaded (attempt {retry_count}/{max_retries}), waiting {10 * retry_count} seconds..."
                            )
                            time.sleep(10 * retry_count)
                        else:
                            error_message = f"✗ Failed to process {filename}: {str(e)}"
                            print(error_message)
                            with open("errors.log", "a") as f:
                                f.write(f"{error_message}\n")
                            error_files += 1
                            break
            elif filename.lower().endswith((".jpg", ".png", ".jpeg", ".gif")):
                print(f"Skipping image file: {filename}")
                skipped_files += 1
                continue
        print(f"\nProcessing complete! Summary:")
        print(f"- Total files: {total_files}")
        print(f"- Successfully processed: {processed_files_count}")
        print(f"- Skipped (already processed): {skipped_files}")
        print(f"- Failed: {error_files}")
        print(f"- Results saved to: {csv_file_path}")


# Example usage:
if __name__ == "__main__":
    # Set your API key

    print("CV Processor - Starting up...")
    # Initialize processor
    processor = Processor(
        model="gemini-2.0-pro-exp-02-05",
    )
    folder_path = "./cvs"  # Replace with the actual path to your folder
    csv_file_path = "cv_data.csv"
    print(f"Using model: gemini-2.0-flash")
    print(f"Processing documents from: {folder_path}")
    print(f"Saving results to: {csv_file_path}")
    processor.process_folder(folder_path, csv_file_path)
