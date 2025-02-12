# Import necessary libraries
import cv2
from PIL import Image
import pytesseract
import re
import json
import psycopg2
from psycopg2 import sql

# Specify Tesseract path (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Step 1: Preprocess the Image
def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy by converting to grayscale
    and applying thresholding. Also enhance contrast.
    """
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    enhanced_image = cv2.convertScaleAbs(gray_image, alpha=1.5, beta=0)
    
    # Apply thresholding to create a binary image
    _, thresh_image = cv2.threshold(enhanced_image, 150, 255, cv2.THRESH_BINARY)
    
    # Save and return the preprocessed image path
    preprocessed_path = "preprocessed_image.png"
    cv2.imwrite(preprocessed_path, thresh_image)
    return preprocessed_path


# Step 2: Extract Text Using Tesseract OCR
def extract_text(image_path):
    """
    This function takes an image path, preprocesses the image, and extracts
    text from it using Tesseract OCR.
    """
    preprocessed_path = preprocess_image(image_path)
    # Extract text using Tesseract
    text = pytesseract.image_to_string(Image.open(preprocessed_path))
    return text

# Step 3: Extract Key Data Points Using Regular Expressions
def extract_key_data(text):
    """
    This function extracts key data points from the OCR-extracted text
    using regular expressions and assembles a structured dictionary.
    """
    # Extract patient details
    patient_name = re.search(r"Patient\s*Name\s*:\s*(\w+\s\w+)?", text)
    dob = re.search(r"DOB\s*:\s*(\d{2}/\d{2}/\d{4})?", text)
    injection = re.search(r"INJECTION\s*:\s*(YES|NO)", text)
    exercise_therapy = re.search(r"Exercise\s*Therapy\s*:\s*(YES|NO)", text)
    
    # Extract difficulty ratings (example fields)
    bending = re.search(r"Bending\s*or\s*Stooping\s*:\s*(\d)", text)
    putting_on_shoes = re.search(r"Putting\s*on\s*shoes\s*:\s*(\d)", text)
    sleeping = re.search(r"Sleeping\s*:\s*(\d)", text)
    
    # Extract patient changes (example fields)
    since_last_treatment = re.search(r"Patient\s*Changes\s*since\s*last\s*treatment\s*:\s*(.*)", text)
    since_start_of_treatment = re.search(r"Patient\s*changes\s*since\s*the\s*start\s*of\s*treatment\s*:\s*(.*)", text)
    last_3_days = re.search(r"Describe\s*any\s*functional\s*changes\s*within\s*the\s*last\s*three\s*days\s*.*:\s*(.*)", text)
    
    # Extract pain symptoms (example fields)
    pain = re.search(r"Pain\s*:\s*(\d+)", text)
    numbness = re.search(r"Numbness\s*:\s*(\d+)", text)
    tingling = re.search(r"Tingling\s*:\s*(\d+)", text)
    burning = re.search(r"Burning\s*:\s*(\d+)", text)
    tightness = re.search(r"Tightness\s*:\s*(\d+)", text)
    
    # Extract medical assistant data
    blood_pressure = re.search(r"Blood\s*Pressure\s*:\s*([\d/]+)", text)
    hr = re.search(r"HR\s*:\s*(\d+)", text)
    weight = re.search(r"Weight\s*:\s*(\d+)", text)
    height = re.search(r"Height\s*:\s*([\d']+)", text)
    spo2 = re.search(r"SpO2\s*:\s*(\d+)", text)
    temperature = re.search(r"Temperature\s*:\s*(\d+.\d+)", text)
    blood_glucose = re.search(r"Blood\s*Glucose\s*:\s*(\d+)", text)
    respirations = re.search(r"Respitrations\s*:\s*(\d+)", text)
    
    # Fallback if fields are not found
    data = {
        "patient_name": patient_name.group(1) if patient_name else "N/A",
        "dob": dob.group(1) if dob else "N/A",
        "injection": injection.group(1) if injection else "N/A",
        "exercise_therapy": exercise_therapy.group(1) if exercise_therapy else "N/A",
        "difficulty_ratings": {
            "bending": int(bending.group(1)) if bending else 0,
            "putting_on_shoes": int(putting_on_shoes.group(1)) if putting_on_shoes else 0,
            "sleeping": int(sleeping.group(1)) if sleeping else 0
        },
        "patient_changes": {
            "since_last_treatment": since_last_treatment.group(1) if since_last_treatment else "N/A",
            "since_start_of_treatment": since_start_of_treatment.group(1) if since_start_of_treatment else "N/A",
            "last_3_days": last_3_days.group(1) if last_3_days else "N/A"
        },
        "pain_symptoms": {
            "pain": int(pain.group(1)) if pain else 0,
            "numbness": int(numbness.group(1)) if numbness else 0,
            "tingling": int(tingling.group(1)) if tingling else 0,
            "burning": int(burning.group(1)) if burning else 0,
            "tightness": int(tightness.group(1)) if tightness else 0
        },
        "medical_assistant_data": {
            "blood_pressure": blood_pressure.group(1) if blood_pressure else "N/A",
            "hr": int(hr.group(1)) if hr else 0,
            "weight": int(weight.group(1)) if weight else 0,
            "height": height.group(1) if height else "N/A",
            "spo2": int(spo2.group(1)) if spo2 else 0,
            "temperature": float(temperature.group(1)) if temperature else 0.0,
            "blood_glucose": int(blood_glucose.group(1)) if blood_glucose else 0,
            "respirations": int(respirations.group(1)) if respirations else 0
        }
    }
    return data



# Step 4: Convert Data to JSON
def convert_to_json(data):
    """
    Convert the extracted data into a structured JSON format.
    """
    json_data = json.dumps(data, indent=4)
    return json_data

# Step 5: Store JSON Data in PostgreSQL Database
def insert_into_database(json_data):
    """
    Insert the structured JSON data into a PostgreSQL database.
    """
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(
            dbname="ocr_project", user="postgres", password="amartya123", host="localhost"
        )
        cursor = connection.cursor()
        
        # Insert JSON data into the forms_data table
        insert_query = """
        INSERT INTO forms_data (patient_id, form_json)
        VALUES (%s, %s)
        """
        patient_id = 1  # You can dynamically fetch this based on the patient details
        cursor.execute(insert_query, (patient_id, json.dumps(json_data)))
        
        # Commit the transaction
        connection.commit()
        print("Data inserted successfully!")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Close the connection
        cursor.close()
        connection.close()

# Main function to run the entire process
def main():
    # Path to the image file
    image_path = "images/sample_form.png"
    
    # Step 1: Extract text from the image
    raw_text = extract_text(image_path)
    
    # Print the raw extracted text to debug
    print("Extracted Text:\n", raw_text)
    
    # Step 2: Extract key data points from the text
    extracted_data = extract_key_data(raw_text)
    
    # Step 3: Convert the extracted data to JSON format
    json_data = convert_to_json(extracted_data)
    
    # Step 4: Insert the JSON data into the database
    insert_into_database(json_data)
    
if __name__ == "__main__":
    main()
