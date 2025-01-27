import cv2
from PIL import Image
import pytesseract
import re
import spacy
from spacy.lang.en import English
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# loading spacy models
nlp = spacy.load("en_core_web_lg")

# download nltk packages
# nltk.download('punkt')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 6
        )
        
        preprocessed_image_path = "preprocessed_image.jpg"
        cv2.imwrite(preprocessed_image_path, adaptive_thresh)
        return preprocessed_image_path
    except Exception as e:
        raise RuntimeError(f"Image preprocessing failed: {e}")

def extract_name_from_text(text):
    try:
        aadhaar_keywords = r'(To|Name|Name of Holder|Name Of Holder|Holder\'s Name)[:|\s]+([A-Za-z]+[\s]*[A-Za-z]*)+'
        match = re.search(aadhaar_keywords, text, re.IGNORECASE)
        if match:
            candidate = text[match.end():].split('\n')[0].strip()
            if len(candidate.split()) >= 2:
                return candidate

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        chunks = ne_chunk(tagged)
        # print("chunks", chunks)

        for chunk in chunks:
            if isinstance(chunk, Tree):
                entity = " ".join([word for word, tag in chunk])
                print(entity)
                if "PERSON" in [tag for word, tag in chunk]:
                    if len(entity.split()) >= 2:
                        return entity
        
        # If NLTK extraction fails, fallback to SpaCy NER
        print("Fallback to SpaCy NER")
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if re.match(r'^[A-Za-z]{1,2}\s[A-Za-z]+$', name):
                    return name
                elif re.match(r'^[A-Za-z]+(?:\s+[A-Za-z]+)+$', name):
                    return name

        # If both NLTK and SpaCy fail, check for title-case names
        title_case = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text)
        if title_case:
            return max(title_case, key=len)

        return "Not Found"
    except Exception as e:
        return f"Error in name extraction: {e}"
def extract_aadhar_details(image_path):
    try:
        preprocessed_path = preprocess_image(image_path)
        custom_config = r'--psm 4 --oem 3 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(
            Image.open(preprocessed_path),
            lang='eng',
            config=custom_config
        )
        
        text = re.sub(r'\s+', ' ', text).strip()
        print(text)
        dob = re.search(r'\b(\d{2}/\d{2}/\d{4})\b', text)
        gender = re.search(r'\b(Male|Female|M|F)\b', text, re.I)
        aadhaar_number = re.search(r'\b(\d{4}\s\d{4}\s\d{4})\b', text)
        
        return {
            "Name": extract_name_from_text(text),
            "DOB": dob.group(0) if dob else "Not Found",
            "Gender": (gender.group(0).title() if gender else "Not Found"),
            "Aadhaar Number": aadhaar_number.group(0) if aadhaar_number else "Not Found"
        }
    except Exception as e:
        return {"Error": str(e)}

if __name__ == "__main__":
    details = extract_aadhar_details("image.jpg")
    print("Extracted Aadhaar Details:")
    for k, v in details.items():
        print(f"{k}: {v}")