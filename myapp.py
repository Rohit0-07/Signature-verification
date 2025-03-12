import streamlit as st

st.set_page_config(
    page_title="AI Signature Verification",
    page_icon="✍️",
    layout="wide"
)
    
import cv2
import numpy as np
import pandas as pd
import tempfile
import pickle
import os
import time
from datetime import datetime
from google.cloud import storage
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import json
# ---------------------------
# CONFIGURATION & GCP SETUP
# ---------------------------

# Set your Google Cloud Storage bucket name here
GCP_BUCKET_NAME = 'verifysign'
DATABASE_FILENAME = 'signature_database.pkl'
VERIFICATION_LOG_FILENAME = 'verification_log.csv'

# Load credentials from streamlit secrets or environment variables
def get_credentials_path():
    try:
        if 'gcp_credentials' in st.secrets:
            # Write credentials to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp:
                credentials_dict = st.secrets["gcp_credentials"]
                temp.write(json.dumps(credentials_dict).encode())
                return temp.name
        else:
            st.error("GCP credentials not found in secrets.")
            return None
    except Exception as e:
        st.error(f"Error accessing credentials: {e}")
        return None

# Use the credentials
def get_storage_client():
    credentials_path = get_credentials_path()
    if credentials_path:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        # Now you can create your GCP client
        from google.cloud import storage
        return storage.Client()

# Download the signature database from GCP (if exists) else return empty dict
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_database_from_gcp():
    client = get_storage_client()
    if not client:
        return {}
    
    bucket = client.bucket(GCP_BUCKET_NAME)
    blob = bucket.blob(DATABASE_FILENAME)
    try:
        data = blob.download_as_bytes()
        database = pickle.loads(data)
        st.success("Database loaded successfully from GCP.")
        return database
    except Exception as e:
        st.warning(f"Database not found or could not be loaded: {e}. Starting with an empty database.")
        return {}

# Upload the updated database to GCP
def upload_database_to_gcp(database):
    client = get_storage_client()
    if not client:
        st.error("Could not connect to GCP. Database not updated.")
        return False
    
    try:
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(DATABASE_FILENAME)
        data = pickle.dumps(database)
        blob.upload_from_string(data)
        st.success("Database updated on GCP.")
        return True
    except Exception as e:
        st.error(f"Error uploading database to GCP: {e}")
        return False

# Log verification attempts
def log_verification_attempt(person_id, similarity, verified, image_hash):
    client = get_storage_client()
    if not client:
        return
    
    bucket = client.bucket(GCP_BUCKET_NAME)
    blob = bucket.blob(VERIFICATION_LOG_FILENAME)
    
    log_entry = f"{datetime.now()},{person_id},{similarity:.4f},{verified},{image_hash}\n"
    
    try:
        # Try to append to existing file
        content = ""
        try:
            content = blob.download_as_string().decode('utf-8')
        except:
            # If file doesn't exist, create header
            content = "timestamp,person_id,similarity,verified,image_hash\n"
        
        content += log_entry
        blob.upload_from_string(content)
    except Exception as e:
        st.warning(f"Could not log verification attempt: {e}")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

# Load image from an uploaded file into an OpenCV format
def load_image(file) -> np.ndarray:
    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        file.seek(0)  # Reset file pointer for potential reuse
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Failed to decode image. Please check the file format.")
            return None
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Generate a simple hash of an image for logging
def generate_image_hash(image):
    if image is None:
        return "none"
    # Simple hash based on downsampled image
    small = cv2.resize(image, (32, 32))
    return str(hash(small.tobytes()))

# Preprocess image: convert to grayscale, denoise, and resize
def preprocess_image(image: np.ndarray, width: int = 800) -> np.ndarray:
    if image is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Resize maintaining aspect ratio
    ratio = width / blurred.shape[1]
    dim = (width, int(blurred.shape[0] * ratio))
    resized = cv2.resize(blurred, dim)
    
    # Normalize pixel values to range [0, 1]
    normalized = resized.astype('float32') / 255.0
    
    return normalized

# Extract signature using adaptive thresholding and contour filtering
def extract_signature(processed_image: np.ndarray) -> np.ndarray:
    if processed_image is None:
        return None
    
    # Convert normalized image back to uint8 for thresholding
    image_uint8 = (processed_image * 255).astype(np.uint8)
    
    # Apply adaptive thresholding for better results with varying lighting
    thresh = cv2.adaptiveThreshold(
        image_uint8, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Optional: Apply morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours to find signature-like shapes
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Skip very small contours (noise)
            continue
            
        # Calculate contour perimeter and derive shape complexity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
            
        complexity = area / perimeter
        aspect_ratio = 1.0
        
        # Calculate bounding box and aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0 and h > 0:
            aspect_ratio = float(w) / h
        
        # Signatures typically have a certain complexity and aspect ratio
        if complexity < 50 and 0.1 < aspect_ratio < 10:
            valid_contours.append(contour)
    
    if not valid_contours:
        return None
    
    # Create a mask combining all valid contours
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, valid_contours, -1, 255, -1)
    
    # Find the bounding rectangle of the combined mask
    x, y, w, h = cv2.boundingRect(mask)
    
    # Extract the signature region
    signature_region = image_uint8[y:y+h, x:x+w]
    
    # Return the signature region
    return signature_region

# Check whether the extracted region is likely to be a signature
def check_signature_presence(signature_region: np.ndarray) -> bool:
    if signature_region is None:
        return False
    
    # Check if the region is too small
    height, width = signature_region.shape
    if height * width < 1000:  # Minimum area threshold
        return False
    
    # Check if the region is too empty (percentage of foreground pixels)
    _, binary = cv2.threshold(signature_region, 127, 255, cv2.THRESH_BINARY)
    foreground_ratio = np.sum(binary == 0) / (height * width)
    if foreground_ratio < 0.01:  # Less than 1% foreground pixels
        return False
    
    # Check if the signature has a reasonable aspect ratio
    aspect_ratio = width / height
    if aspect_ratio < 0.2 or aspect_ratio > 5:  # Likely not a signature
        return False
    
    return True

# Extract more robust features from the signature image
def extract_features(signature_region: np.ndarray) -> np.ndarray:
    if signature_region is None:
        return None
    
    # Resize for consistent feature extraction
    signature_region = cv2.resize(signature_region, (100, 100))
    
    # 1. Histogram of Oriented Gradients (HOG) features
    # Calculate gradients
    gx = cv2.Sobel(signature_region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(signature_region, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy)
    
    # Bin the angles into 9 orientation bins
    bins = np.linspace(0, 2*np.pi, 10)
    hist = np.zeros(9)
    for i in range(9):
        mask = (ang >= bins[i]) & (ang < bins[i+1])
        hist[i] = np.sum(mag[mask])
    
    # Normalize
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)
    
    # 2. Intensity histogram (as before)
    intensity_hist = cv2.calcHist([signature_region], [0], None, [32], [0, 256])
    intensity_hist = cv2.normalize(intensity_hist, intensity_hist).flatten()
    
    # 3. Zoning features
    # Divide the image into 4x4 zones and calculate average intensity
    zone_features = []
    h, w = signature_region.shape
    zone_h, zone_w = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            zone = signature_region[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
            zone_features.append(np.mean(zone))
    
    # 4. Combine all features
    all_features = np.concatenate([hist, intensity_hist, zone_features])
    
    return all_features

# Compute weighted cosine similarity
def compute_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
    if feature1 is None or feature2 is None:
        return 0.0
    
    # Ensure features are same length
    min_length = min(len(feature1), len(feature2))
    feature1 = feature1[:min_length]
    feature2 = feature2[:min_length]
    
    # Compute cosine similarity
    similarity = sk_cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))[0][0]
    return similarity

# Identify signature by comparing with database
def identify_signature(feature_vector: np.ndarray, database: dict, threshold: float = 0.75):
    if feature_vector is None or not database:
        return None, 0.0
    
    results = {}
    # For each person in the database
    for person_id, feature_list in database.items():
        person_similarities = []
        # Compare against all samples for this person
        for stored_feature in feature_list:
            similarity = compute_similarity(feature_vector, stored_feature)
            person_similarities.append(similarity)
        
        # Take the average of the top 3 similarities (or all if fewer than 3)
        person_similarities.sort(reverse=True)
        top_similarities = person_similarities[:min(3, len(person_similarities))]
        avg_similarity = sum(top_similarities) / len(top_similarities) if top_similarities else 0
        
        results[person_id] = avg_similarity
    
    # Find the person with the highest similarity
    if not results:
        return None, 0.0
    
    best_person = max(results.items(), key=lambda x: x[1])
    person_id, similarity = best_person
    
    # Only return a match if it's above the threshold
    if similarity >= threshold:
        return person_id, similarity
    return None, similarity

# Enroll a new person with multiple signature samples
def enroll_person(new_person_id: str, signature_images: list, database: dict):
    if not signature_images:
        st.error("No signature images provided for enrollment.")
        return database
    
    features = []
    valid_count = 0
    
    with st.spinner("Processing signature samples..."):
        progress_bar = st.progress(0)
        for i, img in enumerate(signature_images):
            # Update progress
            progress_bar.progress((i + 1) / len(signature_images))
            
            # Process image
            preprocessed = preprocess_image(img)
            signature_region = extract_signature(preprocessed)
            
            # Check if valid signature found
            if signature_region is None or not check_signature_presence(signature_region):
                st.warning(f"Sample {i+1}: No valid signature detected. Skipping.")
                continue
            
            # Extract features
            feature = extract_features(signature_region)
            if feature is not None:
                features.append(feature)
                valid_count += 1
                
                # Show the extracted signature
                st.image(signature_region, caption=f"Sample {i+1}: Extracted Signature", width=300)
    
    # Check if we got any valid features
    if not features:
        st.error("No valid signature features extracted. Enrollment failed.")
        return database
    
    # Add or update the person's entry in the database
    if new_person_id in database:
        database[new_person_id].extend(features)
        st.success(f"Added {valid_count} new signature samples for existing person '{new_person_id}'.")
    else:
        database[new_person_id] = features
        st.success(f"Enrolled new person '{new_person_id}' with {valid_count} signature samples.")
    
    return database

# ---------------------------
# STREAMLIT APP LAYOUT
# ---------------------------

def main():
    
    st.title("✍️ AI-Based Signature Verification System")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        app_mode = st.selectbox(
                "Choose the app mode",
                ["Verify Signature", "Enroll New Person", "Manage Database"]
            )
            
        st.markdown("---")
        st.subheader("Settings")
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Minimum similarity score required for verification"
        )
        
        st.markdown("---")
        st.caption("© 2025 AI Signature Verification")

    # Load the database from GCP
    database = load_database_from_gcp()
    
    # Display different app modes
    if app_mode == "Verify Signature":
        run_verification_mode(database, similarity_threshold)
    elif app_mode == "Enroll New Person":
        run_enrollment_mode(database)
    elif app_mode == "Manage Database":
        run_management_mode(database)


def run_verification_mode(database, threshold):
    st.header("📝 Signature Verification")
    
    # Show message if database is empty
    if not database:
        st.warning("No signatures enrolled yet. Please enroll someone first.")
        return
    
    # Upload signature image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Signature")
        uploaded_file = st.file_uploader(
            "Upload an image containing a signature",
            type=["png", "jpg", "jpeg", "bmp"],
            help="The image should have good contrast between signature and background"
        )
        
        # Person selection
        selected_person = st.selectbox(
            "Select person to verify against (optional)",
            ["Any person in database"] + list(database.keys()),
            help="Select a specific person or verify against entire database"
        )
    
    if uploaded_file is not None:
        # Load and process image
        image = load_image(uploaded_file)
        if image is None:
            return
            
        image_hash = generate_image_hash(image)
        
        with st.spinner("Processing signature..."):
            start_time = time.time()
            
            # Preprocess image
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is None:
                st.error("Error during image preprocessing.")
                return
                
            # Extract signature region
            signature_region = extract_signature(preprocessed_image)
            
            # Display results
            with col2:
                st.subheader("Verification Results")
                
                if signature_region is None:
                    st.error("❌ No signature detected in the image.")
                    log_verification_attempt("unknown", 0.0, False, image_hash)
                    return
                    
                # Check if it's a valid signature
                if not check_signature_presence(signature_region):
                    st.error("❌ The extracted region does not appear to be a valid signature.")
                    log_verification_attempt("unknown", 0.0, False, image_hash)
                    return
                
                # Extract features
                feature_vector = extract_features(signature_region)
                if feature_vector is None:
                    st.error("❌ Failed to extract signature features.")
                    log_verification_attempt("unknown", 0.0, False, image_hash)
                    return
                
                # Verify against database
                if selected_person == "Any person in database":
                    # Match against entire database
                    person_id, similarity = identify_signature(feature_vector, database, threshold)
                else:
                    # Match only against selected person
                    person_db = {selected_person: database[selected_person]}
                    person_id, similarity = identify_signature(feature_vector, person_db, threshold)
                
                # Display processing time
                processing_time = time.time() - start_time
                st.caption(f"Processing time: {processing_time:.2f} seconds")
                
                # Show results
                if person_id is not None:
                    st.success(f"✅ Signature verified as '{person_id}'")
                    st.metric("Similarity Score", f"{similarity:.2%}")
                    
                    # Additional confidence indicators
                    if similarity > 0.9:
                        st.info("🔒 Very high confidence match")
                    elif similarity > 0.8:
                        st.info("✅ High confidence match")
                    else:
                        st.info("ℹ️ Moderate confidence match")
                        
                    log_verification_attempt(person_id, similarity, True, image_hash)
                else:
                    st.error("❌ Signature verification failed")
                    if similarity > 0.5:
                        st.warning(f"Closest match: similarity score {similarity:.2%} (below threshold)")
                    else:
                        st.warning("No similar signatures found in database")
                        
                    log_verification_attempt("failed", similarity, False, image_hash)
        
        # Display images
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
            
        with col2:
            if signature_region is not None:
                st.image(signature_region, caption="Extracted Signature", use_column_width=True)


def run_enrollment_mode(database):
    st.header("➕ Enroll New Person")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enrollment Details")
        
        # Person ID input with validation
        new_person_id = st.text_input(
            "Enter person ID:",
            help="Use a unique identifier for this person"
        )
        
        # Check if person already exists
        is_existing = new_person_id in database
        if is_existing and new_person_id:
            st.info(f"'{new_person_id}' already exists with {len(database[new_person_id])} signatures. Adding more samples.")
        
        # Upload multiple signature samples
        uploaded_files = st.file_uploader(
            "Upload signature samples (at least 3 recommended)",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            help="Upload multiple samples of the person's signature for better accuracy"
        )
        
        # Show enrollment tips
        with st.expander("Tips for best results"):
            st.markdown("""
            - Upload at least 3-5 different signature samples
            - Ensure good lighting and contrast
            - Include variations of the signature
            - Use clear backgrounds
            - Crop images close to the signature if possible
            """)
    
    with col2:
        st.subheader("Preview")
        if uploaded_files:
            st.write(f"Number of samples: {len(uploaded_files)}")
            
            # Show preview of first 3 samples
            preview_cols = st.columns(min(3, len(uploaded_files)))
            for i, (col, file) in enumerate(zip(preview_cols, uploaded_files[:3])):
                with col:
                    image = load_image(file)
                    if image is not None:
                        st.image(image, caption=f"Sample {i+1}", use_column_width=True)
    
    # Enrollment button
    if st.button("Enroll Person", type="primary", disabled=(not new_person_id or not uploaded_files)):
        if new_person_id.strip() == "":
            st.error("Please provide a valid person ID.")
        elif not uploaded_files:
            st.error("Please upload at least one signature image.")
        else:
            # Read all uploaded images
            images = []
            for file in uploaded_files:
                img = load_image(file)
                if img is not None:
                    images.append(img)
            
            if not images:
                st.error("None of the uploaded files could be processed. Please check the formats.")
                return
                
            # Enroll the person
            database = enroll_person(new_person_id, images, database)
            
            # Save updated database to GCP
            if upload_database_to_gcp(database):
                st.balloons()
                st.success(f"Enrollment completed for '{new_person_id}'!")


def run_management_mode(database):
    st.header("🗄️ Database Management")
    
    # Display database statistics
    st.subheader("Database Statistics")
    
    if not database:
        st.info("Database is empty. No persons enrolled yet.")
    else:
        total_persons = len(database)
        total_signatures = sum(len(sigs) for sigs in database.values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Persons", total_persons)
        with col2:
            st.metric("Total Signatures", total_signatures)
        with col3:
            st.metric("Avg. Signatures per Person", round(total_signatures / total_persons, 1))
        
        # Person details
        st.subheader("Enrolled Persons")
        person_data = []
        for person_id, signatures in database.items():
            person_data.append({
                "Person ID": person_id,
                "Signature Samples": len(signatures),
                "Last Update": datetime.now().strftime("%Y-%m-%d")  # In a real app, store timestamps
            })
        
        if person_data:
            df = pd.DataFrame(person_data)
            st.dataframe(df, use_container_width=True)
    
    # Database management options
    st.subheader("Management Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Delete person
        if database:
            person_to_delete = st.selectbox(
                "Select person to delete",
                list(database.keys()),
                key="delete_person"
            )
            
            if st.button("Delete Person", type="secondary"):
                if st.checkbox(f"Confirm deletion of '{person_to_delete}'"):
                    del database[person_to_delete]
                    if upload_database_to_gcp(database):
                        st.success(f"Person '{person_to_delete}' deleted successfully.")
                        st.experimental_rerun()
    
    with col2:
        # Backup/restore options
        if st.button("Download Database Backup"):
            # Create a downloadable backup
            db_bytes = pickle.dumps(database)
            st.download_button(
                "Download Backup File",
                db_bytes,
                file_name=f"signature_db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                mime="application/octet-stream"
            )
        
        uploaded_backup = st.file_uploader("Upload Database Backup", type=["pkl"])
        if uploaded_backup is not None and st.button("Restore Database"):
            try:
                backup_data = pickle.loads(uploaded_backup.read())
                if not isinstance(backup_data, dict):
                    st.error("Invalid backup file format.")
                else:
                    # Merge with existing database or replace
                    if st.checkbox("Replace entire database? (unchecked = merge with existing)"):
                        database = backup_data
                    else:
                        for person_id, signatures in backup_data.items():
                            if person_id in database:
                                database[person_id].extend(signatures)
                            else:
                                database[person_id] = signatures
                    
                    if upload_database_to_gcp(database):
                        st.success("Database restored successfully.")
                        st.experimental_rerun()
            except Exception as e:
                st.error(f"Error restoring database: {e}")

if __name__ == "__main__":
    main()