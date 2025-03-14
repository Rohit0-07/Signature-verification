# version 2 - Revised Signature Verification using CNN-based Embeddings
import streamlit as st

st.set_page_config(
    page_title="LLM-Based Signature Verification",
    page_icon="✍️",
    layout="wide"
)

import cv2
import numpy as np
import pandas as pd
import pickle
import time
import hashlib
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# CONFIGURATION & GCP SETUP
# ---------------------------

# Set your Google Cloud Storage bucket name here
GCP_BUCKET_NAME = 'verifysign'
DATABASE_FILENAME = 'llm_signature_database.pkl'
VERIFICATION_LOG_FILENAME = 'verification_log.csv'
ADMIN_CONFIG_FILENAME = 'admin_config.pkl'

# Default admin password hash (you should change this during first run)
DEFAULT_ADMIN_PASSWORD = 'adminpassword123'  # This will be hashed before storage

# Google Cloud Storage functions (unchanged)
def get_storage_client():
    try:
        # Directly use Streamlit secrets for authentication
        creds_info = dict(st.secrets["gcp_credentials"])
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        return storage.Client(credentials=credentials, project=creds_info.get('project_id'))
    except KeyError:
        st.error("GCP credentials not found in Streamlit secrets. Please set up the 'gcp_credentials' section.")
        return None
    except Exception as e:
        st.error(f"GCP Authentication Error: {str(e)}")
        st.info("If running locally, make sure your .streamlit/secrets.toml file is properly configured.")
        return None

def ensure_bucket_exists():
    client = get_storage_client()
    if not client:
        return False
    try:
        bucket = client.bucket(GCP_BUCKET_NAME)
        if not bucket.exists():
            st.info(f"Bucket '{GCP_BUCKET_NAME}' does not exist. Creating now...")
            bucket = client.create_bucket(GCP_BUCKET_NAME)
            st.success(f"Bucket '{GCP_BUCKET_NAME}' created successfully.")
        return True
    except Exception as e:
        st.error(f"Error accessing or creating bucket: {e}")
        return False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_database_from_gcp():
    # Add this line at the beginning of the function
    if 'load_database_from_gcp' in st.session_state and hasattr(load_database_from_gcp, 'clear'):
        load_database_from_gcp.clear()
    client = get_storage_client()
    if not client:
        return {}
    if not ensure_bucket_exists():
        return {}
    bucket = client.bucket(GCP_BUCKET_NAME)
    blob = bucket.blob(DATABASE_FILENAME)
    try:
        if not blob.exists():
            st.info(f"Database file '{DATABASE_FILENAME}' not found. Starting with an empty database.")
            return {}
        data = blob.download_as_bytes()
        database = pickle.loads(data)
        st.success("Database loaded successfully from GCP.")
        return database
    except Exception as e:
        st.warning(f"Error loading database: {e}. Starting with an empty database.")
        return {}

def upload_database_to_gcp(database):
    client = get_storage_client()
    if not client:
        st.error("Could not connect to GCP. Database not updated.")
        return False
    if not ensure_bucket_exists():
        return False
    try:
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(DATABASE_FILENAME)
        data = pickle.dumps(database)
        blob.upload_from_string(data)
        load_database_from_gcp.clear()  # Clear cached database
        st.success("Database updated on GCP.")
        return True
    except Exception as e:
        st.error(f"Error uploading database to GCP: {e}")
        return False

def log_verification_attempt(person_id, similarity, verified, image_hash):
    client = get_storage_client()
    if not client:
        st.warning("Could not connect to GCP. Verification attempt not logged.")
        return False
    if not ensure_bucket_exists():
        return False
    try:
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(VERIFICATION_LOG_FILENAME)
        log_entry = f"{datetime.now()},{person_id},{similarity:.4f},{verified},{image_hash}\n"
        if blob.exists():
            content = blob.download_as_string().decode('utf-8')
        else:
            content = "timestamp,person_id,similarity,verified,image_hash\n"
        content += log_entry
        blob.upload_from_string(content)
        return True
    except Exception as e:
        st.warning(f"Could not log verification attempt: {e}")
        return False

# ---------------------------
# ADMIN AUTHENTICATION
# ---------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_admin_config():
    client = get_storage_client()
    if not client:
        return {"password_hash": hash_password(DEFAULT_ADMIN_PASSWORD)}
    if not ensure_bucket_exists():
        return {"password_hash": hash_password(DEFAULT_ADMIN_PASSWORD)}
    bucket = client.bucket(GCP_BUCKET_NAME)
    blob = bucket.blob(ADMIN_CONFIG_FILENAME)
    try:
        if not blob.exists():
            admin_config = {"password_hash": hash_password(DEFAULT_ADMIN_PASSWORD)}
            data = pickle.dumps(admin_config)
            blob.upload_from_string(data)
            return admin_config
        data = blob.download_as_bytes()
        admin_config = pickle.loads(data)
        return admin_config
    except Exception as e:
        st.warning(f"Error loading admin config: {e}. Using default.")
        return {"password_hash": hash_password(DEFAULT_ADMIN_PASSWORD)}

def save_admin_config(admin_config):
    client = get_storage_client()
    if not client:
        st.error("Could not connect to GCP. Admin config not updated.")
        return False
    if not ensure_bucket_exists():
        return False
    try:
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(ADMIN_CONFIG_FILENAME)
        data = pickle.dumps(admin_config)
        blob.upload_from_string(data)
        return True
    except Exception as e:
        st.error(f"Error saving admin config: {e}")
        return False

def verify_admin_password(password, admin_config):
    password_hash = hash_password(password)
    return password_hash == admin_config["password_hash"]

# ---------------------------
# SIGNATURE ANALYSIS USING CNN-BASED EMBEDDING
# ---------------------------

# Enhanced signature detection using image processing
def detect_signature_region(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply GaussianBlur to reduce noise and improve thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Otsu's thresholding after inverting the image
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Use a minimal area threshold to filter noise and avoid cutting out parts of the signature
            if w * h > 1000:
                return image[y:y+h, x:x+w]
        return image
    except Exception as e:
        st.error(f"Error detecting signature region: {e}")
        return image

# Keep the same function name for compatibility with the rest of the code
def detect_signature_with_vision_api(image):
    """Detect signature region using simple image processing."""
    return detect_signature_region(image)

# Load the CNN model (MobileNetV2) once and cache it
@st.cache_resource(show_spinner=False)
def load_cnn_model():
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return model

# Generate signature embedding using MobileNetV2 features
def generate_signature_embedding(image):
    try:
        model = load_cnn_model()
        # Convert image from BGR to RGB and resize to 224x224
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        x = np.expand_dims(resized, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        return features.flatten()  # Fixed length vector (typically 1280 dimensions)
    except Exception as e:
        st.error(f"Error generating signature embedding: {e}")
        # Fallback: return a random vector of fixed dimension
        return np.random.rand(1280).astype(np.float32)

# Normalize embeddings before calculating similarity
def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

# Use Euclidean distance for a more robust similarity measure
def compare_signature_embeddings(embedding1, embedding2):
    if embedding1 is None or embedding2 is None:
        return 0.0
    if len(embedding1) != len(embedding2):
        st.warning("Embedding dimensions don't match. Using partial comparison.")
        min_len = min(len(embedding1), len(embedding2))
        embedding1 = embedding1[:min_len]
        embedding2 = embedding2[:min_len]
    embedding1 = normalize_embedding(embedding1)
    embedding2 = normalize_embedding(embedding2)
    distance = np.linalg.norm(embedding1 - embedding2)
    similarity = 1 / (1 + distance)  # Convert distance to similarity
    return similarity

def identify_signature_with_llm(signature_embedding, database, threshold=0.75):
    """Identify a signature by comparing its CNN embedding with the database."""
    if signature_embedding is None or not database:
        return None, 0.0
    results = {}
    for person_id, embeddings in database.items():
        person_similarities = []
        for stored_embedding in embeddings:
            similarity = compare_signature_embeddings(signature_embedding, stored_embedding)
            person_similarities.append(similarity)
        person_similarities.sort(reverse=True)
        top_similarities = person_similarities[:min(3, len(person_similarities))]
        avg_similarity = sum(top_similarities) / len(top_similarities) if top_similarities else 0
        results[person_id] = avg_similarity
    if not results:
        return None, 0.0
    best_person = max(results.items(), key=lambda x: x[1])
    person_id, similarity = best_person
    if similarity >= threshold:
        return person_id, similarity
    return None, similarity

def verify_signature_with_llm(signature_image, database, specific_person=None, threshold=0.75):
    """Complete signature verification process using CNN-based embedding."""
    with st.spinner("Analyzing signature..."):
        signature_region = detect_signature_with_vision_api(signature_image)
        if signature_region is None:
            return None, 0.0, "No signature detected in the image."
        signature_embedding = generate_signature_embedding(signature_region)
        if signature_embedding is None:
            return None, 0.0, "Failed to analyze signature features."
        if specific_person and specific_person != "Any person in database":
            if specific_person in database:
                person_db = {specific_person: database[specific_person]}
                person_id, similarity = identify_signature_with_llm(signature_embedding, person_db, threshold)
            else:
                return None, 0.0, f"Person '{specific_person}' not found in database."
        else:
            person_id, similarity = identify_signature_with_llm(signature_embedding, database, threshold)
        if person_id is not None:
            return person_id, similarity, "Signature verified successfully."
        else:
            return None, similarity, "Signature verification failed."

def enroll_person_with_llm(new_person_id, signature_images, database):
    """Enroll a new person using CNN-based signature analysis."""
    if not signature_images:
        st.error("No signature images provided for enrollment.")
        return database

    embeddings = []
    valid_count = 0
    with st.spinner("Processing signature samples..."):
        progress_bar = st.progress(0)
        for i, img in enumerate(signature_images):
            progress_bar.progress((i + 1) / len(signature_images))
            signature_region = detect_signature_with_vision_api(img)
            if signature_region is None:
                st.warning(f"Sample {i+1}: No valid signature detected. Skipping.")
                continue
            embedding = generate_signature_embedding(signature_region)
            if embedding is not None:
                embeddings.append(embedding)
                valid_count += 1
                st.image(signature_region, caption=f"Sample {i+1}: Detected Signature", width=300)

    if not embeddings:
        st.error("No valid signature features extracted. Enrollment failed.")
        return database

    if new_person_id in database:
        database[new_person_id].extend(embeddings)
        st.success(f"Added {valid_count} new signature samples for existing person '{new_person_id}'.")
    else:
        database[new_person_id] = embeddings
        st.success(f"Enrolled new person '{new_person_id}' with {valid_count} signature samples.")

    # Upload the updated database to GCP
    if upload_database_to_gcp(database):
        load_database_from_gcp.clear()  # Clear cached database if caching is re-enabled later
        st.success("✅ Database successfully updated!")
        database = load_database_from_gcp()  # Reload updated database
        st.write("Updated Database:", database)  # Debug: show current database
        st.balloons()
        st.success(f"🎉 Enrollment completed for '{new_person_id}'!")
    else:
        st.error("❌ Failed to upload the updated database. Please check GCP configuration or try again.")

    return database
def generate_image_hash(image):
    if image is None:
        return "none"
    small = cv2.resize(image, (32, 32))
    return str(hash(small.tobytes()))

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

# ---------------------------
# STREAMLIT APP LAYOUT
# ---------------------------
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'admin_config' not in st.session_state:
        st.session_state.admin_config = load_admin_config()
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
    
    st.title("✍️ CNN-Based Signature Verification System")
    
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
        st.caption("© 2025 LLM-Based Signature Verification")
    
    database = load_database_from_gcp()
    
    if st.session_state.first_run and st.session_state.admin_config["password_hash"] == hash_password(DEFAULT_ADMIN_PASSWORD):
        st.warning("⚠️ You are using the default admin password. Please change it immediately in the 'Manage Database' section.")
    st.session_state.first_run = False
    
    if app_mode == "Verify Signature":
        run_verification_mode(database, similarity_threshold)
    elif app_mode == "Enroll New Person":
        run_enrollment_mode(database)
    elif app_mode == "Manage Database":
        if not st.session_state.authenticated:
            admin_login()
        else:
            run_management_mode(database)

def admin_login():
    st.header("🔐 Admin Login Required")
    st.write("Authentication is required to access database management features.")
    with st.form("login_form"):
        password = st.text_input("Admin Password", type="password")
        submit_button = st.form_submit_button("Login")
        if submit_button:
            if verify_admin_password(password, st.session_state.admin_config):
                st.session_state.authenticated = True
                st.success("Authentication successful!")
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")

def run_verification_mode(database, threshold):
    st.header("📝 Signature Verification")
    if not database:
        st.warning("No signatures enrolled yet. Please enroll someone first.")
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload Signature")
        uploaded_file = st.file_uploader(
            "Upload an image containing a signature",
            type=["png", "jpg", "jpeg", "bmp"],
            help="The image should have good contrast between signature and background"
        )
        selected_person = st.selectbox(
            "Select person to verify against (optional)",
            ["Any person in database"] + list(database.keys()),
            help="Select a specific person or verify against entire database"
        )
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        if image is None:
            return
        image_hash = generate_image_hash(image)
        start_time = time.time()
        person_id, similarity, message = verify_signature_with_llm(image, database, selected_person, threshold)
        processing_time = time.time() - start_time
        with col2:
            st.subheader("Verification Results")
            st.caption(f"Processing time: {processing_time:.2f} seconds")
            if person_id is not None:
                st.success(f"✅ Signature verified as '{person_id}'")
                st.metric("Similarity Score", f"{similarity:.2%}")
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
            st.text(message)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
def run_enrollment_mode(database):
    st.header("➕ Enroll New Person")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Enrollment Details")
        new_person_id = st.text_input("Enter person ID:", help="Use a unique identifier for this person")
        uploaded_files = st.file_uploader(
            "Upload signature samples (at least 3 recommended)",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            help="Upload multiple samples of the person's signature for better accuracy"
        )

    with col2:
        st.subheader("Preview")
        if uploaded_files:
            st.write(f"Number of samples: {len(uploaded_files)}")
            preview_cols = st.columns(min(3, len(uploaded_files)))
            for i, (col, file) in enumerate(zip(preview_cols, uploaded_files[:3])):
                with col:
                    image = load_image(file)
                    if image is not None:
                        st.image(image, caption=f"Sample {i+1}", use_container_width=True)

    if st.button("Preprocess Images", type="primary", disabled=(not new_person_id or not uploaded_files)):
        if new_person_id.strip() == "":
            st.error("Please provide a valid person ID.")
        elif not uploaded_files:
            st.error("Please upload at least one signature image.")
        else:
            images = [load_image(file) for file in uploaded_files if load_image(file) is not None]

            if not images:
                st.error("None of the uploaded files could be processed. Please check the formats.")
                return

            valid_images = []
            for i, img in enumerate(images):
                st.write(f"Sample {i+1}")
                signature_region = detect_signature_with_vision_api(img)
                if signature_region is not None:
                    st.image(signature_region, caption="Processed Image", use_container_width=True)

                    # Ensure checkbox state sticks per image
                    if f"keep_image_{i}" not in st.session_state:
                        st.session_state[f"keep_image_{i}"] = True

                    keep_image = st.checkbox(
                        f"Keep Sample {i+1}",
                        value=st.session_state[f"keep_image_{i}"],
                        key=f"checkbox_{i}"
                    )

                    if keep_image:
                        valid_images.append(signature_region)
                else:
                    st.error("Please enter a valid signature image. No signature detected in this image.")

            if valid_images and st.button("Enroll Person with AI", type="primary"):
                st.info("📤 Uploading data to the database...")
                # This function already handles upload to GCP and cache clearing
                database = enroll_person_with_llm(new_person_id, valid_images, database)
                # Remove the redundant upload code here
    with st.expander("Tips for best results with AI"):
        st.markdown(
            """
            - Upload at least 3-5 different signature samples  
            - Ensure good lighting and contrast  
            - Include variations of the signature  
            - Use clear backgrounds  
            - Crop images close to the signature if possible  
            - Include both clean signatures and signatures in context (e.g., on forms)
            """
        )

def run_management_mode(database):
    st.header("🗄️ Database Management")
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()
    with st.expander("🔑 Change Admin Password"):
        st.write("Change the administrator password for database management")
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            submit_button = st.form_submit_button("Change Password")
            if submit_button:
                if not verify_admin_password(current_password, st.session_state.admin_config):
                    st.error("Current password is incorrect.")
                elif new_password != confirm_password:
                    st.error("New passwords don't match.")
                elif len(new_password) < 8:
                    st.error("New password must be at least 8 characters long.")
                else:
                    st.session_state.admin_config["password_hash"] = hash_password(new_password)
                    if save_admin_config(st.session_state.admin_config):
                        st.success("Password changed successfully!")
                    else:
                        st.error("Failed to save new password.")
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
        st.subheader("Enrolled Persons")
        person_data = []
        for person_id, signatures in database.items():
            person_data.append({
                "Person ID": person_id,
                "Signature Samples": len(signatures),
                "Last Update": datetime.now().strftime("%Y-%m-%d")
            })
        if person_data:
            df = pd.DataFrame(person_data)
            st.dataframe(df, use_container_width=True)
    st.subheader("Management Options")
    col1, col2 = st.columns(2)
    with col1:
        if database:
            person_to_delete = st.selectbox("Select person to delete", list(database.keys()), key="delete_person")
            confirm_deletion = st.checkbox(f"Confirm deletion of '{person_to_delete}'", key="confirm_deletion")
            if st.button("Delete Person", type="secondary"):
                if confirm_deletion:
                    del database[person_to_delete]
                    if upload_database_to_gcp(database):
                        load_database_from_gcp.clear()  # Clear cache to reflect deletion immediately
                        st.success(f"Person '{person_to_delete}' deleted successfully.")
                        st.rerun()
                else:
                    st.error("Please confirm deletion by checking the box.")
    with col2:
        if st.button("Download Database Backup"):
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
                        st.rerun()
            except Exception as e:
                st.error(f"Error restoring database: {e}")

if __name__ == "__main__":
    main()
