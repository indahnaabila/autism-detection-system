import streamlit as st
import pandas as pd
from datetime import datetime
import cv2
from joblib import load
from SVMneeds import DataPersistence, ImagePreprocessor, LandmarkExtractor, ImageSlopeCorrector, FeatureCalculator  # Ensure to import your custom modules
import numpy as np
import tempfile
import os
from pathlib import Path
from CNNneeds import extract_frames, process_images_in_folder, predict_folder, analyze_data_np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.title("ASD Test")

tab1, tab2, tab3, tab4 = st.tabs(["Personal Information", "Screening", "ASD Test 1 - SVM", "ASD Test 2 - CNN"])
with tab1:
    st.write("Enter your personal information here.")
    filename_consent = str(Path('1. output consent.xlsx'))
    with st.form(key='consent_form'):
        st.write("Personal Information:")
        name = st.text_input("Name:")
        address = st.text_input("Address:")
        phone_number = st.text_input("Phone Number:")
        email = st.text_input("Email:")

        st.write("Child's Information:")
        child_name = st.text_input("Child's Name:")
        child_age = st.number_input("Child's Age:", step=1, format='%d')
        child_dob = st.date_input("Child's Date of Birth:")

        st.write("Consent Statements:")
        consent1 = st.checkbox("I agree to provide data for my child and data usage for ASD detection system performance.")
        consent2 = st.checkbox("I and my child agree to be subjects in the research process without compensation.")
        consent3 = st.checkbox("I and my child agree to be interviewed and for the interview to be recorded.")
        consent4 = st.checkbox("I and my child agree for the data to be kept confidential for further research use.")

        # Date of submission (prefilled with today's date)
        submission_date = st.date_input("Date of Submission:", value=datetime.today())

        # Submit button for the form
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Data to be written to the Excel file
        data_consent = {
            'Submission Date': [submission_date],
            'Name': [name],
            'Address': [address],
            'Phone Number': [phone_number],
            'Email': [email],
            'Child Name': [child_name],
            'Child Age': [child_age],
            'Child Date of Birth': [child_dob],
            'Consent 1': [consent1],
            'Consent 2': [consent2],
            'Consent 3': [consent3],
            'Consent 4': [consent4],
        }

        # Convert the data to a pandas dataframe
        df = pd.DataFrame(data_consent)

        file_path = Path(filename_consent)
        if file_path.is_file():
            # Read the existing data
            existing_df = pd.read_excel(filename_consent)
            # Concatenate the new data
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            updated_df = df

        # Write the updated dataframe to the Excel file
        updated_df.to_excel(filename_consent, index=False)

        st.success('Form data saved to Excel successfully!')
        

        # Define the filename with the current timestamp to avoid overwriting
        #filename = f"consent_form_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"

        # Write the dataframe to an Excel file
        #df.to_excel(filename, index=False)

        #st.success('Form data saved to Excel successfully!')

with tab2:
    filename_scq = str(Path('2. output scq.xlsx'))
    questions = [
        "1. Can they now speak using phrases or short sentences? If not, proceed to Question 8.",
        "2. Do you have back-and-forth conversations with them that involve taking turns or building on what you have said?",
        "3. Do they ever use strange phrases or repeat the same thing over and over in almost the same way (whether phrases they heard from others or ones they made up themselves)?",
        "4. Do they ever use socially inappropriate questions or statements? For example, do they regularly ask personal questions or make personal comments at inappropriate times?",
        "5. Do they ever confuse pronouns (e.g., saying 'you' or 'he' instead of 'I')?",
        "6. Do they ever use words that seem to be made up or self-created; state things in a strange, indirect way; or use metaphorical ways of speaking (e.g., saying 'hot rain' for steam)?",
        "7. Do they ever repeat the same words over and over in the same way or make you repeat words over and over?",
        "8. Do they ever have things that they seem to have to do in a very specific way or a sequence or ritual that they ask you to follow?",
        "9. Do their facial expressions usually seem appropriate to the particular situation, as far as you know?",
        "10. Do they ever use your hand like a tool or as if it were part of their own body (e.g., pointing with your finger, placing your hand on a doorknob to get you to open the door)?",
        "11. Do they ever have interests that capture their attention and might seem strange to others (e.g., traffic lights, drain pipes, schedules)?",
        "12. Do they ever seem more interested in parts of toys or objects (e.g., spinning the wheels of a toy car) than using the object as intended?",
        "13. Do they ever have a special interest that is unusual in its intensity but otherwise appropriate for their age and peer group (e.g., trains or dinosaurs)?",
        "14. Do they ever seem very interested in the sight, touch, sound, taste, or smell of objects or people?",
        "15. Do they ever have odd hand or finger movements, such as flapping or moving their fingers in front of their eyes?",
        "16. Do they ever have complex whole-body movements, such as spinning or repeatedly jumping up and down?",
        "17. Do they ever intentionally hurt themselves, such as biting their arm or banging their head?",
        "18. Do they ever have objects (other than soft toys or comfort blankets) that they have to carry around?",
        "19. Do they ever have a particular friend or best friend?",
        "20. Do they ever talk to you just to be friendly (not to get something)?",
        "21. Do they ever spontaneously imitate you (or others) or what you are doing (such as vacuuming, gardening, or fixing things)?",
        "22. Do they ever spontaneously point out things around them just to show them to you (not because they want them)?",
        "23. Do they ever use gestures, other than pointing or pulling your hand, to let you know what they want?",
        "24. Do they nod to indicate 'yes'?",
        "25. Do they shake their head to indicate 'no'?",
        "26. Do they usually look directly at you when doing something with you or talking to you?",
        "27. Do they smile back if someone smiles at them?",
        "28. Do they ever show things that interest them to get your attention?",
        "29. Do they ever offer to share items other than food with you?",
        "30. Do they ever seem to want you to enjoy something with them?",
        "31. Do they ever try to comfort you when you are sad or hurt?",
        "32. If they want something or need help, do they look at you and use gestures with sounds or words to get your attention?",
        "33. Do they show a normal range of facial expressions?",
        "34. Do they ever spontaneously join and try to imitate actions in social games, such as The Mulberry Bush or London Bridge Is Falling Down?",
        "35. Do they play pretend games or make-believe?",
        "36. Do they seem interested in other children their age whom they do not know?",
        "37. Do they respond positively when another child approaches them?",
        "38. If you enter a room and start talking to them without calling their name, do they usually turn and pay attention to you?",
        "39. Do they ever play imaginative games with another child so that you can tell that each child understands what the other is pretending?",
        "40. Do they play cooperatively in games that require some form of joining with a group of other children, such as hide-and-seek or ball games?"
    ]

    scoring_yes = [False, False, True, True, False, True, True, True, False, True, True, False, False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    scoring_no = [False, True, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, False, False, True, True, False, True, False, True, False, True, False, False, False, True, True]

    assert len(scoring_yes) == len(questions)
    assert len(scoring_no) == len(questions)

    def calculate_score(responses):
        score = 0
        # Check if the first item is marked 'Yes' or 'No' and adjust the questions range accordingly
        question_range = range(1, 40) if responses[0] == "Yes" else range(7, 40)

        for i in question_range:
            if responses[i] == "Yes" and scoring_yes[i]:
                score += 1
            elif responses[i] == "No" and scoring_no[i]:
                score += 1

        return score

    # Create the form in Streamlit
    st.title("Social Communication Questionnaire (SCQ)")

    with st.form("scq_form"):
        st.markdown("Instructions: Please answer each question with YES or NO. Make sure you think about your answers based on your observations over the past three months.")
        responses = []
        for i, question_text in enumerate(questions):
            st.write(f"Question {i+1}")
            response = st.radio(
                label=question_text, 
                options=["Yes", "No"], 
                key=f"question_{i+1}"
            )
            responses.append(response)     
        submitted = st.form_submit_button("Submit")

        if submitted:
            # Convert responses to boolean: "Yes" is True, "No" is False
            #bool_responses = [resp == "Yes" for resp in responses]
            total_score = calculate_score(responses)
            st.write(f"Total score: {total_score}")

            # Data to be written to the Excel file
            data_scq = {
                'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Questionnaire Type': ['SCQ'],
                'Responses': [responses],
                'Total Score': [total_score]
            }

            # Convert the new data to a pandas dataframe
            new_df = pd.DataFrame(data_scq)

            # Check if the Excel file already exists
            file_path = Path(filename_scq)
            if file_path.is_file():
                # Read the existing data
                existing_df = pd.read_excel(filename_scq)
                # Concatenate the new data
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                updated_df = new_df

            # Write the updated dataframe to the Excel file
            updated_df.to_excel(filename_scq, index=False)

            st.success('SCQ data saved to Excel successfully!')
            
def verify_image(image, stage):
    #st.write(f"{stage} image shape: {image.shape}, dtype: {image.dtype}")
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape at {stage}: {image.shape}")
    if image.dtype != 'uint8':
        image = image.astype('uint8')
    return image

with tab3:
    st.header("Support Vector Machine (SVM) Method")
    st.markdown("""
    As explained on the Homepage, there are two classification methods that can be used on this website: SVM and CNN. This tab will use the SVM method, which has an accuracy of 92.5%, precision of 93.4%, recall of 92.5%, and an F1 score of 92.4%. However, before performing ASD classification, there are a few things to consider when recording emotional responses.
    """)
    url = "https://www.streamlit.io"
    st.subheader("Guide for Setting Up Video Recording")
    st.markdown("1. Positioning")
    st.markdown("Position yourself or your child seated on a chair in front of a table with a screen that will display the external stimulus.")
    st.markdown("2. Prepare the External Stimulus")
    st.markdown("Prepare the external stimulus from this [video](https://www.youtube.com/watch?v=mu-YLZpB6is&t=2s), and [this](https://www.youtube.com/watch?v=z5GQ6ov71E4) or you can use another video that is joyful and surprising.")
    st.markdown("3. Record Emotional Response")
    st.markdown("Record the emotional response of yourself or your child for approximately 5 minutes. Once done, save the recording and upload it below.")
    
    st.subheader("Video Upload and Frame Extraction")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    sheet_name = st.text_input("Enter the sheet name for the Excel output", "VideoAnalysis")
    output_file = str(Path('3. output calculation.xlsx'))
    fps_value = st.number_input("Enter the FPS value for processing", min_value=1, value=1)

    if video_file is not None:
        if st.button("Process Video"):
            with st.spinner('Processing...'):
                with tempfile.NamedTemporaryFile(delete=False) as tfile:
                    tfile.write(video_file.read())
                    temp_filename = tfile.name

                cap = cv2.VideoCapture(temp_filename)
                data_persistence = DataPersistence(output_file)
                all_features = []

                original_fps = cap.get(cv2.CAP_PROP_FPS)
                frames_to_skip = int(original_fps / fps_value)  # Adjusting to the user-defined FPS

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame is None:
                        st.error(f"Frame {frame_count} is None.")
                        continue

                    frame = verify_image(frame, "initial")

                    if frame_count % frames_to_skip == 0:
                        preprocessor = ImagePreprocessor(frame)
                        preprocessor.read_and_resize()
                        resize_image = verify_image(preprocessor.resized_image, "resized")

                        extractor = LandmarkExtractor()
                        landmarks = extractor.extract_landmarks(resize_image)

                        if landmarks is None:
                            #st.warning(f"No landmarks detected in frame {frame_count}.")
                            continue

                        corrected_image = ImageSlopeCorrector.rotate_image_based_on_landmarks(resize_image, landmarks)
                        corrected_image = verify_image(corrected_image, "corrected")

                        corrected_landmarks = extractor.extract_landmarks(corrected_image)
                        #st.write(f"Correcting frame {frame_count}, shape: {corrected_image.shape}, dtype: {corrected_image.dtype}")

                        if corrected_landmarks is None:
                            #st.warning(f"No landmarks detected after correction in frame {frame_count}.")
                            continue

                        kalkulasi_fitur = FeatureCalculator()
                        features = kalkulasi_fitur.rumus29(corrected_landmarks)
                        all_features.append(features)

                    frame_count += 1

                cap.release()

                if os.path.isfile(temp_filename):
                    os.unlink(temp_filename)

                new_column_names = {
                    '0': 'F0.1', '1': 'F1.1', '2': 'F2.1', '3': 'F3.1', '4': 'F4.1',
                    '5': 'F5.1', '6': 'F6.1', '7': 'F7.1', '8': 'F10.1', '9': 'F3.2',
                    '10': 'F8.2', '11': 'F9.2', '12': 'F11.2', '13': 'F15.2', '14': 'F27.2',
                    '15': 'F1.3', '16': 'F3.3', '17': 'F8.3', '18': 'F9.3', '19': 'F10.3',
                    '20': 'F15.3', '21': 'F18.3', '22': 'F21.3', '23': 'F23.3', '24': 'F28.3'
                }

                if not all_features:  # No features were extracted
                    st.error("No features extracted.")
                else:
                    features_df = pd.DataFrame(all_features)
                    features_df.rename(columns=new_column_names, inplace=True)
                    if not features_df.empty:
                        data_persistence.save_to_excel(features_df, sheet_name=sheet_name)
                        st.success("Features extracted and saved to Excel.")
                    else:
                        st.error("No features extracted.")

                st.success("Done!")
                st.write("Features extracted and saved to Excel.")

        if st.button('Predict'):

            def analyze_data_np(data):
                count_0 = len(data) - np.count_nonzero(data)
                count_1 = np.count_nonzero(data)
                probability_0 = count_0 / len(data)
                probability_1 = count_1 / len(data)
                decision = 1 if probability_1 > probability_0 else 0
                return probability_0, probability_1, decision

            # Load data from the 'VideoAnalysis' sheet
            video_analysis_data = pd.read_excel(output_file, sheet_name=sheet_name)

            # Display the first few rows of the DataFrame and its information
            video_analysis_data.head(), video_analysis_data.info()
            # Load the SVM model
            svm_model_loaded = load(str(Path('svm_model_rbf_fix.joblib')))

            video_analysis_data = video_analysis_data.drop(columns=['Unnamed: 0'])

            # Predict
            predictions = svm_model_loaded.predict(video_analysis_data)
            probability_0, probability_1, decision = analyze_data_np(predictions)
            st.write("Probability of 0:", probability_0)
            st.write("Probability of 1:", probability_1)
            st.write("Decision:", decision)
            if decision == 1:
                st.error("The patient is highly potential to have or diagnose with ASD.")
            else:
                st.success("The patient will potentially not have ASD.")

# Initialize session state for folder name and Excel writer
if 'frame_extraction_folder' not in st.session_state:
    st.session_state['frame_extraction_folder'] = ""
if 'landmarks_extraction_folder' not in st.session_state:
    st.session_state['landmarks_extraction_folder'] = ""

# Streamlit interface within tab4
with tab4:
    st.header("Convolutional Neural Network (CNN) Method")
    st.markdown("""
    As explained on the Homepage, there are two classification methods that can be used on this website: SVM and CNN. This tab will use the CNN method, which has an accuracy of 84.16%, precision of 83%, recall of 85.8%, and an F1 score of 84.4%. However, before performing ASD classification, there are a few things to consider when recording emotional responses.
    """)
    st.subheader("Guide for Setting Up Video Recording")
    st.markdown("1. Positioning")
    st.markdown("Position yourself or your child seated on a chair in front of a table with a screen that will display the external stimulus.")
    st.markdown("2. Prepare the External Stimulus")
    st.markdown("Prepare the external stimulus from this [video](https://www.youtube.com/watch?v=mu-YLZpB6is&t=2s), and [this](https://www.youtube.com/watch?v=z5GQ6ov71E4) or you can use another video that is joyful and surprising.")
    st.markdown("3. Record Emotional Response")
    st.markdown("Record the emotional response of yourself or your child for approximately 5 minutes. Once done, save the recording and upload it below.")

    st.subheader("Video Upload and Frame Extraction")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    folder_name = st.text_input("Enter the name of the output folder for frames", "extracted_frames")
    fps = st.number_input("Frames per second (FPS)", min_value=1, value=1)

    if st.button("Extract Frames"):
        if video_file is not None:
            video_path = os.path.join("temp", video_file.name)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())
            st.session_state['frame_extraction_folder'] = os.path.join("output", folder_name)
            
            with st.spinner('Extracting frames...'):
                frame_count = extract_frames(video_path, st.session_state['frame_extraction_folder'], fps)
                st.success(f"Extracted {frame_count} frames from {video_file.name} into folder {st.session_state['frame_extraction_folder']}")


    # Image processing and landmark extraction
    st.subheader("Image Processing and Landmark Extraction")
    if st.session_state['frame_extraction_folder']:
        st.write(f"Using frame extraction folder: {st.session_state['frame_extraction_folder']}")
        if 'landmarks_folder_name' not in st.session_state:
            st.session_state['landmarks_folder_name'] = st.session_state['frame_extraction_folder'] + "_landmarks"
        landmarks_folder_name = st.text_input("Enter the output folder for landmarks", st.session_state['landmarks_folder_name'])
        if st.button("Process Images"):
            st.session_state['landmarks_extraction_folder'] = landmarks_folder_name
            with st.spinner('Processing images for landmarks...'):
                process_images_in_folder(st.session_state['frame_extraction_folder'], st.session_state['landmarks_extraction_folder'])
                st.success(f"Processed images and extracted landmarks into folder {st.session_state['landmarks_extraction_folder']}")
    else:
        st.warning("Please extract frames from a video first.")

    # Prediction and analysis
    st.subheader("Prediction and Analysis")
    if st.session_state['landmarks_extraction_folder']:
        st.write(f"Using landmarks extraction folder: {st.session_state['landmarks_extraction_folder']}")
        if 'prediction_folder_name' not in st.session_state:
            st.session_state['prediction_folder_name'] = st.session_state['landmarks_extraction_folder']
        prediction_folder_name = st.text_input("Enter the folder path for prediction", st.session_state['prediction_folder_name'])
        if st.button("Predict and Analyze"):
            with st.spinner('Predicting and analyzing...'):
                predictions = predict_folder(st.session_state['prediction_folder_name'])
                probability_0_np, probability_1_np, decision_np = analyze_data_np(np.array(predictions))

                if probability_0_np is not None and probability_1_np is not None:
                    st.write(f'Probability of class 0: {probability_0_np:.4f}')
                    st.write(f'Probability of class 1: {probability_1_np:.4f}')
                    st.write(f'Decision: {decision_np}')
                    if decision_np == 'Class 1':
                        st.error("The patient is highly potential to have or diagnose with ASD.")
                    else:
                        st.success("The patient will potentially not have ASD.")
                else:
                    st.warning('No predictions to analyze.')
                
                # Save results to Excel
                df = pd.DataFrame({
                    "Image": [os.path.basename(f) for f in os.listdir(st.session_state['prediction_folder_name']) if f.endswith(('.jpg', '.jpeg', '.png'))],
                    "Prediction": predictions
                })
                sheet_name = os.path.basename(st.session_state['prediction_folder_name'])
                with pd.ExcelWriter('results.xlsx', engine='openpyxl', mode='a' if os.path.exists('results.xlsx') else 'w') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                st.success(f'Results saved to results.xlsx with sheet name {sheet_name}')    
    else:
        st.warning("Please process images and extract landmarks first.")
