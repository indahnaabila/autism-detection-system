
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

    filename_consent = r"E:\0_Indah Nabila\8th Sem\Streamlit Tugas Akhir\1. output consent.xlsx"

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
    filename_scq = r"E:\0_Indah Nabila\8th Sem\Streamlit Tugas Akhir\2. output scq.xlsx"

    questions = [
        "1. Apakah dia sekarang bisa berbicara menggunakan frasa atau kalimat pendek? Jika tidak, lanjutkan ke Pertanyaan 8",
        "2. Apakah Anda memiliki percakapan bolak-balik dengan dia yang melibatkan bergantian atau membangun apa yang telah Anda katakan?",
        "3. Apakah dia pernah menggunakan frasa yang aneh atau mengatakan hal yang sama berulang-ulang dengan cara yang hampir sama (baik frasa yang dia dengar dari orang lain atau yang dia buat sendiri)?",
        "4. Apakah dia pernah menggunakan pertanyaan atau pernyataan yang tidak pantas secara sosial? Misalnya, apakah dia secara teratur mengajukan pertanyaan pribadi atau membuat komentar pribadi pada saat yang tidak tepat?",
        "5. Apakah dia pernah mengacaukan kata ganti (misalnya mengatakan kamu atau dia untuk saya)?",
        "6. Apakah dia pernah menggunakan kata-kata yang sepertinya dia ciptakan atau buat sendiri; menyatakan hal-hal dengan cara yang aneh, tidak langsung; atau menggunakan cara metaforis dalam berbicara (misalnya mengatakan hujan panas untuk uap)?",
        "7. Apakah dia pernah mengulang kata-kata yang sama berulang kali dengan cara yang sama atau memaksa Anda untuk mengulang-ulang kata yang sama?",
        "8. Apakah dia pernah memiliki hal-hal yang tampaknya harus dia lakukan dengan cara yang sangat khusus atau urutan atau ritual yang dia minta Anda lalui?",
        "9. Apakah ekspresi wajahnya biasanya tampak sesuai dengan situasi tertentu, sejauh yang Anda ketahui?",
        "10. Apakah dia pernah menggunakan tangan Anda seperti alat atau seolah-olah bagian dari tubuhnya sendiri (misalnya menunjuk dengan jari Anda, meletakkan tangan Anda di kenop pintu untuk membuat Anda membuka pintu)?",
        "11. Apakah dia pernah memiliki minat yang menyita perhatiannya dan mungkin terlihat aneh bagi orang lain (misalnya lampu lalu lintas, pipa pembuangan, jadwal)?",
        "12. Apakah dia pernah tampak lebih tertarik pada bagian-bagian dari mainan atau objek (misalnya memutar roda mobil), daripada menggunakan objek seperti yang dimaksudkan?",
        "13. Apakah dia pernah memiliki minat khusus yang tidak biasa dalam intensitasnya tetapi sebaliknya sesuai untuk usia dan kelompok sebayanya (misalnya kereta api atau dinosaurus)?",
        "14. Apakah dia pernah tampak sangat tertarik pada penglihatan, sentuhan, suara, rasa, atau bau benda atau orang?",
        "15. Apakah dia pernah memiliki gerakan tangan atau jari yang aneh, seperti mengibas-ngibaskan atau menggerakkan jari-jarinya di depan matanya?",
        "16. Apakah dia pernah memiliki gerakan rumit seluruh tubuhnya, seperti berputar atau berulang kali melompat naik turun?",
        "17. Apakah dia pernah secara sengaja melukai dirinya sendiri, seperti menggigit lengannya atau membenturkan kepalanya?",
        "18. Apakah dia pernah memiliki objek (selain boneka lembut atau selimut penghibur) yang harus dia bawa-bawa?",
        "19. Apakah dia pernah memiliki teman tertentu atau sahabat terbaik?",
        "20. Apakah dia pernah berbicara dengan Anda hanya untuk bersikap ramah (bukan untuk mendapatkan sesuatu)?",
        "21. Apakah dia pernah secara spontan meniru Anda (atau orang lain) atau apa yang Anda lakukan (seperti menyedot debu, berkebun, atau memperbaiki barang)?",
        "22. Apakah dia pernah secara spontan menunjuk benda di sekitarnya hanya untuk menunjukkan benda tersebut kepada Anda (bukan karena dia menginginkannya)?",
        "23. Apakah dia pernah menggunakan isyarat, selain menunjuk atau menarik tangan Anda, untuk memberi tahu Anda apa yang dia inginkan?",
        "24. Apakah dia mengangguk untuk menunjukkan 'ya'?",
        "25. Apakah dia menggelengkan kepala untuk menunjukkan 'tidak'?",
        "26. Apakah dia biasanya melihat Anda langsung di wajah saat melakukan sesuatu bersama Anda atau berbicara dengan Anda?",
        "27. Apakah dia tersenyum kembali jika seseorang tersenyum padanya?",
        "28. Apakah dia pernah menunjukkan hal-hal yang menarik baginya untuk menarik perhatian Anda?",
        "29. Apakah dia pernah menawarkan untuk berbagi barang selain makanan dengan Anda?",
        "30. Apakah dia pernah tampak ingin Anda ikut menikmati sesuatu?",
        "31. Apakah dia pernah mencoba menghibur Anda saat Anda sedih atau terluka?",
        "32. Jika dia ingin sesuatu atau membutuhkan bantuan, apakah dia melihat Anda dan menggunakan gerakan dengan suara atau kata-kata untuk mendapatkan perhatian Anda?",
        "33. Apakah dia menunjukkan rentang ekspresi wajah yang normal?",
        "34. Apakah dia pernah secara spontan bergabung dan mencoba meniru aksi dalam permainan sosial, seperti The Mulberry Bush atau London Bridge Is Falling Down?",
        "35. Apakah dia bermain permainan pura-pura atau berpura-pura?",
        "36. Apakah dia tampak tertarik pada anak-anak lain seumurannya yang tidak dia kenal?",
        "37. Apakah dia merespons positif saat anak lain mendekatinya?",
        "38. Jika Anda masuk ke ruangan dan mulai berbicara dengannya tanpa memanggil namanya, apakah dia biasanya menoleh dan memperhatikan Anda?",
        "39. Apakah dia pernah bermain permainan imajinatif dengan anak lain sehingga Anda dapat mengatakan bahwa setiap anak memahami apa yang sedang dipura-purakan oleh yang lain?",
        "40. Apakah dia bermain secara kooperatif dalam permainan yang memerlukan beberapa bentuk bergabung dengan sekelompok anak lain, seperti petak umpet atau permainan bola?"
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
        st.markdown("Instruksi: Silakan jawab setiap pertanyaan dengan YA atau TIDAK. Pastikan Anda memikirkan jawaban Anda berdasarkan pengamatan Anda selama tiga bulan terakhir.")
    
        responses = []
        for i, question_text in enumerate(questions):
            st.write(f"Pertanyaan {i+1}")
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

with tab3:
    st.title("Video Feature Extraction")
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    sheet_name = st.text_input("Enter the sheet name for the Excel output", "VideoAnalysis")
    output_file = r"E:\0_Indah Nabila\8th Sem\Streamlit Tugas Akhir\3. output calculation.xlsx"
    fps_value = st.number_input("Enter the FPS value for processing", min_value=1, value=1)

    if video_file is not None:
        if st.button("Process Video"):
            with st.spinner('Processing...'):
                try:
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

                        if frame_count % frames_to_skip == 0:
                            preprocessor = ImagePreprocessor(frame)
                            preprocessor.read_and_resize()
                            resize_image = preprocessor.resized_image

                            extractor = LandmarkExtractor()
                            landmarks = extractor.extract_landmarks(resize_image)

                            if landmarks is None:
                                print(f"No landmarks detected in {frame_count}.")
                                continue

                            corrected_image = ImageSlopeCorrector.rotate_image_based_on_landmarks(resize_image, landmarks)
                            corrected_landmarks = extractor.extract_landmarks(corrected_image)

                            if corrected_landmarks is None:
                                print(f"No landmarks detected after correction in {frame_count}.")
                                continue

                            kalkulasi_fitur = FeatureCalculator()
                            features = kalkulasi_fitur.rumus29(corrected_landmarks)
                            all_features.append(features)

                        frame_count += 1

                    cap.release()

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
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
                        try:
                            features_df = pd.DataFrame(all_features)
                            features_df.rename(columns=new_column_names, inplace=True)
                            # If DataFrame is not empty, save to Excel and show success
                            if not features_df.empty:
                                data_persistence.save_to_excel(features_df, sheet_name=sheet_name)
                                st.success("Features extracted and saved to Excel.")
                            else:
                                st.error("No features extracted.")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

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
            svm_model_loaded = load("svm_model_rbf_fix.joblib")

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
    
    st.header("Video Upload and Frame Extraction")
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
    st.header("Image Processing and Landmark Extraction")
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
    st.header("Prediction and Analysis")
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
