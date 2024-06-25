import streamlit as st

st.set_page_config(
    page_title="Tugas Akhir",
    page_icon="🙏",
)

st.title("Autism Detection System")
st.subheader("Using Facial Landmarks and Emotional Responses")
st.write("""
Welcome to the Autism Detection System, an advanced tool designed to help detect autism through the analysis of facial expressions and emotional responses.
""")

tab1, tab2, tab3 = st.tabs(["My Research", "About Me", "FAQ"])
with tab1:
    #st.header("My Research")
    st.write("""
    Here, you can find detailed information about our study and methodology in developing the autism detection system.
    """)

    st.subheader("Research Objectives")
    st.write("""
    - To develop an accurate system for detecting autism using facial landmarks and emotional responses.
    - To provide a user-friendly interface for easy accessibility.
    - To offer a more efficient and accessible early diagnosis method for ASD.
    """)

    st.subheader("Background")
    st.write("""
    Autism Spectrum Disorder (ASD) is a developmental disability that affects communication, social interaction, and behavior. The prevalence of ASD is continuously increasing globally, including in Indonesia. Early detection of ASD in children is crucial for timely intervention, as it helps in developing essential communication, social, and behavioral skills.
    
    Traditional screening methods like Autism Spectrum Quotient (AQ), Childhood Autism Rating Scale (CARS-2), and Screening Tool for Autism in Toddlers and Young Children (STAT) are often time-consuming and costly. Therefore, there is a need for more efficient and accessible early diagnosis methods.
    """)

    st.subheader("Methodology")
    st.write("""
    1. **Data Collection**: 
        - The dataset consists of facial images and emotional response data from both ASD and Non-ASD individuals.
        - The ASD dataset is derived from real-time facial emotion recognition systems among children with autism.
        - The Non-ASD dataset is obtained from spontaneous facial expressions of children.

    2. **Data Processing**:
        - Pre-processing steps include resizing images, converting to grayscale, and detecting facial landmarks using histogram-oriented gradients (HOG) and linear support vector machines (SVM).

    3. **Feature Extraction**:
        - Key facial points are extracted to identify distances, angles, and slopes between landmarks, such as the eyes, nose, and mouth. This process helps in capturing the subtle differences in facial expressions.

    4. **Dataset Balancing**:
        - Techniques like SMOTE (Synthetic Minority Over-sampling Technique) are used to balance the dataset, ensuring that the model is trained on a representative set of data.

    5. **Model Training**:
        - Two models are trained: Support Vector Machine (SVM) for linear separations and Convolutional Neural Network (CNN) for more complex patterns in images.
        - The models are evaluated based on accuracy, sensitivity, specificity, precision, and F1 score.

    6. **Validation**:
        - The system undergoes rigorous testing to validate its accuracy and performance in detecting ASD.
    """)

    st.subheader("Results")
    st.write("""
    The system has demonstrated high accuracy in detecting ASD:
    - **SVM**: Accuracy 92.5%, Precision 93.4%, Recall 92.5%, F1 Score 92.4%.
    - **CNN**: Accuracy 84.16%, Precision 83%, Recall 85.8%, F1 Score 84.4%.
    """)

    # st.subheader("Publications")
    # st.write("""
    # - [Real-time facial emotion recognition system among children with autism based on deep learning and IoT](http://example.com)
    # - [A novel database of children's spontaneous facial expressions (LIRIS-CSE)](http://example.com)
    # """)

with tab2:
    #st.header("About Me")
    st.write("""
    Hello! I'm Indah, the creator of this autism detection system. With a background in AI and healthcare, I am dedicated to developing innovative solutions for early autism detection.
    """)

    st.subheader("My Background")
    st.write("""
    - **Education**: Bachelor's Degree in Biomedical Engineering from Institut Teknologi Sepuluh Nopember (ITS), specialized in Image Processing and AI.
    - **Supervisors**: Dr. Achmad Arifin, S.T., M.Eng. and Nada Fitrieyatul Hikmah, S.T., M.T.
    - **Experience**: Worked on various electrical, image processing, and AI projects in healthcare.
    - **Interests**: AI, healthcare technology, and improving accessibility.
    """)

    st.subheader("Contact Information")
    st.write("""
    - Email: [Click here](mailto:indahnaabila@gmail.com)
    - LinkedIn: [Click here](https://www.linkedin.com/in/indahnaabila/)
    - Instagram: [Click here](https://www.instagram.com/indahnaabila/)
    """)


with tab3:
    #st.header("Frequently Asked Questions")
    
    st.subheader("What is this system?")
    st.write("""
    This system is designed to help detect autism through the analysis of facial landmarks and emotional responses.
    """)

    st.subheader("How do I use it?")
    st.write("""
    1. Upload a photo or a video.
    2. Click on "Analyze".
    3. View your results and detailed analysis.
    """)

    st.subheader("Is my data secure?")
    st.write("""
    Yes, your data is securely handled and processed with utmost confidentiality.
    """)

    st.subheader("How accurate is the system?")
    st.write("""
    Our system has shown high accuracy in tests:
    - SVM: 92.5% accuracy
    - CNN: 84.16% accuracy
    """)

# Additional buttons for Get Started and Learn More if needed
if st.button("Get Started"):
    st.switch_page("pages/2_📃_Test.py")  
