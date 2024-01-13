from pathlib import Path 
import streamlit as st
from PIL import Image

# Define the base path to the images folder
#images = Path("images")
#assets = Path("assets")


# Function to display each experience with added emojis for flair
def display_experience(company_logo, role, company_and_duration, location, description, skills):
    col1, col2 = st.columns([1, 4])  # Create two columns for layout
    with col1:  # Column for the logo
        st.image(company_logo, width=100)  # Adjust width as needed
    with col2:  # Column for the text
        st.subheader(f"{role} 🚀")  # Added rocket emoji for excitement
        st.write(f"{company_and_duration} 🗓️")  # Calendar emoji for duration
        st.write(f"{location} 📍")  # Location pin emoji for location
        st.markdown(description)
        st.markdown(f"**Acquired valuable skills in:** {skills} 💡")  # Light bulb emoji for skills


# Function to display each study experience
def display_study(logo, degree, university, duration, description):
    col1, col2 = st.columns([1, 4])  # Create two columns for layout
    with col1:  # Column for the logo
        st.image(logo, width=100)  # Adjust width as needed
    with col2:  # Column for the text
        st.subheader(f"{degree} at {university} 🎓")
        st.write(duration)
        st.markdown(description)
        #st.markdown(f"**Skills:** {skills}")

def education():
    st.write('\n')
    st.title("Education")
    

    # Fachhochschule Kiel
    with st.expander("Erasmus Exchange Program at Fachhochschule Kiel , Germany "):
        display_study(
            "images/FH_kiel_70.jpg",
            "🎓 Erasmus Exchange Program at Fachhochschule Kiel , Germany ",
            "Fachhochschule Kiel/University of applied sciences Kiel",
            "Mar 2023 - july 2023",
            """
            * Participated in an Erasmus exchange program at Fachhochschule Kiel, Germany, during my Master's program in Information Processing at the University of Hassan II Mohammedia, Casablanca.

            * Enhanced skills and gained valuable international exposure in the field of Data science, with a focus on sonar, image processing, underwater techniques, and deep learning.

            ##### Enrolled in specialized courses during this semester, including:

            * **Image Processing with Deep Learning:** Engaged in advanced coursework, applying deep learning techniques to image processing challenges.
            * **Underwater Techniques:** Developed expertise in underwater technologies, exploring methods for efficient data acquisition and processing in aquatic environments.
            This Erasmus exchange experience at Fachhochschule Kiel provided a unique opportunity to broaden my academic and cultural horizons, contributing to my overall growth in the field of Data science.
                    
             """
        )

    # Université Hassan II Mohammedia - Master's
    with st.expander("Master's in Information Processing with a Focus on Machine Learning"):
        display_study(
            "images/UH2C_70.png",
            "🎓 Master's in Information Processing with a Focus on Machine Learning",
            "Hassan II University of Casablanca",
            "Sep 2021 - Sep 2023",
            """
            * Successfully completed a Master's degree in Information Processing with a specialized focus on machine learning, including advanced topics such as data analysis, information retrieval, and computational methods.

            * Specialized in optimizing algorithms for hardware, particularly exploring ways to enhance the efficiency of machine learning algorithms in the hardware environment.

            * In the final year, concentrated on deep learning, gaining proficiency in computer vision, which was further enhanced during a semester in Germany as an exchange student.

            ##### Developed a comprehensive set of skills, including:

            * **Machine Learning:** Proficient in machine learning techniques for predictive modeling and pattern recognition.
            * **Deep Learning:** Specialized in deep learning methodologies with a focus on computer vision applications.
            * **Image Processing:** Applied advanced techniques for image analysis and processing.
            * **Advanced C++ and Python:** Developed expertise in programming languages crucial for computational tasks.
            * **Operating Systems:** Acquired knowledge in the design and implementation of operating systems.
            * **Signal Processing:** Applied signal processing techniques for data analysis and interpretation.
            * **Microcontrollers and IoT:** Programmed microcontrollers and gained insights into the Internet of Things (IoT) technologies.
            * Engaged in research projects and collaborative initiatives within the field of Information Processing, demonstrating a commitment to advancing expertise in this dynamic domain.

            This Master's program at Université Hassan II Mohammedia, with a concentration on machine learning and deep learning, has equipped me with a diverse skill set essential for addressing contemporary challenges in the information technology landscape.
                    
               """,
            
        )

    # Université Hassan II Mohammedia - Bachelor's
    with st.expander("Bachelor's in Electrical and Electronics Engineering"):
        display_study(
            "images/UH2C_70.png",  # Replace with the path to your logo
            "Bachelor's degree, Electrical and Electronics Engineering",
            "Hassan II University of Mohammedia",
            "Sep 2020 - Jul 2021",
            """
            Completed a comprehensive Bachelor's program in Electrical and Electronics Engineering, encompassing coursework in circuit theory, electronics, digital signal processing, and control systems.

            ##### Successfully undertook courses in:

            * **Statistics:** Developed a foundation in statistical methods for data analysis.
            * **Linear Algebra:** Acquired a strong understanding of linear algebra concepts and applications.
            * **Probability:** Explored probabilistic models and statistical inference.
            * **Programming Skills:** Gained proficiency in programming, providing a versatile skill set for engineering applications.
            * **Electronics:** Explored the principles and applications of electronic systems.
            * **Signal Processing:** Acquired skills in digital signal processing techniques.

            * Applied theoretical knowledge in practical settings, participating in hands-on projects and laboratory work.

            * Graduated with a strong understanding of electrical and electronics engineering concepts, along with a diverse skill set encompassing statistical analysis, linear algebra, probability, programming, electronics, and signal processing.

            This Bachelor's program at Université Hassan II Mohammedia provided a robust education, equipping me with a broad range of skills and knowledge for success in the field of Electrical and Electronics Engineering.
                    
            """,
        )

def experience():
    st.write('\n')
    st.title("Experience")

    with st.expander("📷 AUV Team Tomkyle Internship - Underwater Image Processing"):
        display_experience(
            "images/AUV_Tomkyle_70.jpg",  # Replace with the path to your logo
            "Stereo Vision by Means of Sonar and Optical Camera",
            "AUV Team Tomkyle · Internship, Mar 2023 - Des 2023",
            "Kiel, Schleswig-Holstein, Germany · On-site",
            """
            Engaged in the intricate realm of stereo vision, addressing challenges posed by low-light conditions in underwater environments.

            * Conducted research and implemented methods for calibrating images captured from two distinct positions and sources.

            * Utilized Zhang's Method for camera calibration and implemented the YOLO algorithm for effective object detection in challenging underwater conditions.

            * Successfully addressed issues related to underwater image resolution, significantly enhancing data accuracy crucial for exploration activities.

            ##### Acquired proficiency in:
            * **UNDERWATER TECHNIQUES:** Applied specialized techniques to navigate the complexities of underwater environments.
            * **Computer Vision:** Leveraged computer vision methodologies to process and interpret underwater imagery.
            * **Scientific Papers:** Engaged with and applied findings from relevant scientific literature to enhance project outcomes.
            * **Deep Learning:** Implemented deep learning techniques, including the YOLO algorithm, to improve object detection accuracy.
                        """,
                        "UNDERWATER TECHNIQUES · Computer Vision · Scientific Papers · Deep Learning"
        )


    with st.expander("📡💡 Orange Maroc Internship - IoT and Innovation in Partnership with Google and EY"):
        display_experience(
        "images/Orange_70.jpg",
        "Innovative Intern with a Focus on IoT Projects 🚀",
        "Internship Duration: July 2022 - September 2022 (3 months)",
        "Orange Maroc, Google, and EY Collaboration, Rabat, Morocco",
        """
        * Spearheaded initiatives as part of the Orange Maroc, Google, and EY Summer Challenge 2022, focusing on creating cutting-edge solutions for social impact and innovation in the Internet of Things (IoT).

        * Collaborated on the award-winning #KIDCAN project, which earned Google's Favorite Award and secured 3rd prize from Orange Maroc. This encompassed:

        * Leveraging a mix of technical skills, including 3D Printing, Arduino, Python, Raspberry Pi, SOLIDWORKS, and C++, to develop transformative solutions.
        
        This internship, in a strategic collaboration between Orange Maroc, Google, and EY, provided a unique platform to contribute to innovative projects with a tangible impact on social causes. The recognition received for the #KIDCAN project underscores the success of our collaborative efforts.

        """,
        """
        - 3D Printing
        - Arduino
        - Python (Programming Language)
        - Raspberry Pi
        - SOLIDWORKS
        - C++
        """
    )


    with st.expander("🎓 🚀 Smart Electronic Consumption Monitoring and Cost Conversion Meter Development"):
        display_experience(
        "images/UH2C_70.png" ,
        "Intern in Electrical Engineering and Programming",
        "Internship Duration: April 2021 - June 2021 (3 months)",
        "Information Processing Laboratory, Université Hassan II, Casablanca-Settat, Morocco",
        """
        Led the development of a smart electronic meter in the Information Processing Laboratory, demonstrating proficiency in monitoring and calculating the cost of electronic consumption.

        * Acquired hands-on experience in electrical engineering, contributing to the design and implementation of the meter's circuits.

        * Applied programming skills, particularly in Arduino and C++, to ensure seamless integration and functionality of the developed meter.

        ##### Gained expertise in:

        * Electronics: Applied theoretical knowledge to practical situations, contributing to the successful implementation of the smart meter.
        * Programming with Arduino: Developed software solutions to optimize the performance and accuracy of the electronic consumption monitoring system.
        * C++: Utilized programming skills to enhance the functionality and efficiency of the developed meter.
        
        This internship at the Information Processing Laboratory, Université Hassan II Mohammedia, provided a valuable opportunity to bridge theoretical knowledge with practical application. The successful development of the smart electronic consumption meter reflects my ability to contribute to innovative projects through a combination of electrical engineering and programming skills.
            
         """,
        "Electronics · Programming with Arduino · C++"
    )
        

def set_sidbar():
    with st.sidebar:
        st.title("SUMMARY")
        st.subheader("""
            With a Master's degree in Data Processing, specializing in deep learning, I bring a blend of theoretical expertise and practical application in Computer Vision and Natural Language Processing. My hands-on experiences, including a Google-prize-winning internship with Orange Morocco and a role as a Data Scientist at AUV Team Tomkyle in Germany, have honed my skills in real-world data science applications.
        """)

        st.write("\n \n ")
        st.title("LANGUAGES")
        st.progress(85 , "🇬🇧 English")
        st.progress(90 , "🇫🇷 French")
        st.progress(100 , "🇸🇦 Arabic")
        st.progress(20 , "🇩🇪 German")

        st.write("\n \n ")
        st.title("extracurricular activities")
        st.markdown("""
                * the vice president of the Robotics and IoT club of Hassan 2 University
                * Programming trainer in the CRIoT club of Hassan 2 University
                    """)



def skills():
    # --- SKILLS ---
    st.write('\n')
    
    # Adding a title for the skills section with an emoji
    st.title('Skills 🛠️')

    # Embedding badges in Markdown
    badges = """
    [![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white&logoSize=2)](#)
    [![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white&logoSize=2)](#)
    [![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&logoSize=2)](#)
    [![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white&logoSize=2)](#)
    [![NLP](https://img.shields.io/badge/-NLP-4db33d?style=for-the-badge&logoSize=2)](#)
    [![Data Viz](https://img.shields.io/badge/-Data%20Viz-ff69b4?style=for-the-badge&logoSize=2)](#)
    [![YOLO](https://img.shields.io/badge/-YOLO-990000?style=for-the-badge&logoSize=2)](#)
    [![scikit-learn](https://img.shields.io/badge/-scikit%20learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white&logoSize=2)](#)
    [![Keras](https://img.shields.io/badge/-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white&logoSize=2)](#)
    [![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white&logoSize=2)](#)
    [![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white&logoSize=2)](#)
    [![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white&logoSize=2)](#)
    [![Seaborn](https://img.shields.io/badge/-Seaborn-4E5180?style=for-the-badge&logo=python&logoColor=white&logoSize=2)](#)
    [![Langchain](https://img.shields.io/badge/-Langchain-2b77b2?style=for-the-badge&logoSize=2)](#)

    [![C](https://img.shields.io/badge/-C-A8B9CC?style=for-the-badge&logo=c&logoColor=white&logoSize=2)](#)
    [![C++](https://img.shields.io/badge/-C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white&logoSize=2)](#)
    [![Matlab](https://img.shields.io/badge/-Matlab-0076A8?style=for-the-badge&logo=mathworks&logoColor=white&logoSize=2)](#)
    [![CNN](https://img.shields.io/badge/-CNN-FF0000?style=for-the-badge&logoSize=2)](#)
    [![RNN](https://img.shields.io/badge/-RNN-FF0000?style=for-the-badge&logoSize=2)](#)
    [![LLM](https://img.shields.io/badge/-LLM-00599C?style=for-the-badge&logoSize=2)](#)
    [![GANs](https://img.shields.io/badge/-GANs-FF69B4?style=for-the-badge&logoSize=2)](#)

    [![Git](https://img.shields.io/badge/-Git-F05032?style=for-the-badge&logo=git&logoColor=white&logoSize=2)](#)
    [![Amazon](https://img.shields.io/badge/-Amazon-232F3E?style=for-the-badge&logo=amazon&logoColor=white&logoSize=2)](#)
    [![Docker](https://img.shields.io/badge/-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white&logoSize=2)](#)
    [![SQL/Mysql](https://img.shields.io/badge/-SQL/Mysql-4479A1?style=for-the-badge&logo=mysql&logoColor=white&logoSize=2)](#)

    [![Arduino](https://img.shields.io/badge/-Arduino-00979D?style=for-the-badge&logo=arduino&logoColor=white&logoSize=2)](#)
    [![Raspberry Pi](https://img.shields.io/badge/-Raspberry%20Pi-A22846?style=for-the-badge&logo=raspberry%20pi&logoColor=white&logoSize=2)](#)
    [![Linux](https://img.shields.io/badge/-Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white&logoSize=2)](#)
    """


    # Adding badges to the Streamlit app
    st.markdown(badges)


def projects():
    # --- Projects & Accomplishments ---

    ML = {
        "🏆 Polynomial Regression Project From Scrach": "https://github.com/mouraffa/polynomial_regression_from_scratch",
        "🏆 Linear Regression from Scratch using C++": "https://github.com/mouraffa/LinearRegression_CPP_FromScratch",
        "🏆 K-Nearest Neighbors (KNN) Implementation from Scratch": "https://github.com/mouraffa/KNN-From-Scratch-Iris-Classifier",
        "🏆 Breast Cancer Classification using SVM": "https://github.com/mouraffa/Cancer_Classification_SVM",
    }
    ML_descreption = {
        "🏆 Polynomial Regression Project From Scrach": "This project implements a polynomial regression algorithm from scratch using Python. The model is capable of fitting a polynomial to a given set of data points and is tested on synthetic data generated within the project.",
        "🏆 Linear Regression from Scratch using C++": "This repository contains a simple implementation of linear regression from scratch in C++ and includes Python scripts for data generation and error visualization. Linear regression is a fundamental machine learning algorithm used for modeling the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.",
        "🏆 K-Nearest Neighbors (KNN) Implementation from Scratch": "This project is an implementation of the K-Nearest Neighbors (KNN) algorithm from scratch, applied to the classic Iris dataset for classification. It showcases the foundational concepts of KNN, provides comprehensive visualization, and offers insights into the underlying mechanics of the algorithm.",
        "🏆 Breast Cancer Classification using SVM": "This repository contains a Jupyter notebook that demonstrates the classification of breast tumors into malignant or benign categories using the Support Vector Machines (SVM) algorithm. The project covers various stages of a machine learning pipeline, including data exploration, visualization, model training, and evaluation.",
    } 

    DL = {
        "🏆 From Theory to Implementation: A Comprehensive Perceptron Project with One Hidden Layer and Softmax Activation": "https://github.com/mouraffa/Perceptron-with-one-hidden-layer",
        "🏆 Handwritten_Digits_Classifier": "https://github.com/mouraffa/Handwritten_Digits_Classifier",
        
    }
    DL_descreption = {
        "🏆 From Theory to Implementation: A Comprehensive Perceptron Project with One Hidden Layer and Softmax Activation": "This project is an implementation of a Perceptron with one hidden layer and softmax function. The purpose of this project is to build a neural network that can classify input data into different categories.",
        "🏆 Handwritten_Digits_Classifier": "A neural network-based approach for handwritten digit classification using the MNIST dataset. Explore different models, techniques, and architectures to achieve accurate digit recognition.",

    }

    CNN = {
        "🏆 Real Time Object Detection Web App with YOLOv5 and Streamlit": "https://github.com/mouraffa/RealTime-Object-Detection-YOLOv5",
        "🏆 CIFAR10 Image Classification using Artificial Neural Network (ANN) ansd Convolutional Neural Network(CNN)": "https://github.com/mouraffa/CIFAR10-Image-Classification-Comparing-CNN-and-ANN-Models",
        "🏆 Zhang's Camera Calibration: Lens Distortion Correction with Math Insights": "https://github.com/mouraffa/CameraCalibration-using-the-Zhang-s-methode",
        "🏆 StereoVision-DepthEstimation": "https://github.com/mouraffa/StereoVision-DepthEstimation",
        "🏆 Stereo Disparity Map Generator" : "https://github.com/mouraffa/Stereo-Disparity-Map-Generator",
        "🏆 Face-Detection-using-arduino" : "https://github.com/mouraffa/Face-Detection-using-arduino" ,
    }

    CNN_descreption = {
        "🏆 Real Time Object Detection Web App with YOLOv5 and Streamlit" : "In this project, I trained the YOLOv5 model for object detection and created a Streamlit web application to perform object detection on uploaded images. The YOLOv5 model was trained to detect various objects, and the trained model is integrated into a user-friendly web interface using Streamlit.",
        "🏆 CIFAR10 Image Classification using Artificial Neural Network (ANN) ansd Convolutional Neural Network(CNN)" : "This project demonstrates image classification using the CIFAR10 dataset in TensorFlow. It compares the performance of a simple Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN) for this task. The purpose is to understand why CNNs are preferred over ANNs for image classification.",
        "🏆 Zhang's Camera Calibration: Lens Distortion Correction with Math Insights" : "This program implements a camera calibration algorithm based on Zhang's method. It allows you to calibrate your camera, estimate its intrinsic and extrinsic parameters, and correct for lens distortion effects. The calibration parameters obtained can be used for various computer vision tasks, such as 3D reconstruction, camera pose estimation, and image rectification.",
        "🏆 StereoVision-DepthEstimation" : "The StereoVision-DepthEstimation project utilizes computer vision techniques to estimate real-time depth from a webcam. It employs stereo vision and facial feature tracking, providing accurate depth measurements. Explore the world of 3D perception and unlock new possibilities in computer vision.",
        "🏆 Stereo Disparity Map Generator" : "This code generates a disparity map using the StereoSGBM algorithm. The disparity map represents the difference in pixel coordinates between corresponding points in a pair of stereo images. It provides depth information for each pixel, which can be used for tasks such as 3D reconstruction, object detection, and scene understanding.",
        "🏆 Face-Detection-using-arduino" : "It's a simple program for detecting faces and turning on a LED. If there is a detection then the green LED is on otherwise, the yellow LED lights up",
        
    }
    
    NLP = {
        "🏆 Sentimental_Analysis_LogRegClassification": "https://github.com/mouraffa/Sentimental_Analysis_LogRegClassification",
        "🏆 Skimlit-NLP-Model-for-Sentence-Classification-in-Paper-Abstracts" : "https://github.com/mouraffa/Skimlit-NLP-Model-for-Sentence-Classification-in-Paper-Abstracts"
    }

    NLP_descreption = {
        "🏆 Sentimental_Analysis_LogRegClassification": "This project is a Twitter sentiment analysis tool that uses logistic regression to classify tweets as either positive or negative. The project is structured into three main components: function definitions, model training, and user interaction for testing the model or viewing the accuracy plot.",
        "🏆 Skimlit-NLP-Model-for-Sentence-Classification-in-Paper-Abstracts" : "In this NLP project, I endeavored to replicate the paper {Neural Networks for Joint Sentence Classification in Medical Paper Abstracts} using the PubMed 200k RCT dataset. Leveraging my skills in the NLP field and knowledge of TensorFlow, I aimed to create an NLP model capable of classifying abstract sentences into specific roles (e.g., objective, methods, results) to facilitate efficient literature skimming for researchers.",
        
    }

    Generative_AI = {
        "🏆 Generative-Adversarial-Networks-GANs-for-Anime-Image-Generation": "https://github.com/mouraffa/Generative-Adversarial-Networks-GANs-for-Anime-Image-Generation",
        "🏆 LLM-gemini-streamlit-app-end-to-end-llm" : "https://github.com/mouraffa/LLM-gemini-streamlit-app-end-to-end-llm"
    }

    Generative_AI_descreption = {
        "🏆 Generative-Adversarial-Networks-GANs-for-Anime-Image-Generation": "In this project, I created a Generative Adversarial Networks (GANs) model for artistic image synthesis using PyTorch. The model consists of a generator and discriminator trained on the Anime Face Dataset from either GitHub or Kaggle. The training involved 100 epochs, and the progress is visualized in a video showcasing generated images for each epoch.",
        "🏆 LLM-gemini-streamlit-app-end-to-end-llm" : "This project implements a Streamlit web application powered by the Gemini API for both question/chat and image generation. Users can choose between the two options and interact with the Gemini model to receive responses.",
        
    }

     
    st.write('\n')
    st.title("Projects & Accomplishments")
    st.write("---")
    
    
    tab1 , tab2 , tab3 , tab4  , tab5= st.tabs(["traditional machine learning" , "Deep learning" , "computer vision" , "NLP" , "Generative AI"])

    with tab1:
        for project, link in ML.items():
            with st.expander(project):
                st.subheader(project)
                st.markdown(ML_descreption[project])
                st.link_button("📁 Go to GitHub 🌐" , link)

    
    with tab2:
        for project, link in DL.items():
            with st.expander(project):
                st.subheader(project)
                st.markdown(DL_descreption[project])
                st.link_button("visite in GitHub" , link)

    with tab3:
        for project, link in CNN.items():
            with st.expander(project):
                st.subheader(project)
                st.markdown(CNN_descreption[project])
                st.link_button("visite in GitHub" , link)

    with tab4:
        for project , link in NLP.items():
            with st.expander(project):
                st.subheader(project)
                st.markdown(NLP_descreption[project])
                st.link_button("visite in GitHub" , link)
        
    with tab5:
        for project , link in Generative_AI.items():
            with st.expander(project):
                st.subheader(project)
                st.markdown(Generative_AI_descreption[project])
                st.link_button("visite in GitHub" , link)


def certaficate():
    def display_certificate(image_path, name, issuer, date , linke):
        col1 , col2 = st.columns(2)
        with col1:
            st.image(image_path, caption=name, use_column_width=True)
        with col2:
            st.subheader(name)
            st.write(f"**Issuer:** {issuer}")
            st.write(f"**Date Earned:** {date}")
            st.write("---")
            st.link_button("🏅Show credential📜" , linke)

    # Replace these with your actual certificate data
    certificates_data = [
        #{"image_path": "images/python.png", "name": "Python for Data Science, AI & Development", "issuer": "Coursera", "date": "jul 2022" , "linke" : "https://www.coursera.org/account/accomplishments/certificate/GU3RS3TP4FVR"},
        {"image_path": "images/NLP.jpeg", "name": "Natural Language Processing with Classification and Vector Spaces", "issuer": "Coursera & DeepLearning.AI", "date": "Nov 2023" , "linke" : "https://coursera.org/share/b3806f4a147982958bc0c44c5501db45"},
        {"image_path": "images/yolo.jpg", "name": "YOLO: Custom Object Detection & Web App in Python", "issuer": "Udemy", "date": "jul 2023" , "linke" : "https://www.udemy.com/certificate/UC-fe981680-0d95-4381-a600-28b1bdd3c4fa/"},
        {"image_path": "images/python.png", "name": "Python for Data Science, AI & Development", "issuer": "Coursera", "date": "jul 2022" , "linke" : "https://www.coursera.org/account/accomplishments/certificate/GU3RS3TP4FVR"},
        {"image_path": "images/IOT.png", "name": "Introduction and Programming with IoT Boards", "issuer": "Coursera", "date": "jul 2022" , "linke" : "https://www.coursera.org/account/accomplishments/certificate/F6NW57PNER9W"},
        {"image_path": "images/Matlab.png", "name": "MATLAB Onramp", "issuer": "MatWorks", "date": "Oct 2022" , "linke" : "https://matlabacademy.mathworks.com/progress/share/certificate.html?id=b5eb38b5-4dc2-4d19-977c-1a5d35cf5619&"},
        {"image_path": "images/streamlit.jpg", "name": "Learn & Deploy Data Science Web Apps with Streamlit", "issuer": "Udemy", "date": "Nov 2023" , "linke" : "https://www.udemy.com/certificate/UC-d5cf44e4-04e6-4cd4-8afe-c9f2294b7b35/"},
        {"image_path": "images/sql.jpg", "name": "MySQL for Data Analytics and Business Intelligence", "issuer": "Udemy", "date": "Aug 2022" , "linke" : "https://www.udemy.com/certificate/UC-84b80404-48cc-429e-bae4-077c52cc1b0e/"},

        
        # Add more certificates as needed
    ]

    st.title("My Certificates")

    for certificate in certificates_data:
        display_certificate(certificate["image_path"], certificate["name"], certificate["issuer"], certificate["date"] , certificate["linke"])





def main():

    # --- PATH SETTINGS ---
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    css_file = current_dir / "styles" / "main.css"
    resume_file = current_dir / "assets" / "CV_Mouraffa_Youssef.pdf"
    profile_pic = current_dir / "assets" / "profile-pic-3.png"

    # --- GENERAL SETTINGS ---
    PAGE_TITLE = "Digital CV | Mouraffa Youssef"
    PAGE_ICON = "assets/Icone.png"
    NAME = "Youssef Mouraffa"
    DESCRIPTION = """
    Information Processing MS Graduate | Researcher @ AUV Team Tomkyle | Specializing in Computer Vision and NLP
    """
    EMAIL = "mouraffayoussef@gmail.com"

    SOCIAL_MEDIA = {
        "XING": "https://www.xing.com/profile/Youssef_Mouraffa/cv?expandNeffi=true",
        "LinkedIn": "https://www.linkedin.com/in/youssef-mouraffa-316663201/",
        "GitHub": "https://github.com/mouraffa",
        #"Twitter": "https://twitter.com",
    }

    SOCIAL_MEDIA_logs = [ "images/XING_50.png" , "images/LinkdIn_50.png" , "images/GitHub_50.png" , "images/other_50.png" ]

    
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON , layout="wide")
    
    # --- set the bar side ---
    set_sidbar()
    

    # --- LOAD PDF & PROFIL PIC ---
    with open(resume_file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    profile_pic = Image.open(profile_pic)


    # --- Hero section ---

    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image(profile_pic, width=300)

    with col2:
        st.title(NAME)
        st.subheader(DESCRIPTION)
        st.download_button(
            label=" 📄 Download Resume",
            data=PDFbyte,
            file_name=resume_file.name,
            mime="application/octet-stream",
        )
        st.write("📫", EMAIL)


    # --- SOCIAL LINKS ---
    
    st.write('\n')
    cols = st.columns(len(SOCIAL_MEDIA))
    for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
        cols[index].image(SOCIAL_MEDIA_logs[index])
        cols[index].link_button(platform , link)

    
    # --- Experience section---
    experience()

    # --- Education section---
    education()

    # ---skills section ---
    skills()

    # ---Project section ---
    projects()

    certaficate()



if __name__ == "__main__":
    main()
