import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import numpy as np


diabetes_model=pickle.load(open("C:/Users/KIIT/Desktop/FP/multiple-disease-prediction-final/models/diabetes.pkl",'rb'))
heart_model=pickle.load(open("C:/Users/KIIT/Desktop/FP/multiple-disease-prediction-final/models/heart.pkl",'rb'))
cancer_model=pickle.load(open("C:/Users/KIIT/Desktop/FP/multiple-disease-prediction-final/models/cancer.pkl",'rb'))
kidney_model=pickle.load(open("C:/Users/KIIT/Desktop/FP/multiple-disease-prediction-final/models/kidney.pkl",'rb'))
liver_model=pickle.load(open("C:/Users/KIIT/Desktop/FP/multiple-disease-prediction-final/models/liver.pkl",'rb'))


with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System in ML", ["Diabetes Prediction", 'Heart Prediction','Cancer Prediction','Kidney Prediction',
        'Liver Prediction','Pneumonia Prediction'], 
        icons=['house', 'heart','cancer','kidney'], default_index=0)
    selected
if(selected=='Diabetes Prediction'):
    st.title("Diabetes Prediction Using ML")
    col1,col2,col3=st.columns(3)
    with col1:
        pregnancies=st.text_input("Number of Pregnency")
    with col2:
         Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)





if(selected=='Heart Prediction'):
    st.title("Heart Disease Prediction Using ML")
    
    age=st.text_input("Enter Your Age",placeholder="Age eg: 37")
    sex=st.text_input("Enter your sex",placeholder="sex (Male:1, female:0)")
    cp=st.text_input("chest pain",placeholder="chest pain type eg:1")
    trestbps=st.text_input("Resting blood pressure in mm",placeholder="eg:130")
    chol=st.text_input("Serum cholestoral in mg/dl",placeholder="eg:250")
    fbs=st.text_input("Fasting blood sugar",placeholder="(1 = true; 0 = false)")
    restecg=st.text_input("resting electrocardiographic results",placeholder="eg:1")
    thalach=st.text_input("Maximum heart rate",placeholder="eg:187")
    exang=st.text_input("exercise induced angina",placeholder="(1 = yes; 0 = no)")
    oldpeak=st.text_input("ST depression induced by exercise",placeholder="eg:3.5")
    slope=st.text_input("slope of the peak exercise ST segment",placeholder="eg:0")
    ca=st.text_input("number of major vessels (0-3)",placeholder="eg:0")
    thal=st.text_input("thal")

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)









if(selected=='Cancer Prediction'):
    st.title("Cancer Prediction Using ML")

    rad=st.text_input("Radius mean",placeholder="eg:20.57")
    text=st.text_input("Texture mean",placeholder="eg:17.77")
    per=st.text_input("Perimeter mean",placeholder="eg: 132.90")
    area=st.text_input("Area mean",placeholder="eg: 1326.0")
    smooth=st.text_input("Smoothness mean",placeholder="eg: 0.08474")
    compact=st.text_input("Compactness mean",placeholder="eg: 0.07874")
    concave=st.text_input("Concavity mean",placeholder="eg: 0.0869")
    conpt=st.text_input("Concave points mean",placeholder="eg: 0.07017")
    symm=st.text_input("Symmetry mean",placeholder="eg: 0.1812")
    radse=st.text_input("Radius_se",placeholder="eg: 0.05667")
    perse=st.text_input("Perimeter se",placeholder="eg: 3.398")
    arease=st.text_input("Area se",placeholder="eg: 74.08")
    compactse=st.text_input("Compactness se mean",placeholder="eg: 0.01308")
    conase=st.text_input("Concavity se",placeholder="eg: 0.01860")
    conpse=st.text_input("Concave points se",placeholder="eg: 0.01340")
    frac=st.text_input("fractal_dimebsional_se",placeholder="eg: 0.003532")
    radwor=st.text_input("Radius worst",placeholder="eg: 24.99")
    textwor=st.text_input("Texture worst",placeholder="eg: 23.41")
    perimw=st.text_input("Perimeter worst",placeholder="eg: 158.80")
    areaw=st.text_input("Area worst",placeholder="eg: 1956.0")
    smooth_wor=st.text_input("Smoothness worst",placeholder="eg: 0.1238")
    com=st.text_input("Compactness worst",placeholder="eg: 0.1866")
    cons=st.text_input("Concavity worst",placeholder="eg: 0.2416")
    alpha=st.text_input("Concave points worst",placeholder="eg: 0.1860")
    beta=st.text_input("Symmetry Worst",placeholder="eg: 0.2750")
    gamma=st.text_input("Fractal dimension worst",placeholder="eg: 0.08902")
    
    # code for Prediction
    cancer_diagnosis = ''

    # creating a button for Prediction

    if st.button('Cancer Disease Test Result'):

        user_input = [rad,text,per,area,smooth,compact,concave,conpt,symm,radse,perse,arease,compactse,conase,conpse,frac,radwor,textwor,perimw,areaw,smooth_wor,com,cons,alpha,beta,gamma]

        user_input = [float(x) for x in user_input]

        cancer_prediction = heart_model.predict([user_input])

        if cancer_prediction[0] == 1:
            cancer_diagnosis = 'Person is having Cancer'
        else:
            cancer_diagnosis = 'The person does not have any Cancer disease'

    st.success(cancer_diagnosis)














if(selected=='Kidney Prediction'):
    st.title("Kidney Prediction Using ML")    
    v1=st.text_input("Age",placeholder="eg: 48")
    v2=st.text_input("Blood pressure",placeholder="eg: 70.0")    
    v3=st.text_input("Al",placeholder="eg: 4.0")
    v4=st.text_input("Su",placeholder="eg: 0.0")
    v5=st.text_input("Rbc",placeholder="eg: 0")
    v6=st.text_input("Pc",placeholder="eg: 1")
    v7=st.text_input("Pcc",placeholder="eg: 1")
    v8=st.text_input("Ba",placeholder="eg: 0")
    v9=st.text_input("Bgr",placeholder="eg: 117.0")
    v10=st.text_input("Bu",placeholder="eg: 56.0")
    v11=st.text_input("Sc",placeholder="eg: 308")
    v12=st.text_input("Pot",placeholder="eg: 2.5")
    v13=st.text_input("Wc",placeholder="eg: 6700")
    v14=st.text_input("Htn",placeholder="eg: 1")
    v15=st.text_input("Dm",placeholder="eg: 0")
    v16=st.text_input("Cad",placeholder="eg: 0")
    v17=st.text_input("Pe",placeholder="eg: 1")
    v18=st.text_input("Ane",placeholder="eg: 1")

    # code for Prediction
    kidney_diagnosis = ''

    # creating a button for Prediction

    if st.button('Kidney Disease Test Result'):

        user_input = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18]

        user_input = [float(x) for x in user_input]

        kidney_prediction = kidney_model.predict([user_input])

        if kidney_prediction[0] == 1:
            kidney_diagnosis = 'Person is having Kidney problem'
        else:
            kidney_diagnosis = 'The person does not have any Kidney Problem'

    st.success(kidney_diagnosis)


if(selected=='Liver Prediction') :
    st.title("Liver Dieases Prediction System In Ml")

    n1=st.text_input(" Enter Age",placeholder="eg: 62")
    n2=st.text_input("Total Bilirubin",placeholder="eg: 7.3")
    n3=st.text_input(" Direct Bilirubin",placeholder="eg: 4.1")
    n4=st.text_input("Alkaline_Phosphotase",placeholder="eg: 490")
    n5=st.text_input("Alamine_Aminotransferase",placeholder="eg: 70")
    n6=st.text_input("Aspartate_Aminotransferase",placeholder="eg: 68")
    n7=st.text_input("Total Protiens",placeholder="eg: 7.0")
    n8=st.text_input("Albumin",placeholder="eg: 3.3")
    n9=st.text_input("Albumin_and_Globulin_Ratio",placeholder="eg: 0.89")
    n10=st.text_input("Gender",placeholder="(male: 1,Female: 0)")

    # code for Prediction
    liver_diagnosis = ''

    # creating a button for Prediction

    if st.button('Liver Disease Test Result'):

        user_input = [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10]

        user_input = [float(x) for x in user_input]

        liver_prediction = kidney_model.predict([user_input])

        if liver_prediction[0] == 1:
            liver_diagnosis = 'Person is having Liver problem'
        else:
            liver_diagnosis = 'The person does not have any Liver Problem'

    st.success(liver_diagnosis)


####### here trying for pneumonia


if(selected=='Pneumonia Prediction'):
    model=load_model("C:/Users/KIIT/Desktop/Final_Project_2/Multi_Disease_Predictor/models/pneumonia.h5")   # here model i gave is of Final_project_2 folder.you can try with fp
    st.title("Pneumonia Prediction Using Chest X Ray")
    def preprocess_image(image):
        img = image.convert('L').resize((36, 36))
        img_array = np.asarray(img)
        img_array = img_array.reshape((1, 36, 36, 1))
        img_array = img_array / 255.0
        return img_array
    def predict(image):
        img_array = preprocess_image(image)
        prediction = np.argmax(model.predict(img_array)[0])
        return prediction
    
    

    st.write("Upload an image for pneumonia detection.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            try:
                prediction = predict(image)
                if prediction == 0:
                    st.success("Prediction: Normal")
                else:
                    st.success("Prediction: Pneumonia")
            except:
                st.write("An error occurred while processing the image.")









