import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model(picle file)
with open('Titanic_Logistic_Regre_model.pkl', 'rb') as file:
    model=pickle.load(file)

# Streamlit UI
st.title("Titanic Survival Prediction")

# Here we should give the inputs.
st.sidebar.header("Enter Passenger Details")
def Features():
    pclass=st.sidebar.selectbox('Passenger Class',(1,2,3))
    sex=st.sidebar.selectbox('Sex',('Male','Female'))
    age=st.sidebar.slider('Age',0,100,30)
    sibsp=st.sidebar.slider('Number of Siblings',0,10,0)
    parch=st.sidebar.slider('Number of Parents',0,10,0)
    fare=st.sidebar.slider('Fare',0,520,50)
    embarked=st.sidebar.selectbox('Boarding Location or Embarked Port',('C:Cherbourg,France','Q:Queenstown,Ireland','S:Southampton,England'))
    data={
        'Pclass':pclass,
        'Sex':1 if sex=='Male' else 0,
        'Age':age,
        'SibSp':sibsp,
        'Parch':parch,
        'Fare':fare,
        'Embarked':embarked
    }
    features=pd.DataFrame(data,index=[0])
    return features

# Collecting Data.
inp=Features()

# Feature Preprocessing
inp['Embarked'] = inp['Embarked'].map({'C:Cherbourg,France':0,'Q:Queenstown,Ireland':1,'S:Southampton,England':2})

# Printing Given data.
st.subheader('Passenger Details')
st.write(inp)

# Predictions.
if st.button('Predict'):
    prediction=model.predict(inp)
    result="Survived" if prediction[0]==1 else "Passenger Not Survived"
    st.subheader(f"Prediction : {result}")