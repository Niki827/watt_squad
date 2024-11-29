import streamlit as st
import requests

st.title("My Cost Savings App")

# st.write("Give your flexibility degree:")

flexibility = st.number_input("Give your flexibility degree: ", value=0)


# response = requests.get(f'http://127.0.0.1:8000/predict?flexibility_degree={flexibility_degree}'

if st.button("Predict Your Cost Savings"):

    url = 'http://localhost:8000/predict'
    # url = response = requests.get(f'http://127.0.0.1:8000/predict?

    params = {
        "flexibility": flexibility
        }

    # response = requests.get(url, params=params)
    response = requests.get(f'http://127.0.0.1:8000/predict?flexibility_degree={flexibility}')


    cost_pred = response.json()
    cost = cost_pred["df"][0]

    cost_wos = cost["Total Cost Without Shifting"]
    cost_ws = cost['Total Cost With Shifting']
    cost_s = cost['Cost Savings']
    #'Total Cost With Shifting': 7402.1394766528065, 'Cost Savings': 1218.2230921669852}


    # cost = fare_pred["fare"]

    # st.write(f"Your predicted cost savings are NOK{round(fare, 2)}")

    st.write(f"Total Cost Without Shifting: {cost_wos}")
    st.write(f"Total Cost With Shifting: {cost_ws}")
    st.write(f"Cost Savings: {cost_s}")

    # 'Total Cost Without Shifting'
