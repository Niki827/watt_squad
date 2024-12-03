import streamlit as st
import requests
import pandas as pd

st.title("My Cost Savings App")

flexibility = st.number_input("Give your flexibility degree: ", value=0)


if st.button("Predict Your Cost Savings"):

    url = 'http://localhost:8000/predict'
    url_prod = 'https://watt-squad-mvp-image-1071061957527.europe-west1.run.app'

    params = {
        "flexibility": flexibility
        }

    # response = requests.get(url, params=params)
    response = requests.get(f'http://127.0.0.1:8000/predict?flexibility_degree={flexibility}')


    cost_pred = response.json()
    cost = cost_pred["df"]

    df = pd.DataFrame(cost)

    st.write(df.head())
    # Set the index (optional, for better display)
    df.set_index('timestamp', inplace=True)
    x = df['timestamp']

    # cost_wos = cost["Total Cost Without Shifting"]
    # cost_ws = cost['Total Cost With Shifting']
    # cost_s = cost['Cost Savings']
    # #'Total Cost With Shifting': 7402.1394766528065, 'Cost Savings': 1218.2230921669852}


    # # cost = fare_pred["fare"]

    # # st.write(f"Your predicted cost savings are NOK{round(fare, 2)}")

    # st.write(f"Total Cost Without Shifting: {cost_wos}")
    # st.write(f"Total Cost With Shifting: {cost_ws}")
    # st.write(f"Cost Savings: {cost_s}")

    # 'Total Cost Without Shifting'
