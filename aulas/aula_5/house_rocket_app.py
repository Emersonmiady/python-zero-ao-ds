import pandas as pd
import streamlit as st

st.title('House Rocket Company')
st.markdown('Welcome to House Rocket Data Analysis')
st.header('Load Data')

# read data
@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_csv(path)
    data['date'] = pd.to_datetime(data['date'])
    return data

# load data
path = 'C:/Users/Emerson/Documents/Jupyter Projects/pthon_do_zero_ao_ds/datasets/kc_house_data.csv'
df = get_data(path)