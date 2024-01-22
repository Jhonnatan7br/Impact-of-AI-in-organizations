import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title('Publications per Year - Line Graph')

# File uploader to allow users to upload their own CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Processing the data to count the number of publications per year
    publications_per_year = data['Year'].value_counts().sort_index()

    # Creating a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(publications_per_year.index, publications_per_year.values, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Number of Publications')
    plt.title('Number of Publications per Year')
    plt.grid(True)
    st.pyplot(plt.gcf())
else:
    st.write("Please upload a CSV file to display the graph.")
