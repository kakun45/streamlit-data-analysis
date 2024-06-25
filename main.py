# to run $ streamlit run main.py
import pandas as pd
import streamlit as st
import time

# Custom css for a header fix
st.markdown(
    """
    <style>
    .css-1y4p8pa {
    padding: 0rem 1rem 10rem;
    }
    .e13qjvis2 {
    background: #D2D3C7;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Custom CSS to make the header image full-width
st.markdown(
    """
    <style>
    .header-image {
        width: 100vw;
        height: 16rem;
        margin-left: -50vw;
        margin-right: -50vw;
        position: relative;
        left: 50%;
        right: 50%;
        object-fit: cover; /* Ensures the image covers the entire container */
        object-position: 50% 43%; /* Aligns the image to the bottom center */
        top: 0; /* Removes any top margin */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the full-width header image
st.markdown(
    """
        <img src="https://images.unsplash.com/photo-1496588152823-86ff7695e68f?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" class="header-image">
    </div>
    """,
    unsafe_allow_html=True
)
# Rest of Streamlit content

header = st.container()
dataset = st.container()
features = st.container()
map = st.container()

@st.cache_data
def get_csv_data(filename):
    try:
        mta_df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        st.error(f'File not found: {csv_filepath}')
    except pd.errors.ParserError:
        st.error(f'Error parsing the file: {csv_filepath}')
    else:
        return mta_df

with header:
    # Display media
    # st.image('https://images.unsplash.com/photo-1496588152823-86ff7695e68f?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')
    st.title("Welcome to my app")
    long_text = '''
    In this project I will try to find if there is an influence of proximity to a subway station on Rental Prices in all 5 boroughs of NYC and some right
    across the river part of NJ'''
    st.markdown(long_text)

with dataset:
    st.header("Datasets")  # the same as st.markdown(' ## heading2')
    st.subheader("Rental Prices Zillow 2020, and MTA stations")
    long_text = '''I obtained the Prices dataset from Kaggle https://www.kaggle.com/datasets/sab30226/zillow-rents-2020 Subway info from MTA Public data
    https://new.mta.info/open-data \n For a header image thanks to: unsplash.com'''
    st.markdown(long_text)
    # Display data
    csv_filepath = './data/grouped_df.csv'
    df = get_csv_data(csv_filepath)
    st.markdown('### MTA stations')
    st.table(df.head())  # full width, for narrower view:  st.dataframe(mta_df.head())


with features:
    st.header("The feature I created")
    st.text("In this project I discoverd... That's how I calculate...")
    st.code('for i in range(n): print(1)')


with map:
    st.header("NYC Map visualization")
    st.text("On this map you're going to see...")

    # Multiselect
    st.multiselect('Choose data to plot', ['Rent Prices', 'Subway Stations'])

    # Slider
    st.select_slider('Choose Rent Price', ['min', 'median', 'max'])

    col1, col2, col3 = st.columns(3)
    with col2:
        # button
        btn = st.button('Generate the Map')

        if btn:
            with st.spinner('Creating a Map according to your reqest'):
                time.sleep(3)
            st.success('Map is ready!')


