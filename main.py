# to run $ streamlit run main.py
import folium
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit_folium import folium_static
import time

# Custom css to remove space atop of a header image; default was 6 rem on top
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


# Load data Fn
@st.cache_data
def get_csv_data(filename):
    try:
        new_df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        st.error(f'File not found: {csv_filepath}')
    except pd.errors.ParserError:
        st.error(f'Error parsing the file: {csv_filepath}')
    else:
        return new_df


with header:
    st.subheader('Author: Xeniya Shoiko, 2024 (c) All rights reserved')
    # Display media
    st.title("Welcome to my Streamlit app")
    long_text = '''
    In this project I will try to find if there is an influence of proximity to a subway station on Rental Prices in
    all 5 boroughs of NYC and some right across the river part of NJ which are easy accessible by Path service. 
    People sometimes refer to this part of NJ as the "sixth borough" of New York City or West West Village.'''
    st.markdown(long_text)

with dataset:
    st.header("Datasets")  # the same as st.markdown(' ## heading2')
    st.subheader("Rental Prices Zillow 2020, and MTA stations")
    long_text = '''I obtained the Prices dataset from Kaggle https://www.kaggle.com/datasets/sab30226/zillow-rents-2020 
    Subway info from MTA Public data https://new.mta.info/open-data
    For a header image thanks to: unsplash.com'''
    st.markdown(long_text)

    # Display Prices dataset
    csv_filepath = './data/zillow.csv'
    rent_df = get_csv_data(csv_filepath)
    st.markdown("**Zillow 2020 dataset** ")
    st.dataframe(rent_df.head())

    # Display MTA dataset
    csv_filepath = './data/grouped_df.csv'
    mta_df = get_csv_data(csv_filepath)
    st.markdown('**MTA stations (after processing)**')
    st.table(mta_df.head())  # full width, for narrower view:  st.dataframe(mta_df.head())

with features:
    st.header("Data Cleaning")
    long_text = """
    As with any data science project you embark on, there will always be some data cleaning needed to accomplish your goal.
    Let's get intimate with the dataset.
    """
    st.markdown(long_text)

    st.markdown(
        "Utilizing `.unique()` to filter unique values on one col to check what types of homes are available here:")
    st.code("df['columnName'].unique()")


    # Cache the unique home types
    @st.cache_data
    def get_unique_by_col(df, col):
        return df[col].unique()


    # Display the unique 'homeType' -s
    unique_home_types = get_unique_by_col(rent_df, 'homeType')
    st.write("- **Unique 'homeType':** ", str(unique_home_types), ";")
    st.write("- **Unique states** are present: ", str(get_unique_by_col(rent_df, 'state')), "I'll keep NY, and NJ;")

    st.write("""I gradually apply each step passing a result dataFrame from one function to another as a part of a cleaning process. 
    I DO NOT manipulate the original dataset. Next, I remove rows with missing values on 'homeStatus' column and find all 
             unique values of the column after the drop:""")


    @st.cache_data
    def dropna_unique_home_stat(df):
        df2 = df.dropna(subset=['homeStatus'])
        return df2


    nona_unique_home_st = dropna_unique_home_stat(rent_df)
    unique_home_status = get_unique_by_col(nona_unique_home_st, 'homeStatus')
    st.write("-  **Unique 'homeStatus':** ", str(unique_home_status), "\nI'll keep 'FOR_RENT' only from this step.")


    @st.cache_data
    def select_cols_rows(df):
        new_df = df[
            ['yearBuilt', 'homeStatus', 'longitude', 'latitude', 'price', 'city', 'state', 'postal_code', 'bedrooms',
             'bathrooms', 'area']]
        # df drop anything that is not NY or NJ in state col and that is not FOR_RENT in homeStatus
        new_df = new_df.drop(new_df[~new_df['state'].isin(['NY', 'NJ'])].index)
        new_df = new_df.drop(new_df[~new_df['homeStatus'].isin(['FOR_RENT'])].index)
        return new_df


    st.subheader("Let's **drop some** columns and rows:")
    st.write(
        ">> specifically: `df.drop()` anything that is not NY or NJ in 'state' col and that is not FOR_RENT in 'homeStatus' col")
    st.code("""new_df = new_df.drop(new_df[~new_df['state'].isin(['NY', 'NJ'])].index), 
        \nnew_df = new_df.drop(new_df[~new_df['homeStatus'].isin(['FOR_RENT'])].index)""")
    st.markdown("**How the dataset looks now**")
    selected_rent_df = select_cols_rows(nona_unique_home_st)
    st.dataframe(selected_rent_df.head())


    # yearBuilt plot
    # how many unique values are in 'yearBuilt' col, put the unique vals into bins of 12 and plot the bins
    # Cache the plot generation function
    @st.cache_data
    def plot_year_built(df):
        unique_year_built_values = df['yearBuilt'].unique()
        num_unique_values = len(unique_year_built_values)
        st.write(f"There are {num_unique_values} unique values in the 'yearBuilt' column.")

        # Create bins of 12
        bins = np.append(np.arange(1800, 2021, 12), 2021)

        # Plot the bins
        fig, ax = plt.subplots()
        ax.hist(df['yearBuilt'], bins=bins)
        ax.set_xlabel('Year Built')
        ax.set_ylabel('Number of Units')
        ax.set_title('Distribution of Houses by Year Built')
        return fig


    st.write("""If you're as curious as I'm about the 'yearBuilt' of NYC estates, 
             I asked how many unique values are in 'yearBuilt' column, next I put the unique values into bins of 12 and plot the bins""")
    # or a  Button. Works too
    # btn_yearBuilt = st.button("Generate Plot on yearBuilt")
    # if btn_yearBuilt:
    with st.expander("Generate Plot on 'yearBuilt'"):
        fig = plot_year_built(selected_rent_df)
        st.pyplot(fig)

    st.subheader("Time to clean up the 'price' column!")
    st.write("""I created a function that takes a price string and removes unwanted characters at front `.startswith('$')`,
            converts it to a float `float(price_string.replace(',', ''))`, and adjusts the value based on the ending if
            it detects a `+` sign in $1000+ `.endswith('+/mo')` to add 0.99 and remove the rest:""")
    st.code("if price_string.startswith('$'): \n \tprice_string = price_string[1:] \n ")


    @st.cache_data
    def clean_price_col(df):

        def fix_price(price_string):
            """
            This function takes a price string and removes unwanted characters,
            converts it to a float, and adjusts the value based on the ending:
            """
            if isinstance(price_string, float):
                return price_string
            if price_string.startswith('$'):
                price_string = price_string[1:]
            if price_string.endswith('+/mo'):
                price_string = price_string[:-4]
                price_string = str(float(price_string.replace(',', '')) + .99)
            if price_string.endswith('/mo'):
                price_string = price_string[:-3]
            return float(price_string.replace(',', ''))

        df['price_fixed'] = df['price'].apply(fix_price)  # 'price_fixed' col now is float64
        return df


    st.markdown("**Prices fixed in a new column 'price_fixed'**")
    prices_cleaned = clean_price_col(selected_rent_df)
    st.dataframe(prices_cleaned.head())

    # OUTLIERS
    # @st.cache_data
    # def outliers_analyze(df):
    #     return new_df
    # Slider
    # st.select_slider('Choose Rent Price', ['min', 'median', 'max'])

with map:
    st.header("NYC Map visualization")

    @st.cache_data
    def generate_geo_map(df):
        """
        Plot a geographical map in folium of NJ and NY based on the lat and long circles with their price_fixed,
        color by price_fixed. Due to the default behavior of Streamlit's caching mechanism. When you interact with other
        widgets, Streamlit re-renders the entire app, potentially resetting the state of components that are not cached explicitly.
        To address this and ensure that the map remains cached and available even after interacting
        with other elements, you can modify your approach by using st.cache decorator for your map generation function
        and ensuring that the map is only generated once and remains available throughout the session.
        """
        # Create a map object with initial location and zoom level 12 for NYC only
        nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)

        # Add circles to the map for each row in the DataFrame
        for index, row in df.iterrows():
            circle = folium.Circle(location=[row['latitude'], row['longitude']],
                                   radius=row['price_fixed'] * 0.002,
                                   color='blue')
            # display infobox on hover
            tooltip = folium.Tooltip(f"Price: ${row['price_fixed']}")
            circle.add_child(tooltip)
            nyc_map.add_child(circle)

        # Display the map
        return nyc_map

    # Initialize session state for the map
    if 'nyc_map' not in st.session_state:
        st.session_state['nyc_map'] = None

    # Function to handle map generation
    def handle_map_generation():
        with st.spinner('Creating a Map according to your request...'):
            st.session_state['nyc_map'] = generate_geo_map(prices_cleaned)
        st.success('Map v1.0 is ready! Displaying...')

    st.subheader("Let's Plot the map of NYC!")

    # Button to generate the map
    if st.button('Generate the Map v1'):
        handle_map_generation()

    # Display the cached map if it exists
    if st.session_state['nyc_map']:
        folium_static(st.session_state['nyc_map'])
