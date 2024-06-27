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
data_clean = st.container()
st_map = st.container()  # Container for the st.map()
plots = st.container()
map = st.container()  # Container for the Folium map for Prices
st_map_mta = st.container()  # Container for the Folium map for MTA stations

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
    long_text = '''I obtained the [Prices dataset](https://www.kaggle.com/datasets/sab30226/zillow-rents-2020) from Kaggle, 
    Subway info from [MTA Public data](https://new.mta.info/open-data) I use precleaned version here, the cleaning process of which deserves a whole another articlee!
    For a header image thanks to: unsplash.com'''
    st.markdown(long_text)

    # Display Prices dataset
    csv_filepath = './data/zillow.csv'
    rent_df = get_csv_data(csv_filepath)

    # Display MTA dataset
    csv_filepath = './data/grouped_df.csv'
    mta_df = get_csv_data(csv_filepath)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Zillow 2020 dataset** ")
    with col2:
        st.markdown("**MTA stations** (Cleaned)")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(rent_df.head())
    with col2:
        st.dataframe(mta_df.head())  # full width, for narrower view:  st.dataframe(mta_df.head())

with data_clean:
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

with st_map:
    st.header("NYC Map visualization of Prices")
    st.subheader("Let's add all those units for rent to the map of NYC!")
    st.map(prices_cleaned, zoom=10, use_container_width=False)
    st.markdown("Looks *good*. Or is it?! I need to see what kind of prices are there. Statistics to the resque.")

with plots:
    @st.cache_data
    def bar_plot(df):
        """
        Get the frequency of each price then Sort the price counts by price, plot, make a plot a screenwide
        """
        price_fixed = df['price_fixed'].astype(int)
        price_counts = price_fixed.value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(150, 10))
        price_counts.plot(kind='bar', ax=ax)  #to limit to 50 first places ...head(50).plot...
        ax.set_xlabel("Price")
        ax.set_ylabel("Count")
        ax.set_title("Frequency of Prices")
        return fig

    @st.cache_data
    def bar_plot_sorted(df):
        # plot a frequency of the price, make a plot figzie 30, sort by value in a price_fixed
        fig, ax = plt.subplots(figsize=(30, 2))
        df['price_fixed'].value_counts().head(40).plot(kind='bar', ax=ax)
        ax.set_title('Frequency of Price')
        return fig

    @st.cache_data
    def identify_outliers(df):
        """
        Analyzes df['price_fixed'] range occurrence, median, mean, outliers; to plot all these values
        """
        def format_as_financial(val):
            """
            Takes a float value and returns it formatted as a financial string.
            :param val: float
            :return: str
            """
            return f"${val:,.2f}"

        p_min, p_max = df['price_fixed'].min(), df['price_fixed'].max()
        price_range = (p_max - p_min)
        price_median = df['price_fixed'].median()
        price_mean = df['price_fixed'].mean()

        # Calculate the lower and upper bounds for outliers
        q1 = df['price_fixed'].quantile(0.25)
        q3 = df['price_fixed'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr  # decrese 1.5 based on domain knowlege?

        # Identify outliers
        outliers = df[~((df['price_fixed'] >= lower_bound) & (df['price_fixed'] <= upper_bound))]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Price Range:** {format_as_financial(price_range)}")
            st.markdown(f"**Median Price:** {format_as_financial(price_median)}")
            st.markdown(f"**Min Price:** {format_as_financial(p_min)}")
        with col2:
            st.markdown(f"**Number of Outliers:** {len(outliers)}")
            st.markdown(f"**Mean Price:** {format_as_financial(price_mean)}")
            st.markdown(f"**Max Price:** {format_as_financial(p_max)}")

        return price_range, price_median, price_mean, outliers

    @st.cache_data
    def plot_with_outliers(df):
        """
        Plot the price distribution with outliers using seaborn and display it in Streamlit.
        """
        # Create a new fig
        fig, ax = plt.subplots(figsize=[10, 3])
        # plot the prices dist
        sns.boxplot(x=df['price_fixed'], ax=ax)
        ax.set_xlabel('Monthly Rent, $')
        ax.set_ylabel('Value distribution')
        ax.set_title('Price Distribution with Outliers')
        return fig

    @st.cache_data
    def plot_without_outliers(df, outliers):
        """
        Plot the price distribution without outliers.
        Filters out outliers to plot calculated values in box plot shows whiskers.
        Uses a boolean indexing operation that filters out rows where the 'price_fixed' value is in the
        'outliers' DataFrame. Essentially, it removes the outliers.
        y-axis Label: (No label needed since a box plot doesn't traditionally represent frequency on the y-axis)
            # plt.ylabel('Value distribution')
        """
        fig, ax = plt.subplots(figsize=[10, 3])
        sns.boxplot(x=df[~df['price_fixed'].isin(outliers['price_fixed'])]['price_fixed'], ax=ax)  # bool ind
        # the width would indicate the range of values between Q1 and Q3 along the x-axis.
        ax.set_xlabel('Monthly Rent, $')
        ax.set_title('Price Distribution without Outliers')
        return fig

    @st.cache_data
    def plot_without_outliers_with_bedrooms(df):
        """
        Creates a boxplot of 'price_fixed', grouped by bedrooms.
        Filter the dataframe to include only rows where price_fixed is between 1150 and 6000
        """
        new_df_filtered = df[(df['price_fixed'] >= 1200) & (df['price_fixed'] <= 6000)]

        fig, ax = plt.subplots(figsize=[10, 6])
        new_df_filtered.boxplot(column='price_fixed', by='bedrooms', ax=ax)
        ax.set_xlabel('Bedrooms')
        ax.set_ylabel('Monthly Rent, $')
        ax.set_title('Price Distribution by Number of Bedrooms')
        return fig

    st.subheader("Analyze Monthly Rent")
    long_text = """
    This is unreadable, I know, but there are a lot of rows in this dataset. I display it just for a visual sense of data distribution. let's sort and check on the 
    first 50 rows that have any significance in their count. I want to point out the values that rounded are the ones which accumulate greater counts vs a rundom numer on a numberline, which makes sence for us as humans when intersting in decimal system to talk sbout prices. We're dealing with humans in this industry, not just science or finances. It'll be handy to select a normalization method later on. I'll show you wahat I mean."""
    st.markdown(long_text)
    with st.expander("See frequency of Monthly Rent on all dataset"):
        with st.spinner("Retrieved dataset. Plotting..."):
            st.pyplot(bar_plot(prices_cleaned))

    with st.expander("See frequency of Monthly Rent on first 40 values (sorted)"):
        with st.spinner("Retrieved data set. Plotting..."):
            st.pyplot(bar_plot_sorted(prices_cleaned))

    st.subheader("Why 1.5 times IQR?")
    st.write("The use of 1.5 times the interquartile range (IQR) to define the whiskers in a box plot is a common convention in statistical analysis. This method, popularized by John Tukey, helps in identifying potential outliers in the data.")
    st.markdown("**How to Identify Outliers:**")
    st.code("""
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[~((df['price_fixed'] >= lower_bound) & (df['price_fixed'] <= upper_bound))]
    """)
    price_range, price_median, price_mean, df_outliers = identify_outliers(prices_cleaned)

    st.subheader("Let's see the visuals")
    with st.expander("Show the plot with outliers"):
        with st.spinner("Retrieved the dataset. Plotting..."):
            st.pyplot(plot_with_outliers(prices_cleaned))

    with st.expander("Show the plot without outliers"):
        with st.spinner("Retrieved the Dataset. Plotting..."):
            st.pyplot(plot_without_outliers(prices_cleaned, df_outliers))
    st.markdown('The blue width of the boxplot would indicate the range of values between Q1 and Q3 along the x-axis.')

    st.write("I'll remove outliers, let's max it up at $6K and add a number of bedrooms - the metric that tend to influence the price to see if that makes more sence")
    with st.expander("Show the plot without outliers with Num of Bedrooms"):
        with st.spinner("Retrieved the Dataset. Plotting..."):
            st.pyplot(plot_without_outliers_with_bedrooms(prices_cleaned))

with st_map_mta:
    st.header("NYC MTA Map visualization")
    st.subheader("Time to add the MTA system to the map of NYC!")
    st.map(mta_df, zoom=10, use_container_width=True)

with map:
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

    # map component of Streamlit
    # nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12) # works quickly, but...
    # nyc_map = generate_geo_map(prices_cleaned) # (takes too long for one map)
    # st_folium(nyc_map, width=700)

    # Initialize session state for the map
    if 'nyc_map' not in st.session_state:
        st.session_state['nyc_map'] = generate_geo_map(prices_cleaned)

    st.header("Map v1.0: Scaled markers of montly rent")
    # Function to handle map generation
    with st.spinner('Creating a Map according to your request...'):
        def handle_map_generation():
            st.session_state['nyc_map'] = generate_geo_map(prices_cleaned)

        handle_map_generation()
        st.markdown("I'm pretty sure there are expensive places for rent of *$160,000.00/mo* and *$140,000.00+/mo* in NYC. However, \
        I'll remove a couple of extreme values on the long end just do no skew results and due to a concern if they are non repeatin single two cases makes mesuspesious of anr error in data \
        true numbers. I'll scale the size of a mark to a Rent value and plot with OpenStreet data and Folium this time. \
        Heads up, it does seam to take *some extra time*.")
        st.success('Map v1.0 is ready! Displaying...')

    # Displaying the cached map if it exists
    if st.session_state['nyc_map']:
        folium_static(st.session_state['nyc_map'])
