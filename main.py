# to run $ streamlit run main.py
import folium
import numpy as np
import matplotlib.cm as cmx
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
map = st.container()  # the Folium map for Prices v1.0
st_map_mta = st.container()  # the st.map() for MTA stations
map_v2 = st.container()  # the Folium map for Prices v2.0
normalization = st.container()  # The matplotlib coolwarm maps
custom_colored_map = st.container()  # My colors for the prices+MTA map v3.0


# Load data
@st.cache_data
def get_csv_data(csv_filepath):
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
    # Display 'upfold' media
    image_path = "map_norm_quantile_binning.png"
    st.image(image_path, caption="Geo Map of NYC with Monthly Rent Prices, $ (Source: Zillow, 2020)", use_column_width=True)
    st.title("Welcome to my Streamlit app")
    long_text = '''
    In this Data Analysis project, I will investigate the potential influence of proximity to a subway station on residential rental prices in all five boroughs of New York City, as well as some areas in New Jersey that are easily accessible by the PATH service. These areas are sometimes referred to as the "sixth borough" of New York City or "West West Village."'''
    st.markdown(long_text)

with dataset:
    st.header("Datasets")  # the same as st.markdown(' ## heading2')
    st.subheader("Rental Prices Zillow 2020, and MTA stations")
    long_text = '''I obtained a [Zillow, 2020 dataset of Prices](https://www.kaggle.com/datasets/sab30226/zillow-rents-2020) from Kaggle, and Subway info from [MTA Open data](https://new.mta.info/open-data) MTA data is already cleaned (the cleaning process of which deserves a whole other article!) For the header image thanks to: unsplash.com'''
    st.markdown(long_text)

    # start = time.time()
    csv_filepath = './data/zillow.csv'
    rent_df = get_csv_data(csv_filepath)
    # st.info(f"⏱ data loaded in: {time.time() - start:.2f}sec")  # .01s

    csv_filepath = './data/grouped_df.csv'
    mta_df = get_csv_data(csv_filepath)

    # Display Prices dataset
    with st.expander("Zillow 2020 dataset 5 first rows"):
        # st.dataframe(rent_df.head())  # works; dynamic is removed due to streamlit.io memory issue
        st.dataframe(get_csv_data('./data/zillow_5rows.csv'))  # reduced file Sz
    # Display MTA dataset
    with st.expander("MTA stations data (Cleaned) 5 first rows"):
        st.dataframe(mta_df.head())  # full width, for narrower view:  st.dataframe(mta_df.head())

with data_clean:
    st.header("Data Cleaning")
    long_text = """
    As with any data science project you embark on, there will always be some data cleaning needed to accomplish your goal.
    Let's get intimate with the dataset.
    """
    st.markdown(long_text)

    st.markdown(
        "Utilizing `.unique()` to filter unique values on one column to check what types of homes are available here:")
    st.code("df['columnName'].unique()")

    # Cache the unique home types
    @st.cache_data
    def get_unique_by_col(df, col):
        return df[col].unique()


    # Display the unique 'homeType' -s
    # start_unique_home_types = time.time()
    unique_home_types = get_unique_by_col(rent_df, 'homeType')
    st.write("- **Unique by 'homeType':** ", str(unique_home_types), ";")
    # st.info(f"⏱ data loaded in: {time.time() - start_unique_home_types:.2f}sec")  # .03s OK

    # start_unique_by_col = time.time()
    st.write("- **Unique by 'state':**", str(get_unique_by_col(rent_df, 'state')), "*I'll keep NY, and NJ;*")
    # st.info(f"⏱ data loaded in: {time.time() - start_unique_by_col:.2f}sec")  # .02s OK

    st.write("""
    I gradually apply each step, passing the resulting DataFrame from one function to another as part of a cleaning process. I **do not** manipulate the original dataset.
    
    Next, I remove rows with missing values in the 'homeStatus' column and find all unique values in the column after the drop:
    """)


    @st.cache_data
    def dropna_unique_home_stat(df):
        df2 = df.dropna(subset=['homeStatus'])
        return df2

    # tested performance: no need to remove dynamic file ref
    # start = time.time()
    nona_unique_home_st = dropna_unique_home_stat(rent_df)
    unique_home_status = get_unique_by_col(nona_unique_home_st, 'homeStatus')
    st.write("-  **Unique 'homeStatus':** ", str(unique_home_status), "\nI'll keep 'FOR_RENT' only from this step.")
    # st.info(f"⏱ data loaded in: {time.time() - start:.2f}sec")  # .04s OK

    @st.cache_data
    def select_cols_rows(df):
        # works; dynamic usage removed this file preprocessed before loading it into the Streamlit app, due to the cloud RAM constrains
        new_df = df[
            ['yearBuilt', 'homeStatus', 'longitude', 'latitude', 'price', 'city', 'state', 'postal_code', 'bedrooms',
             'bathrooms', 'area']]
        # df drop anything that is not NY or NJ in state col and that is not FOR_RENT in homeStatus
        new_df = new_df.drop(new_df[~new_df['state'].isin(['NY', 'NJ'])].index)
        new_df = new_df.drop(new_df[~new_df['homeStatus'].isin(['FOR_RENT'])].index)
        return new_df


    st.subheader("Now I know which columns and rows to drop")
    st.markdown(
        "Specifically, `df.drop()` anything that is not NY or NJ in 'state' column and that is not FOR_RENT in 'homeStatus' column:")
    st.code("""
        df_NYNJ = df.drop(df[~df['state'].isin(['NY', 'NJ'])].index)
        df_rents = df_NYNJ.drop(df_NYNJ[~df_NYNJ['homeStatus'].isin(['FOR_RENT'])].index)
        new_df = df_rents[['yearBuilt', 'homeStatus', 'longitude', 'latitude', 'price', 'state', 'bedrooms', 'bathrooms', 'area']]
    """)

    # selected_rent_df = select_cols_rows(nona_unique_home_st)  # works; removed dynamic file processing: remove old file ref
    selected_rent_df = get_csv_data('./data/zillow_dropped_col_rows.csv')
    with st.expander("Show how the dataset looks now (5 rows)"):
        st.dataframe(selected_rent_df.head())

    # yearBuilt plot
    @st.cache_data
    def plot_year_built(df):
        """
        how many unique values are in 'yearBuilt' col, put the unique vals into bins of 12 and plot the bins
        Cache the plot generation function
        """
        unique_year_built_values = df['yearBuilt'].unique()
        num_unique_values = len(unique_year_built_values)
        st.info(f"There are {num_unique_values} unique values in the 'yearBuilt' column.")

        # Create bins of 12: 1800y - 2021y
        bins = np.append(np.arange(1800, 2021, 12), 2021)

        # Plot the bins
        fig, ax = plt.subplots()
        ax.hist(df['yearBuilt'], bins=bins)
        ax.set_xlabel('Year Built bins')
        ax.set_ylabel('Number of Units')
        ax.set_title('Distribution of Houses by Year Built')
        return fig

    @st.cache_data
    def remove_col(df, col_name):
        df = df.drop(col_name, axis=1)
        return df

    st.write("""If you're as curious as I am about the 'yearBuilt' of NYC estates, I asked how many unique values are in the 'yearBuilt' column. Next, I put the unique values into bins of 12 and plotted the bins""")
    with st.expander("Generate Plot on 'yearBuilt'"):
        fig = plot_year_built(selected_rent_df)
        st.pyplot(fig)

    selected_rent_cleanup = remove_col(selected_rent_df, 'yearBuilt')

    st.subheader("Time to clean up the 'price' column!")
    st.write("""I created a function that takes a price string, removes unwanted characters at the front if it starts with '\$', converts it to a float by replacing commas, and adjusts the value based on the ending. If it detects a plus sign in '$1000+/mo', it adds 0.99 and removes the rest:""")
    st.code("""
    if price_string.startswith('$'): 
        price_string = price_string[1:]            
    if price_string.endswith('+/mo'):
        price_string = price_string[:-4]
        price_string = str(float(price_string.replace(',', '')) + .99)
    ...
    """)

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
        df = df.drop('price', axis=1)  # remove old 'price' col to speedup
        return df


    prices_cleaned = clean_price_col(selected_rent_cleanup)
    with st.expander("Show the dataset with a new column 'price_fixed' (5 rows)"):
        st.dataframe(prices_cleaned.head())

    st.markdown("Here is how to find all the prices that end with .99 after the decimal point:")
    st.code("price_99 = df[df['price_fixed'].apply(lambda x: str(x).endswith('.99'))]")
    st.write("Let's see how many of the listings used '+/mo'. I counted insugnificant 7. Well, I'll leave them in - they might be cool to look for on visualiations later.")
    with st.expander("Show the dataframe 'price_99'"):
        @st.cache_data
        def endswith_99(df):
            return df[df['price_fixed'].apply(lambda x: str(x).endswith('.99'))]
        st.dataframe(endswith_99(prices_cleaned))


with st_map:
    st.header("NYC Map Visualization of Prices")
    st.markdown("Let's add all those rent listings to the map of NYC!")
    st.map(prices_cleaned, zoom=10, use_container_width=False)
    st.markdown("Looks *good*. Or is it?! I need to see what kind of prices are there. Statistics to the rescue.")

with plots:
    @st.cache_data
    def bar_plot(df):
        """
        # works; dynamic usage removed due to RAM. cloud memory restrains of Streamlit.io hosting
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
        """
        # works; dynamic usage removed due to RAM. cloud memory restrains of Streamlit.io hosting
        Plot a frequency of the price, make a plot figzie 30 wide, sort by value in a price_fixed
        """
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
        upper_bound = q3 + 1.5 * iqr  # decrease 1.5 based on domain knowledge?

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
    This is unreadable, I know, but there are a lot of rows in this dataset, and the data is very skewed. I display it just to give a visual sense of the data distribution. Let's sort and check the first 50 rows that have any significance in their count. I want to point out that the rounded values are the ones that accumulate greater counts versus a random number on a number line, which makes sense for us as humans when talking about prices in the base ten system. It's easier to talk about \$2,000/mo rent rather than \$1,987.33/mo.

    *Fun fact: In the past, not knowing about this, I converted and changed the prices of an entire stock of an electronics retail store due to a currency-tied value change overnight. It took minutes for me to find out how angry customers were with a price of 341.97 in local currency. Learn from my mistakes so you don't have to!* Working with humans is not the same as working with hard science or finances. I'll call it: 'Humans vs. Math Numbers' for later when I select a normalization method.
    """
    st.markdown(long_text)
    # start = time.time()
    with st.expander("See the frequency of Monthly Rent on the first 500 rows of dataset"):
        # with st.spinner("Retrieved dataset. Plotting..."):
            # st.pyplot(bar_plot(prices_cleaned)) # works; dynamic usage removed due to RAM issue.
        image_path = "price_counts_500.png"
        st.image(image_path, caption="The frequency of Monthly Rent on the first 500 of the dataset",
                 use_column_width=True)
        # st.info(f"⏱ pic loaded in: {time.time() - start:.2f} sec")  # 0.25-0.39sec for .png VS bar_plot() -> 12.44sec

    with st.expander("See the frequency of Monthly Rent for the first 40 highest values"):
        # with st.spinner("Retrieved data set. Plotting..."):
            # st.pyplot(bar_plot_sorted(prices_cleaned)) # works; dynamic usage removed due to RAM issue.
        image_path_sorted = "price_counts_40.png"
        st.image(image_path_sorted, caption="The frequency of Monthly Rent on the first 40 highest prices",
                 use_column_width=True)

    st.subheader("*Quantiles*")
    st.markdown("*Quantiles are points in data that divide it into equal-sized intervals. For instance, if you divide data into 4 quantiles (quartiles), each quantile will contain 25% of the data.*")
    st.subheader("Why Use Quantile Binning?")
    st.markdown("""- **Handling Skewed Data:** Quantile binning is particularly useful when your data is skewed. It ensures that each bin has the same number of data points, which can provide a more balanced view of the distribution.
- **Visualization:** When visualizing data, quantile binning helps in creating plots where each color represents an equal fraction of the data, making it easier to interpret patterns and distributions.""")

    st.subheader("Why 1.5 times IQR?")
    st.write("The use of 1.5 times the interquartile range (IQR) to define the whiskers in a box plot is a common convention in statistical analysis. This method, popularized by John Tukey, helps in identifying potential outliers in the data. The choice of 1.5 times the IQR as a threshold for outliers is somewhat arbitrary but has been found to work well in practice for many datasets. It balances between being too lenient and too strict in identifying outliers.")
    st.markdown("**How to Identify Outliers:**")
    st.code("""
        q1 = df['price_fixed'].quantile(0.25)
        q3 = df['price_fixed'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[~((df['price_fixed'] >= lower_bound) & (df['price_fixed'] <= upper_bound))]
    """)
    price_range, price_median, price_mean, df_outliers = identify_outliers(prices_cleaned)

    # start = time.time()
    st.subheader("Let's inspect the visuals on how obvious outliers are")
    with st.expander("Show the plot with outliers"):
        # with st.spinner("Retrieved the dataset. Plotting..."):
            # st.pyplot(plot_with_outliers(prices_cleaned))
        image_path = "price_distribution_with_outliers.png"
        st.image(image_path, caption="The plot with outliers", use_column_width=True)
        # st.info(f"⏱ plot loaded in: {time.time() - start:.2}sec")  #plot+spinner: .47s, plot,No spinner: .5s, png,no spin .0031s

    # start = time.time()
    with st.expander("Show the plot without outliers"):
        # with st.spinner("Retrieved the Dataset. Plotting..."):
            # st.pyplot(plot_without_outliers(prices_cleaned, df_outliers))
        image_path = "price_distribution_without_outliers.png"
        st.image(image_path, caption="The plot without outliers", use_column_width=True)
        # st.info(f"⏱ plot loaded in: {time.time() - start:.2}sec")  #plot+spinner: 0.32s, plon,No spinner: 0.31s-0.41s, png,no spin .0021s
    st.markdown('*The blue width of the boxplot would indicate the range of values between Q1 and Q3 along the x-axis.*')

    st.write("I'll remove outliers, setting the maximum rent at \$6,000, and add the number of bedrooms—a metric that tends to influence the price—to see if that makes more sense.")
    # start = time.time()
    with st.expander("Show the plot without outliers with Num of Bedrooms"):
        # with st.spinner("Retrieved the Dataset. Plotting..."):
            # st.pyplot(plot_without_outliers_with_bedrooms(prices_cleaned))  # works; removing dynamic plot due to poor performance
        image_path = "price_bedrooms.png"
        st.image(image_path, caption="The plot without outliers with Num of Bedrooms", use_column_width=True)
        # st.info(f"⏱ plot loaded in: {time.time() - start:.2f} sec") # plot+spinner: .55s, plot,NO spinner: .76s-1.17s, png+spin->.0s, png,NO spinner ->.0s

with map:
    @st.cache_data
    def generate_geo_map_v1(df):
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
    # nyc_map = generate_geo_map(prices_cleaned) # ..takes too long for one map
    # st_folium(nyc_map, width=700)

    # Initialize session state for the map
    if 'nyc_map' not in st.session_state:
        st.session_state['nyc_map'] = generate_geo_map_v1(prices_cleaned)

    st.header("Map v1.0: Scaled Markers of Monthly Rent")
    # Function to handle map v1.0 generation
    with st.spinner('Creating a Map according to your request...'):
        def handle_map_generation():
            st.session_state['nyc_map'] = generate_geo_map_v1(prices_cleaned)

        handle_map_generation()
        st.markdown("""
        I'm pretty sure there are expensive places for rent in NYC. However, the \$160,000.00/mo and \$140,000.00+/mo... I may leave out those extreme values on the high end just to avoid skewing the results, and the non-repeating two cases make me suspicious of an error in the data. 
        
        Next up: I scale the size of a marker to the rent value and plot it with `OpenStreetMap` and `Folium`. Heads up, this may take ***some extra time*** to load.
        """)
        st.success('Map v1.0 is ready! Displaying...')

    # Displaying the cached map if it exists
    if st.session_state['nyc_map']:
        folium_static(st.session_state['nyc_map'])


with st_map_mta:
    st.header("NYC Map Visualization of MTA Stops")
    st.markdown("Time to add the MTA system to the map of NYC!")
    # Streamlit basic map of MTA stops
    st.map(mta_df, zoom=10, use_container_width=True)

with map_v2:
    @st.cache_data
    def generate_geo_map_v2(df):
        """
        Show this linear split of scale map to spell it out "Why this doesn't work due to long tail range especially when dealing with skewed distributions." 
        """
        # Create a map object with initial location and zoom level
        nyc_map2 = folium.Map(location=[40.7549, -73.9840], zoom_start=11)

        # Create a colormap, usage of 4 color due to 4Q
        colormap = folium.LinearColormap(colors=['blue', 'lightblue', 'green', 'orange'],
                                         vmin=df['price_fixed'].min(), vmax=df['price_fixed'].max())

        # Add circles to the map for each row in df
        for index, row in df.iterrows():
            # Get the color for the current price
            color = colormap(row['price_fixed'])
            circle = folium.Circle(location=[row['latitude'], row['longitude']],
                                   radius=row['price_fixed'] * 0.002,
                                   color=color,
                                   fill=True,
                                   fill_color=color)
            folium.Popup(f"Price: ${row['price_fixed']}", parse_html=True).add_to(circle)
            nyc_map2.add_child(circle)

        # Add red circles to the map for each row in df_mta
        for index, row in mta_df.iterrows():
            circle = folium.Circle(location=[row['lat'], row['lon']], radius=5, color='red')
            nyc_map2.add_child(circle)

        # Add the colormap to the map
        colormap.caption = 'Monthly Rent, $'
        colormap.add_to(nyc_map2)
        return nyc_map2


    st.markdown("""The best thing to do next is to combine both: MTA stations and color-code the unit locations based on the range of the 'price_fixed' column.""")
    st.header("Map v2.0: Scaled and color-coded markers of monthly rent with MTA stations in red")
    nyc_map2 = generate_geo_map_v2(prices_cleaned)
    with st.spinner(st.success("The map v2.0 is ready. Displaying...")):
        time.sleep(3)
    folium_static(nyc_map2)
    st.markdown("""
    You see how everything falls into almost a uniform lowest color range on the plot? Let's talk about data normalization. Look at the median calculated above—most of the dataset is around those values, which makes the outliers of the highest prices skew the graphics. Only those high prices are yellow and green (let's see if you can spot them! Hint? Of course—**Millionaires' Row**), pushing everything else into the blue—lower price range—side of the colormap-number line.

    The way to solve this is through ***normalization or scaling***. It involves adjusting the values in the dataset to fit within a specific range, making it easier to compare and visualize the data, especially when there are outliers or a large range of values. This process is essential when dealing with datasets that have varying scales, such as prices ranging from a few hundred dollars to tens of thousands of dollars.
    """)

with normalization:
    @st.cache_data
    def plot_normalized_prices_humans(df):
        """
        Dynamic, tested, works; usage removed due to online Cloud RAM usage on Streamlit
        Calculate quantiles for the 'price_fixed' column
        """
        quantiles, bins = pd.qcut(df['price_fixed'], q=4, retbins=True, labels=False, duplicates='drop')

        # Add the quantiles as a new column in the DataFrame
        df['quantile'] = quantiles

        # Create a colormap
        cmap = cmx.coolwarm

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create a scatter plot with longitude and latitude as coordinates, colors the points based on the quantiles, and sizes the points based on price_fixed.
        scatter = ax.scatter(df['longitude'], df['latitude'],
                             c=df['quantile'], cmap=cmap,
                             s=df['price_fixed'] * 0.002)

        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')
        # Create custom labels for the colorbar
        quantile_labels = [f'Quantile {i}: ${bins[i]:,.0f} - ${bins[i + 1]:,.0f}' for i in range(len(bins) - 1)]
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(quantile_labels)
        cbar.set_label('Price Ranges, $')

        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geo map of NYC with price normalized by Quantile Binning')

        return fig

    @st.cache_data
    def plot_normalized_prices_math(df):
        """
        Dynamic, tested, works; usage removed due to online Cloud RAM usage on Streamlit
        min-max scaling normalization. Filters out and shows all, creates df['quantile'] -> for ex.: Q4 2800 - 100000
        """
        # Normalize the price_fixed column using min-max scaling
        df['price_normalized'] = (df['price_fixed'] - df['price_fixed'].min()) / (
                    df['price_fixed'].max() - df['price_fixed'].min())

        # Calculate quantiles for the normalized 'price_fixed' column
        quantiles, bins = pd.qcut(df['price_normalized'], q=4, retbins=True, labels=False, duplicates='drop')

        # Add the quantiles as a new column in the DataFrame
        df['quantile'] = quantiles

        # Create a colormap
        cmap = cmx.coolwarm  # viridis - is abstracting Highs # PiYG - looks good/cute

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the data points
        scatter = ax.scatter(df['longitude'], df['latitude'],
                             c=df['quantile'], cmap=cmap,
                             s=df['price_fixed'] * 0.002)

        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical')

        # Create custom labels for the colorbar
        quantile_labels = [
            f'Quantile {i + 1}: ${int(bins[i] * df["price_fixed"].max()):,} - ${int(bins[i + 1] * df["price_fixed"].max()):,}'
            for i in range(len(bins) - 1)]
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(quantile_labels)
        cbar.set_label('Price Ranges, $')

        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geo Map of NYC with Price Normalized by Min-Max Scaling')

        return fig

    st.header("Using normalization or scaling")
    st.markdown("### `qcut` function")
    st.markdown("The `pd.qcut()` function in Pandas divides the data into equal-sized bins based on the quantiles. This ensures that each bin has the same number of data points, making it useful for visualizing distributions and for creating evenly distributed groups:")
    st.code("quantiles, bins = pd.qcut(df['price_fixed'], q=4, retbins=True, labels=False, duplicates='drop')")
    # start = time.time()
    # st.pyplot(plot_normalized_prices_humans(prices_cleaned))  # works; removed due to Streamlit.io RAM
    image_path = "map_norm_quantile_binning.png"
    st.image(image_path, caption="Geo map of NYC with price normalized by Quantile Binning", use_column_width=True)
    # st.info(f"⏱ plot loaded in: {time.time() - start:.2f}sec")  # 1.17s plot; .01s png
    st.markdown("""
    Yes, the color scheme is crucial. Not all gradients are suitable; some can lead to misinterpretations of the data. The **'viridis'** shades that work well in many contexts may not be ideal for this specific setup. Highlighter yellow, for instance, blends into my white backdrop, reducing contrast and obscuring the fact that Downtown has the highest rent prices. This became apparent when I compared them side by side. I could demonstrate this with a GIF if you'd like.

    Using **'rainbow'** colormap is also discouraged by convention nowadays because the middle of the spectrum appears more intense visually, which can mislead regarding the significance of mid-range values. Currently, I'm using the **'coolwarm'** colormap from Matplotlib to color the scatter plot based on quantiles, and it's effective—so far. I'm still fine-tuning it.

    Remember our discussion about 'humans' VS math' numbers? Here's another example: **normalization using Min-Max Scaling**. This technique scales data to a fixed range, typically [0, 1]. It's intriguing that the legend perfectly reflects this in range values - try to compare the legends on these two graphs. Do you find it easier to understand the quantile ranges picked directly from the data rather than calculating them purely through mathematical formulas? That's what I find fascinating!
    """)
    st.markdown("##### This is how to normalize the prices column of dataFrame using min-max scaling "
                "The formula: **X' = ( X - Xmin ) / ( Xmax - Xmin ):**")
    st.code("""
        df['price_normalized'] = (df['price_fixed'] - df['price_fixed'].min()) / 
            df['price_fixed'].max() - df['price_fixed'].min()""")
    # start = time.time()
    # st.pyplot(plot_normalized_prices_math(prices_cleaned))  # works; removed due to Streamlit.io RAM
    image_path = "map_norm_min_max_scaling.png"
    st.image(image_path, caption="Geo Map of NYC with Price Normalized by Min-Max Scaling", use_column_width=True)
    # st.info(f"⏱ plot loaded in: {time.time() - start:.2}sec")  # 1.2-1.3s Plot; .0048s .png

with custom_colored_map:
    @st.cache_data
    def generate_custom_col_map_v3(df):
        # Create a map object with initial location and zoom level
        nyc_map3 = folium.Map(location=[40.7549, -73.9840], tiles="cartodb positron", zoom_start=11)

        # Calculate price thresholds for outliers (adjust these as needed)
        outlier_threshold_high = df['price_fixed'].quantile(0.95)
        outlier_threshold_low = df['price_fixed'].quantile(0.05)

        # Create a colormap for non-outlier values
        colormap = folium.LinearColormap(
            colors=['#4EB7FF', '#FF8200'],  # #0dbeff, 7FCFFF (lightblue), #ff6a00 (orange) -> mix midpoint
            vmin=df[(df['price_fixed'] >= outlier_threshold_low) &
                    (df['price_fixed'] <= outlier_threshold_high)]['price_fixed'].min(),
            vmax=df[(df['price_fixed'] >= outlier_threshold_low) &
                    (df['price_fixed'] <= outlier_threshold_high)]['price_fixed'].max()
        )

        # Add circles to the map
        for index, row in df.iterrows():
            price = row['price_fixed']
            bedrooms = row['bedrooms']
            bathrooms = row['bathrooms']
            area = row['area']

            if price > outlier_threshold_high:
                color = '#C1291D'  # RED Color for high outliers
                radius = row['price_fixed'] * 0.003  # Larger radius for high outliers
            elif price < outlier_threshold_low:
                color = 'blue'  # Color for low outliers
                radius = 10  # Regular radius for low outliers
            else:
                color = colormap(price)
                radius = row['price_fixed'] * 0.005  # radius scalable for everything between

            folium.Circle(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                tooltip=f"Rent: ${price:,.0f} Bdr: {bedrooms} Bthr: {bathrooms} Area: {area}"
            ).add_to(nyc_map3)

        # Add red circles to the map for each row in df_mta
        for index, row in mta_df.iterrows():
            circle = folium.Circle(location=[row['lat'], row['lon']], radius=25,
                                   color='#BCFF46')  # marker needs to be hi-vis color
            nyc_map3.add_child(circle)

        # Add the colormap to the map (for non-outliers)
        colormap.caption = 'Price (excluding outliers)'
        colormap.add_to(nyc_map3)
        return nyc_map3

    st.subheader("Map 3.0: Custom-colored price bins with MTA stops")
    st.markdown("""
    Do you know how artists blend colors to create new ones? Well, I used the HEX [Color blender](https://colorkit.io/) tool, where you input hex codes, set the number of steps, and it generates a blended color. I needed these blended colors to stand out against the 'cartodb positron' tiles in Folium `tiles="cartodb positron"`. The challenge was dealing with colors and outliers on both ends—low and high. To highlight those extremes, I assigned them 'dark blue' and 'dark red', respectively. However, for the values in between, I needed colors that would step down logically from from dark red and dark blue, would blend well and make sense visually against a grayish background of the map and with high-visibility 'highlighter green' of MTA stops.

    Blue and orange are ***complementary colors***, and when you mix them, you get various shades of brown or gray in RGB, depending on the proportions. Achieving a blend that stands out on a gray background with enough contrast required several adjustments, working through intermittent display issues, and a new word in my vocabulary—'blendomania'.
    """)
    nyc_map3 = generate_custom_col_map_v3(prices_cleaned)

    st.markdown("#### I remove extreme outliers from the legend and focus on the bulk of the data")
    st.markdown("This is how I exclude the bottom 5% and top 5% of data points, focusing on the central 90%:")
    st.code("""
    outlier_threshold_high = df['price_fixed'].quantile(0.95) # 'red'
    outlier_threshold_low = df['price_fixed'].quantile(0.05) # 'blue'
    """)
    with st.spinner(st.success("The map v3.0 is ready. Displaying...")):
        time.sleep(3)
    folium_static(nyc_map3)
    st.markdown("""*This map version uses cooler colors for lower thresholds and warmer colors for higher ones, with darker shades highlighting outliers: low in <span style="color: blue;">**darker blue**</span> and high in <span style="color: #C1291D;"> **darker red** </span>*""", unsafe_allow_html=True)

    st.markdown("""
    I do plot outliers but focus primarily on the most popular price range, hence the dual legend. The visual midpoint of two blended colors results in a tan-brown hue (what the Queen of Chromolinguistics calls cocoa). A slight change in the hue value of orange turns it into [mauve](https://meyerweb.com/eric/tools/color-blend/#FF6666:6699CC:1:hex). Color is not a physical entity; it exists in the eye of the viewer.
    
    After refining the blend and plotting it, I noticed this midpoint color tends to cluster around MTA stops (marked in highlighter green). Play with the map a bit: at a certain zoom-out, you can see which areas in this color are influenced by proximity to MTA stops. It's not everywhere, but it is particularly noticeable in Williamsburg, DUMBO, and UWS. As a frequent New York commuter, I understand how proximity to Midtown and Downtown Manhattan significantly impacts prices in those "close-enough" areas. However, beyond a certain distance, it stops being as significant because it gets far enough that you wouldn't want to commute as often. If you can afford it, you move inwards to save time.
    """)
    st.header("Conclusions")
    st.markdown("""- **Quantiles 0.05 and 0.95:** Captures central 90% without extreme values, potentially more compact and focused on the majority of data.\n
- **IQR with 1.5 Multiplier:** Captures central data but allows for a wider range, including a larger portion of data as non-outliers.""")
    st.markdown("*Quantile binning* is a powerful technique for normalizing data, especially when dealing with skewed distributions. By dividing the data into equal-sized bins, it ensures a balanced representation, making it useful for both visualization and analysis.")
    st.markdown("*In practice*, the choice between these methods depends on the specific data characteristics and the analysis goals. Quantiles offer a more precise exclusion of outliers, while the IQR method provides a broader inclusion of data points.")

    st.subheader("Fan observation")
    st.markdown("#### Surprising High-Rent Area in Queens, NY")
    st.markdown("""
    Interesting location: high outliers are clustered in one spot far away from Manhattan, deep in Queens. Look for a blob of red and orange around "Terrace Heights." To test my theory about higher square footage, I added the number of bedrooms ("Bdr"), bathrooms ("Bthr"), and "Area" (in ft²) to the info popups on hover. Note that area data is often missing on Zillow as it's not a required field. Playing with the popups confirmed the theory: prices are higher in that location because it rents out larger places, confirming that proximity to subway stops is not a significant price influencer there.
    """)
