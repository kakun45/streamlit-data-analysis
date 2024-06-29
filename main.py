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
    image_path = "map_upfold.png"
    st.image(image_path, caption="Geo Map of NYC with Monthly Rent Prices, $", use_column_width=True)
    st.title("Welcome to my Streamlit app")
    long_text = '''
    In this project, I will investigate the potential influence of proximity to a subway station on rental prices in all five boroughs of New York City, as well as some areas in New Jersey that are easily accessible by the PATH service. These areas are sometimes referred to as the "sixth borough" of New York City or "West West Village."'''
    st.markdown(long_text)

with dataset:
    st.header("Datasets")  # the same as st.markdown(' ## heading2')
    st.subheader("Rental Prices Zillow 2020, and MTA stations")
    long_text = '''I obtained the [Prices Zillow, 2020 dataset](https://www.kaggle.com/datasets/sab30226/zillow-rents-2020) from Kaggle, and Subway info from [MTA Public data](https://new.mta.info/open-data) MTA data is already cleaned (the cleaning process of which deserves a whole other article!) For the header image thanks to: unsplash.com'''
    st.markdown(long_text)

    csv_filepath = './data/zillow.csv'
    rent_df = get_csv_data(csv_filepath)

    csv_filepath = './data/grouped_df.csv'
    mta_df = get_csv_data(csv_filepath)

    # Display Prices dataset
    with st.expander("Zillow 2020 dataset 5 first rows"):
        st.dataframe(rent_df.head())
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
    unique_home_types = get_unique_by_col(rent_df, 'homeType')
    st.write("- **Unique by 'homeType':** ", str(unique_home_types), ";")
    st.write("- **Unique by 'state':**" , str(get_unique_by_col(rent_df, 'state')), "*I'll keep NY, and NJ;*")

    st.write("""
    I gradually apply each step, passing the resulting DataFrame from one function to another as part of a cleaning process. I **do not** manipulate the original dataset.
    
    Next, I remove rows with missing values in the 'homeStatus' column and find all unique values in the column after the drop:
    """)


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


    st.subheader("Now I know which columns and rows to drop")
    st.markdown(
        "Specifically, `df.drop()` anything that is not NY or NJ in 'state' column and that is not FOR_RENT in 'homeStatus' column:")
    st.code("""
    new_df = new_df.drop(new_df[~new_df['state'].isin(['NY', 'NJ'])].index) 
    new_df = new_df.drop(new_df[~new_df['homeStatus'].isin(['FOR_RENT'])].index)
    """)
    selected_rent_df = select_cols_rows(nona_unique_home_st)
    with st.expander("Show how the dataset looks now"):
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


    st.write("""If you're as curious as I am about the 'yearBuilt' of NYC estates, I asked how many unique values are in the 'yearBuilt' column. Next, I put the unique values into bins of 12 and plotted the bins""")
    with st.expander("Generate Plot on 'yearBuilt'"):
        fig = plot_year_built(selected_rent_df)
        st.pyplot(fig)

    st.subheader("Time to clean up the 'price' column!")
    st.write("""I created a function that takes a price string, removes unwanted characters at the front if it starts with '\$', converts it to a float by replacing commas, and adjusts the value based on the ending. If it detects a plus sign in '$1000+/mo', it adds 0.99 and removes the rest:""")
    st.code("if price_string.startswith('$'): \n \tprice_string = price_string[1:] \n")

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


    prices_cleaned = clean_price_col(selected_rent_df)
    with st.expander("Prices fixed in a new column 'price_fixed'"):
        st.dataframe(prices_cleaned.head())

    st.markdown("Here is how to find all the prices that end with .99 after decimal point:")
    st.code("price_99 = df[df['price_fixed'].apply(lambda x: str(x).endswith('.99'))]")
    st.write("Let's see how many of the listings used '+/mo' to evaluate if I need to deal with them. I counted insugnificant 7. Well, I'll leave it in - some interesting results to look for on visualiations later.")
    with st.expander("Show the dataset 'price_99'"):
        @st.cache_data
        def endswith_99(df):
            return df[df['price_fixed'].apply(lambda x: str(x).endswith('.99'))]
        st.dataframe(endswith_99(prices_cleaned))

    # OUTLIERS
    # @st.cache_data
    # def outliers_analyze(df):
    #     return new_df
    # Slider
    # st.select_slider('Choose Rent Price', ['min', 'median', 'max'])

with st_map:
    st.header("NYC Map visualization of Prices")
    st.markdown("###### Let's add all those units for rent to the map of NYC!")
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
    This is unreadable, I know, but there are a lot of rows in this dataset. And the data is very scewed. I display it just for a visual sense of data distribution. let's sort and check on the 
    first 50 rows that have any significance in their count. I want to point out the values that rounded are the ones which accumulate greater counts vs a rundom numer on a numberline, which makes sence for us as humans when talking about prices in base ten system. it's eathier talking about \$2,000/mo rent rather then \$1,987.33/mo.
    *Fun fact, in the past not knowing about this, I converted and changed the prices of an entire stock of an electronics retail store due to the curency-tied items change in value overnight. It was in minutes that I've found out all about how angry customers were with a price like 341,97 in local curency. Learn from my mistakes so you don't have to!* When working with humans is not the same as hard science or finances. I'll call it: humans' VS math' -numbers for later when I'll select a normalization method."""
    st.markdown(long_text)
    with st.expander("See frequency of Monthly Rent on full dataset"):
        with st.spinner("Retrieved dataset. Plotting..."):
            st.pyplot(bar_plot(prices_cleaned))

    with st.expander("See frequency of Monthly Rent on first 40 highest values (sorted)"):
        with st.spinner("Retrieved data set. Plotting..."):
            st.pyplot(bar_plot_sorted(prices_cleaned))

    st.subheader("*Quantiles*")
    st.markdown("*Quantiles are points in data that divide it into equal-sized intervals. For instance, if you divide data into 4 quantiles (quartiles), each quantile will contain 25% of the data.*")
    st.subheader("Why Use Quantile Binning?")
    st.markdown("""- **Handling Skewed Data:** Quantile binning is particularly useful when your data is skewed. It ensures that each bin has the same number of data points, which can provide a more balanced view of the distribution.
- **Visualization:** When visualizing data, quantile binning helps in creating plots where each color represents an equal fraction of the data, making it easier to interpret patterns and distributions.""")

    st.subheader("Why 1.5 times IQR?")
    st.write("The use of 1.5 times the interquartile range (IQR) to define the whiskers in a box plot is a common convention in statistical analysis. This method, popularized by John Tukey, helps in identifying potential outliers in the data. The choice of 1.5 times the IQR as a threshold for outliers is somewhat arbitrary but has been found to work well in practice for many datasets. It balances between being too lenient and too strict in identifying outliers.")
    st.markdown("**How to Identify Outliers:**")
    st.code("""
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[~((df['price_fixed'] >= lower_bound) & (df['price_fixed'] <= upper_bound))]
    """)
    price_range, price_median, price_mean, df_outliers = identify_outliers(prices_cleaned)

    st.subheader("Let's inspect the visuals on how obvious outliers are")
    with st.expander("Show the plot with outliers"):
        with st.spinner("Retrieved the dataset. Plotting..."):
            st.pyplot(plot_with_outliers(prices_cleaned))

    with st.expander("Show the plot without outliers"):
        with st.spinner("Retrieved the Dataset. Plotting..."):
            st.pyplot(plot_without_outliers(prices_cleaned, df_outliers))
    st.markdown('The blue width of the boxplot would indicate the range of values between Q1 and Q3 along the x-axis.')

    st.write("I'll remove outliers, let's max up the rent at $6,000 and add a number of bedrooms - the metric that tend to influence the price, to see if that makes more sence")
    with st.expander("Show the plot without outliers with Num of Bedrooms"):
        with st.spinner("Retrieved the Dataset. Plotting..."):
            st.pyplot(plot_without_outliers_with_bedrooms(prices_cleaned))

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

    st.header("Map v1.0: Scaled markers of montly rent")
    # Function to handle map generation
    with st.spinner('Creating a Map according to your request...'):
        def handle_map_generation():
            st.session_state['nyc_map'] = generate_geo_map_v1(prices_cleaned)

        handle_map_generation()
        st.markdown("""I'm pretty sure there are expensive places for rent in NYC. However, the \$160,000.00/mo and \$140,000.00+/mo* ... I may leave out those extreme values on the long end just do no skew results and non-repeating single two cases makes me suspesious of an error in the data.\n Next up: I scale the size of a mark to a rent value and plot with `OpenStreetMap` and `Folium`. Heads up, it does seam to take ***some extra time***.""")
        st.success('Map v1.0 is ready! Displaying...')

    # Displaying the cached map if it exists
    if st.session_state['nyc_map']:
        folium_static(st.session_state['nyc_map'])


with st_map_mta:
    st.header("NYC MTA Map visualization")
    st.subheader("Time to add the MTA system to the map of NYC!")
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


    st.markdown("""Best thing to do next is to combine both: MTA stations and colorcode the units' locations based on 
    the range of the column 'price_fixed'""")
    st.header("Map v2.0: Scaled and color-coded markers of monthly rent with MTA stations")
    nyc_map2 = generate_geo_map_v2(prices_cleaned)
    with st.spinner(st.success("The map v2.0 is ready. Displaying...")):
        time.sleep(3)
    folium_static(nyc_map2)
    st.markdown("""
    \tYou see how everything fell into almost uniform lowest color of a range on the plot? Let's talk about **data normalization**. Look at median calculated above - the mojority of the dataset is around those values, that makes the outlies of the hiest prices skewing the graphics. it's only them are yellow and green (let's see if you can spot them! Hint? Of course - **Millionaires' Row**), and pushes everything else is on a blue - lower price range - side of a colormap-numberline.
    The way to solve it is **using normalization or scaling**. It involves adjusting the values in the dataset to fit within a specific range, making it easier to compare and visualize the data, especially when there are outliers or a large range of values. This process is essential when dealing with datasets that have varying scales, such as prices ranging from a few hundred dollars to tens of thousands of dollars. 
    """)

with normalization:
    @st.cache_data
    def plot_normalized_prices_humans(df):
        # Calculate quantiles for the 'price_fixed' column
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
        cbar.set_label('Price Ranges')

        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geo Map of NYC with Price Normalized by Min-Max Scaling')

        return fig

    st.header("Using normalization or scaling")
    st.markdown("### `qcut` function")
    st.markdown("The `pd.qcut` function in Pandas divides the data into equal-sized bins based on the quantiles. This ensures that each bin has the same number of data points, making it useful for visualizing distributions and for creating evenly distributed groups:")
    st.code("quantiles, bins = pd.qcut(df['price_fixed'], q=4, retbins=True, labels=False, duplicates='drop')")
    st.pyplot(plot_normalized_prices_humans(prices_cleaned))
    st.markdown("""Yes, color-scheme is very important. The gradient is not enough, some gradients can and will lead to missinterpritation of the data. The `viridis` that is today considered one of the most comprehensive do not work best for this particualar set up. The highlighter yellow is blending with my white backdrop reducing of contrast and abstracting the fact that Downtown is the most expensive in Rent. But I only could notice it when I put them both on a loop. Ask me to post a GIF in here. The `rainbow` nowadays not recommended due the middle of the spectrum visually being higher in intencity of color don't bare significance in values hence misslead in importance of the mid-values. This is `coolwarm` colormap from matplotlib, which is used to color the scatter plot based on quantiles. it works. For now. I'm still pocking at it. "
                "Remember, humans' VS math' numbers? Here is another run for normalization using `Min-Max Scaling` - the technique scales the data to a fixed range, usually [0, 1]. It's fascinating to me that the legend just picked at exactly that. Would you say is easier to get your head around the ranges that `Quantile` picked up fom the data itself rather than calculting the ranges with pure math? That's what I'm talking about!""")
    st.markdown("##### This is how to mormalize the prices column of dataFrame using min-max scaling: "
                "**X' = ( X - Xmin ) / ( Xmax - Xmin ):**")
    st.code("""
        df['price_normalized'] = (df['price_fixed'] - df['price_fixed'].min()) / 
            df['price_fixed'].max() - df['price_fixed'].min()""")
    st.pyplot(plot_normalized_prices_math(prices_cleaned))

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

    st.subheader("Map 3.0: Custom colored price bins with MTA stops")
    st.markdown("""You know how artists blend colors to produce new ones? Well, I used HEX [Color blender](https://colorkit.io/) fun tool where you give it your hex code set up number of steps and look for a resulting blend. I was looking for the blend color to stend out on on outlines of tiles="cartodb positron" of `Folium`. The problem is, I have two sides outliers low and high, to make it obviouse  I assigned them to `blue` and `red`, however everything in between needed to blend and make a visual sence on a gray-ish background against 'highlighter green'. Blue complements orange and when you mix blue and orange together, you get various shades of brown or gray, depending on the proportions of each color used. And to make that blend to stand out on a gray background with enough power of contrast took a few twiks, internet failier, and a new cursed word - `blendomaania`.""")
    nyc_map3 = generate_custom_col_map_v3(prices_cleaned)
    st.markdown("#### Here idea is the low treshold in cooler and high treshhold - warmer. So respectfully Outliers low are blue, high - red")
    with st.spinner(st.success("The map v3.0 is ready. Displaying...")):
        time.sleep(3)
    folium_static(nyc_map3)

    st.markdown("""I do plot the outliers, but I wanted to focus only on the middle most 'popular' in count price range. That's why you see the legend of two. The visual midpoint of 2 colors blended in a tan-brown-ish in RGB (The Queen of Chromoliguistics calls this cocoa or HSL average of orange and blue it might be a shade of green? The Queen of Chromolinguists calls it "between army green and forest green". Sllight change in hue value of orange and it's [mauve](https://meyerweb.com/eric/tools/color-blend/#FF6666:6699CC:1:hex). Color is not a physical entity. It's in the eye of a viewer. Anyway). And what do you know that color tend to stick around of MTA. Play with the map a bit. On the certain zoom-out you can see what areas in that color IS effected by proximity of MTA stops (stops are marked in highliter green). But it's not everywhere, most noticable are Williamsburg, DUMBO and UWS. I know the answer for it being a NewYorker and simply by riding a train a lot here. The main price infiencer after the number of bedrooms and bathrooms is the proximity to Manhattan's Midtown and Downtown, but at a certain point it stops being as matter, because it gets far enough that you woldn't want to spend time to commute as often and maybe go to the City only a fewer times a week or if you can offord it, you move inwords.""")
    st.subheader("Visual Representation")
    st.markdown("#### I remove extreme outliers to focus on the bulk of the data")
    st.markdown("To exclude the bottom 5% and top 5% of data points, focusing on the central 90%:")
    st.code("""
    outlier_threshold_high = df['price_fixed'].quantile(0.95)
    outlier_threshold_low = df['price_fixed'].quantile(0.05)
    """)
    st.markdown("""- **Quantiles 0.05 and 0.95:** Captures central 90% without extreme values, potentially more compact and focused on the majority of data.\n
- **IQR with 1.5 Multiplier:** Captures central data but allows for a wider range, including a larger portion of data as non-outliers.""")
    st.markdown("*In practice*, the choice between these methods depends on the specific data characteristics and the analysis goals. Quantiles offer a more precise exclusion of outliers, while the IQR method provides a broader inclusion of data points.")

    st.subheader("Fan observation")
    st.markdown("""Interesting, that high outliers nested at one spot far away from Manhattan - deep in Queens, look for a blob of red and orange around "Terrace Heights". To test my theory of higher square footage I added to my info Popup a number of bedrooms denoted "Bdr", bathrooms - "Bthr", and "Area" of data represents ft2 (but as not a required field on Zillow, area rarely shows up). Playing with popups confirmed the theory that the prices are higher because that location rents for bigger places and proximity to "Grand Central Terminal" in Midtown nor Subway is not a price influencer.""")
    st.header("Conclusion")
    st.markdown("Quantile binning is a powerful technique for normalizing data, especially when dealing with skewed distributions. By dividing the data into equal-sized bins, it ensures a balanced representation, making it useful for both visualization and analysis.")

with above_the_fold_img:
    folium_static(nyc_map3)
