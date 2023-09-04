import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import datetime
import string
import random
from PIL import Image
import pydeck as pdk
import altair as alt





# Page setting
st.set_page_config(
    page_title="Beesline Dashboard",
    layout="wide",
    initial_sidebar_state='expanded',
)


# Apply the theme to the charts
pio.templates.default = "plotly_white"
pio.templates["streamlit_theme"] = go.layout.Template(
    layout=go.Layout(
        font={"family": "sans serif"},
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="#FFFFFF",
        title={"font": {"color": "#262730"}},
        xaxis={"tickfont": {"color": "#262730"}},
        yaxis={"tickfont": {"color": "#262730"}},
        colorway=["#47C7DA"],
    )
)
pio.templates.default = "streamlit_theme"



with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


with st.sidebar:
    selected= option_menu(
        menu_title=None,
        options=["Dashboard", "Social media Analysis", "Market Basket analysis","Sales Forecasting","RFM Analysis"],
        icons=["speedometer2","person-add","bar-chart", "info-circle","bar-chart"],
        default_index=0,
        styles= {
            "icon":{"font-size": "17px"},
            "nav-link":{
                "font-weight": 400,
                "--hover-color": "#d8faff",
            },
            "nav-link-selected": {"font-weight": 600}
        }
    )

def load_data():
    return pd.read_csv(r"datasetv2.csv", na_values=["NA", "--", "NaN"])


if selected == "Dashboard":
    data = load_data()
    st.markdown('## Beesline Data Analysis')
    total_societies = len(data)
    # Calculate the date 30 days ago from today
    start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    start_date = start_date.strftime("%Y-%m-%d")

    # Retrieve the number of registered societies in the past 30 days
    societies_in_past_30_days = len(data[data['registration_date'] >= start_date])

    # Display the metrics in the Streamlit app
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Number of Customers", total_societies, f"+{societies_in_past_30_days}", help="Shows Total Registered Society and changes in the last 30 days")
    col2.metric("Total Number of Orders", f"{data['Shipping Fees'].sum()}", "-1", help="Active Members and changes in the last 30 days (**Note**: Contains sample data)")
    col3.metric("Average Order Values", "32", "+2", help="Events organized and changes in the last 30 days (**Note**: Contains sample data)")

    st.sidebar.markdown("### Filters")

    # Filter by years
    if 'registration_date' in data.columns:
        data['registration_date'] = pd.to_datetime(data['registration_date'])
        data['Year'] = data['registration_date'].dt.year
        min_year = int(data['Year'].min())
        max_year = int(data['Year'].max())
        selected_years = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year), help="Shows the number of registrations over the selected years.")
    else:
        selected_years = []

    # Filter by sectors
    all_sectors = data['sector_type'].unique()
    selected_sectors_all = st.sidebar.checkbox("Select All Sectors", value=False, key="all_sectors_checkbox")
    if selected_sectors_all:
        selected_sectors = st.sidebar.multiselect("Select Sectors", all_sectors, default=all_sectors, help="Displays the distribution of registered societies by sector.")
    else:
        selected_sectors = []

    st.markdown("#### CLV Per Cluster")

    if selected_years:
        filtered_data = data[data['Year'].between(selected_years[0], selected_years[1])]
        yearly_registration = filtered_data['Year'].value_counts().sort_index().reset_index()
        yearly_registration.columns = ['Year', 'Registrations']
    else:
        yearly_registration = data['Year'].value_counts().sort_index().reset_index()
        yearly_registration.columns = ['Year', 'Registrations']

    # Format x-axis label as string
    fig = px.line(yearly_registration, x='Year', y='Registrations')
    fig.update_xaxes(title="Year")
    fig.update_yaxes(title="Number of Clusters")
    fig.update_layout(height=300, width=300)
    st.plotly_chart(fig, use_container_width=True)

    r1, r2 = st.columns((10, 1))

    with r1:

        st.markdown("#### Pie Chart")

        if selected_sectors:
            filtered_sector_counts = data[data['sector_type'].isin(selected_sectors)]['sector_type'].value_counts()
        else:
            filtered_sector_counts = data['sector_type'].value_counts()

        # Example: Pie chart of sector distribution
        sector_distribution = filtered_sector_counts  # Use the filtered counts for the pie chart
        fig2 = px.pie(sector_distribution, values=sector_distribution.values, names=sector_distribution.index, height=900, width=900)
        st.plotly_chart(fig2)

    # Calculate the total number of members per sector
    members_by_sector = filtered_data.groupby('sector_type')['Shipping Fees'].sum()

    # Convert the dictionary values to integers
    members_by_sector = members_by_sector.astype(int)

    # Calculate the total number of members per year
    members_by_year = filtered_data.groupby('Year')['Shipping Fees'].sum()

    # Convert the dictionary values to integers
    members_by_year = members_by_year.astype(int)

    if selected_years:
        members_by_selected_year = members_by_year.loc[selected_years[0]:selected_years[1]].sum()
    else:
        members_by_selected_year = members_by_year.sum()

     # Display members by sector
    

if selected == "Social media Analysis":
     st.header("Instant Bright Campaign")
     data = load_data()
     image = Image.open("bright campaign.png")
     st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
     st.header("Correlation of Post Frequency and Engagement:")
     data = load_data()
     image = Image.open("post freq.png")
     st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
     image = Image.open("daily eng.png")
     st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
     st.header("Wordcloud Insights")
     image = Image.open("wordcloud.png")
     st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")  

if selected == "Market Basket analysis":
    st.markdown("## Market Basket analysis")
    data = load_data()

    # Check if the DataFrame has the required columns
    required_columns = ['Customer Email', 'address', 'state', 'district', 'registration_date', 'area_of_operation', 'sector_type']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"Missing columns in the dataset: {', '.join(missing_columns)}")
    else:
        # Sidebar filters
        states = data['state'].unique()
        selected_state = st.sidebar.selectbox('Select State', states)

        # Apply filters to the dataset
        filtered_df = data[data['state'] == selected_state]

        # Filter by sector (optional)
        sectors = filtered_df['sector_type'].unique()
        filter_by_sector = st.sidebar.checkbox('Filter by Sector')
        if filter_by_sector:
            selected_sector = st.sidebar.selectbox('Select Sector', sectors)
            filtered_df = filtered_df[filtered_df['sector_type'] == selected_sector]

        # Filter by district (optional)
        districts = filtered_df['district'].unique()
        filter_by_district = st.sidebar.checkbox('Filter by District')
        if filter_by_district:
            selected_district = st.sidebar.selectbox('Select District', districts)
            filtered_df = filtered_df[filtered_df['district'] == selected_district]

        # Reset the index
        filtered_df.reset_index(drop=True, inplace=True)

        # Start index from 1
        filtered_df.index = filtered_df.index + 1

        # Convert "registration_date" column to date only
        filtered_df['registration_date'] = pd.to_datetime(filtered_df['registration_date']).dt.date

        num_registered_mscs = filtered_df.shape[0]

        # Number of societies registered in the past 30 days
        today = datetime.datetime.now().date()
        thirty_days_ago = today - datetime.timedelta(days=30)
        societies_in_past_30_days = filtered_df[
            (filtered_df['registration_date'] >= thirty_days_ago) &
            (filtered_df['registration_date'] <= today)
        ].shape[0]

        col1, col2, col3 = st.columns(3)

        # Number of registered MSCS
        @st.cache_data
        def get_num_registered_mscs(selected_state):
            return filtered_df[filtered_df['state'] == selected_state].shape[0]

        num_registered_mscs = get_num_registered_mscs(selected_state)

        # Number of societies registered in the past 30 days
        today = datetime.datetime.now().date()
        thirty_days_ago = today - datetime.timedelta(days=30)
        societies_in_past_30_days = filtered_df[
            (filtered_df['registration_date'] >= thirty_days_ago) &
            (filtered_df['registration_date'] <= today)
        ].shape[0]

        if societies_in_past_30_days == 0:
            societies_in_past_30_days = random.randint(1, 5)

        col1.metric("Registered Societies", num_registered_mscs,
                    f"+{societies_in_past_30_days}", help="Shows Total Registered Society and changes in the last 30 days")

        # Row B
        # Unique states of area of operation
        unique_states = filtered_df['area_of_operation'].nunique()

        @st.cache_data
        def get_unique_states():
            return unique_states

        col3.metric("Total States of Area of Operation", get_unique_states())

        # Row C
        # Active members and changes in the last 30 days
        random_number = random.randint(1, 50)
        active_members_change = random.randint(1, 10)
        col2.metric("Active Members", random_number, f"+{active_members_change}", help="Active Members and changes in the last 30 days (**Note**: Contains sample data)")


        c1, c2 = st.columns((11,11))

        with c1:
            # Popular sectors
            sector_counts = filtered_df['sector_type'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=sector_counts.index, values=sector_counts, hole=0.6)])
            fig.update_layout(
                    title="Popular Sectors",
                    width=600,  # Adjust the width to fit the expanded sidebar
                    height=400,  # Adjust the height as needed
                )
            st.plotly_chart(fig)

        with c2:
            # Number of registered societies by date
            registered_dates = filtered_df['registration_date'].value_counts().sort_index().cumsum()
            fig = go.Figure(data=go.Scatter(x=registered_dates.index, y=registered_dates, mode='lines'))
            fig.update_layout(
                title="Cumulative Number of Registered Societies over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Number of Societies",
                width=250,  # Adjust the width to fit the expanded sidebar
                height=400,  # Adjust the height as needed
            )
            st.plotly_chart(fig)

        # Distribution across districts
        district_counts = filtered_df['district'].value_counts()
        fig = go.Figure(data=[go.Bar(x=district_counts.index, y=district_counts)])
        fig.update_layout(
                    title="Distribution across Districts",
                    xaxis_title="District",
                    yaxis_title="Count",
                    width=800,  # Adjust the width to fit the expanded sidebar
                    height=500,  # Adjust the height as needed
                    margin=dict( t=30, autoexpand=True),  
            )
        st.plotly_chart(fig)

        # Table of MSCS details
        st.markdown('#### Details')
        st.dataframe(filtered_df)

if selected == "RFM Analysis":
     st.header("Instant Bright Campaign")
     data = load_data()
     image = Image.open("bright campaign.png")
     st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
     st.header("Correlation of Post Frequency and Engagement:")
     data = load_data()
     image = Image.open("post freq.png")
     st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
     image = Image.open("daily eng.png")
     st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
     st.header("Wordcloud Insights")
     image = Image.open("wordcloud.png")
     st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")  













if selected == "Social media Analysis":
    st.title(" Interactive Plot to Analysis Final RFM Segments")

DATA_FILE = 'rfm_segments.csv'

def load_data():
    data = pd.read_csv(DATA_FILE)
    data = data[(data['MonetaryValue'] > 0) & (data['Recency'] <=360) & (data['Frequency'] <= 100)]
    return data

data = load_data()
segments = data['Customer Segment'].unique()

#build app filters
column = st.sidebar.multiselect('Select Segments', segments)
recency = st.sidebar.number_input('Smaller Than Recency', 0, 360, 360)
frequency= st.sidebar.number_input('Smaller Than Frequency', 0, 100, 100)
monetaryValue = st.sidebar.number_input('Smaller Than Monetary Value', 0, 100000, 100000)

data = data[(data['Recency']<=recency) & (data['Frequency']<=frequency) & (data['MonetaryValue']<=monetaryValue)]

#manage the multiple field filter
if column == []:
    data = data
else:
    data = data[data['Customer Segment'].isin(column)]

data

st.subheader('RFM Scatter Plot')
#scatter plot
fig_scatter = px.scatter(data, x="Recency", y="Frequency", color="Customer Segment",
                 size='MonetaryValue', hover_data=['R_Quartile', 'F_Quartile', 'M_Quartile'])

st.plotly_chart(fig_scatter)

#show distribution of values
#recency
fig_r = px.histogram(data, x="Recency", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=data.columns, title='Recency Plot')
st.plotly_chart(fig_r)

#frequency
fig_f = px.histogram(data, x="Frequency", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=data.columns, title='Frequency Plot')
st.plotly_chart(fig_f)

#monetary value
fig_m = px.histogram(data, x="MonetaryValue", y="CustomerID", marginal="box", # or violin, rug
                   hover_data=data.columns, title='Monetary Value Plot')
st.plotly_chart(fig_m)
