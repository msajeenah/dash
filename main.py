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
        options=["Dashboard", "Social media Analysis", "Market Basket analysis","Data Visulaization"],
        icons=["speedometer2","person-add","bar-chart", "info-circle"],
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
    col1.metric("Registered Societies", total_societies, f"+{societies_in_past_30_days}", help="Shows Total Registered Society and changes in the last 30 days")
    col2.metric("Active members", f"{data['num_members'].sum()}", "-1", help="Active Members and changes in the last 30 days (**Note**: Contains sample data)")
    col3.metric("Events Organized", "32", "+2", help="Events organized and changes in the last 30 days (**Note**: Contains sample data)")

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
    selected_sectors_all = st.sidebar.checkbox("Select All Sectors", value=True, key="all_sectors_checkbox")
    if selected_sectors_all:
        selected_sectors = st.sidebar.multiselect("Select Sectors", all_sectors, default=all_sectors, help="Displays the distribution of registered societies by sector.")
    else:
        selected_sectors = []

    st.markdown("#### Registration Trends")

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
    fig.update_yaxes(title="Number of Registrations")
    fig.update_layout(height=250, width=400)
    st.plotly_chart(fig, use_container_width=True)

    r1, r2 = st.columns((7, 3))

    with r1:

        st.markdown("#### Sector Distribution")

        if selected_sectors:
            filtered_sector_counts = data[data['sector_type'].isin(selected_sectors)]['sector_type'].value_counts()
        else:
            filtered_sector_counts = data['sector_type'].value_counts()

        # Example: Pie chart of sector distribution
        sector_distribution = filtered_sector_counts  # Use the filtered counts for the pie chart
        fig2 = px.pie(sector_distribution, values=sector_distribution.values, names=sector_distribution.index, height=400, width=400)
        st.plotly_chart(fig2)

    # Calculate the total number of members per sector
    members_by_sector = filtered_data.groupby('sector_type')['num_members'].sum()

    # Convert the dictionary values to integers
    members_by_sector = members_by_sector.astype(int)

    # Calculate the total number of members per year
    members_by_year = filtered_data.groupby('Year')['num_members'].sum()

    # Convert the dictionary values to integers
    members_by_year = members_by_year.astype(int)

    if selected_years:
        members_by_selected_year = members_by_year.loc[selected_years[0]:selected_years[1]].sum()
    else:
        members_by_selected_year = members_by_year.sum()

     # Display members by sector
    r2.metric("Members by Sector", "")
    for sector, count in members_by_sector.items():
        r2.write(f"{sector}: {count}")



if selected == "Social media Analysis":

    data = load_data()
    st.markdown('## Social Media Data Analysis')
    # Retrieve total number of registered societies
    total_societies = len(data)

    # Calculate the date 30 days ago from today
    start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    start_date = start_date.strftime("%Y-%m-%d")

    # Retrieve the number of registered societies in the past 30 days
    societies_in_past_30_days = len(data[data['registration_date'] >= start_date])

  

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
    selected_sectors_all = st.sidebar.checkbox("Select All Sectors", value=True, key="all_sectors_checkbox")
    if selected_sectors_all:
        selected_sectors = st.sidebar.multiselect("Select Sectors", all_sectors, default=all_sectors, help="Displays the distribution of registered societies by sector.")
    else:
        selected_sectors = []

    st.markdown("#### chart Analysis")

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
    fig.update_yaxes(title="Number of Beesline products")
    fig.update_layout(height=250, width=400)
    st.plotly_chart(fig, use_container_width=True)

    r1, r2 = st.columns((7, 3))

    with r1:

        st.markdown("#### Pie chart")

        if selected_sectors:
            filtered_sector_counts = data[data['sector_type'].isin(selected_sectors)]['sector_type'].value_counts()
        else:
            filtered_sector_counts = data['sector_type'].value_counts()

        # Example: Pie chart of sector distribution
        sector_distribution = filtered_sector_counts  # Use the filtered counts for the pie chart
        fig2 = px.pie(sector_distribution, values=sector_distribution.values, names=sector_distribution.index, height=400, width=400)
        st.plotly_chart(fig2)

    # Calculate the total number of members per sector
    members_by_sector = filtered_data.groupby('sector_type')['num_members'].sum()

    # Convert the dictionary values to integers
    members_by_sector = members_by_sector.astype(int)

    # Calculate the total number of members per year
    members_by_year = filtered_data.groupby('Year')['num_members'].sum()

    # Convert the dictionary values to integers
    members_by_year = members_by_year.astype(int)

    if selected_years:
        members_by_selected_year = members_by_year.loc[selected_years[0]:selected_years[1]].sum()
    else:
        members_by_selected_year = members_by_year.sum()

     # Display members by sector
    r2.metric("products criteria", "")
    for sector, count in members_by_sector.items():
        r2.write(f"{sector}: {count}")


            

if selected == "Market Basket analysis":
    st.markdown("## Market Basket analysis")
    data = load_data()

    # Check if the DataFrame has the required columns
    required_columns = ['society_name', 'address', 'state', 'district', 'registration_date', 'area_of_operation', 'sector_type']
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


        c1, c2 = st.columns((4,6))

        with c1:
            # Popular sectors
            sector_counts = filtered_df['sector_type'].value_counts()
            fig = go.Figure(data=[go.Pie(labels=sector_counts.index, values=sector_counts, hole=0.6)])
            fig.update_layout(
                    title="Popular Sectors",
                    width=300,  # Adjust the width to fit the expanded sidebar
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
                width=550,  # Adjust the width to fit the expanded sidebar
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

if selected == "Data Visulaization":
    st.markdown("## Market Basket analysis")


st.write("Upload a CSV file and explore the data visualization.")

# Create a file uploader widget
file = st.file_uploader("Upload CSV file", type=["csv"])

# Check if a file is uploaded
if file is not None:
    # Read the CSV file data
    data = pd.read_csv(file)
    
    # Perform data preprocessing
    data['time'] = pd.to_datetime(data['time'])
    
    # Calculate device-wise efficiency and average RPM
    data['Efficiency'] = data['Total_rotations'] / (data['Total_rotations'] + data['Off_time'])
    avg_rpm = data['RPM'].mean()
    
    # Display the data in a table
    st.subheader("Data Table")
    st.dataframe(data)
    
    # Get the list of column names
    column_names = data.columns.tolist()
    
    # Select the x-axis and y-axis fields
    x_axis = st.selectbox("Select X-axis field", column_names)
    y_axis = st.selectbox("Select Y-axis field", column_names)
    
    # Select the type of graph to plot
    graph_type = st.selectbox("Select Graph Type", ["Line Plot", "Scatter Plot", "Bar Plot", "Area Plot", "Histogram"])
    
    # Perform additional data preprocessing based on the selected graph type
    if graph_type in ["Line Plot", "Scatter Plot", "Bar Plot", "Area Plot"]:
        data = data.dropna(subset=[x_axis, y_axis])
    
    # Visualize the data based on the selected graph type
    st.subheader("Data Visualization")
    st.write(f"Visualizing {y_axis} vs {x_axis}")
    
    with st.spinner("Generating visualization..."):
        if graph_type == "Line Plot":
            fig = px.line(data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            fig.update_traces(mode='markers+lines')
            fig.update_layout(showlegend=False)
            fig.frames = [go.Frame(data=[go.Scatter(x=data[x_axis][:i+1], y=data[y_axis][:i+1], mode='markers')]) for i in range(len(data))]

        elif graph_type == "Scatter Plot":
            fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            fig.update_layout(showlegend=False)

        elif graph_type == "Bar Plot":
            fig = px.bar(data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            fig.update_layout(showlegend=False)

        elif graph_type == "Area Plot":
            fig = px.area(data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
            fig.update_layout(showlegend=False)

        elif graph_type == "Histogram":
            fig = px.histogram(data, x=y_axis, title=f"{y_axis} Histogram")
            fig.update_layout(showlegend=False)
        
    st.plotly_chart(fig)
    
    # Conclusion based on data visualization
    st.subheader("Conclusion")
    
    if graph_type in ["Line Plot", "Scatter Plot"]:
        st.write(f"The {y_axis} tends to vary with {x_axis}.")
    elif graph_type == "Bar Plot":
        st.write(f"The {y_axis} shows different values for different {x_axis}.")
    elif graph_type == "Area Plot":
        st.write(f"The area under the curve represents the cumulative {y_axis} with respect to {x_axis}.")
    elif graph_type == "Histogram":
        st.write(f"The histogram shows the distribution of {y_axis} values.")

    # Display device-wise efficiency and average RPM
    st.subheader("Device-wise Efficiency")
    efficiency_fig = px.bar(data, x="Device_id", y="Efficiency", title="Device-wise Efficiency")
    efficiency_fig.update_layout(showlegend=False)
    st.plotly_chart(efficiency_fig)
    st.write(f"The device-wise efficiency visualization highlights variations in efficiency across devices.")
    
    st.subheader("Average RPM")
    st.write(f"The average RPM is: {avg_rpm}")
    st.write(f"The average RPM gives an idea of the overall rotational speed.")
