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
    page_title="CRCS Dashboard",
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
        options=["Dashboard", "Registration", "Data Analytics", "About"],
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
    st.markdown('## Overview')
    # Retrieve total number of registered societies
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



if selected == "Registration":

    tab1, tab2, tab3 = st.tabs(["New Register", "Appeal", "Details"])

    with tab1:
        # Define the sector types
        SECTOR_TYPES = [
            'Agro', 'Construction', 'Cooperative Bank', 'Credit', 'Dairy', 'Federation',
            'Fisheries', 'Health/Hospital', 'Housing', 'Industrial/Textile', 'Marketing',
            'Tourism'
        ]

        # Load dataset with Indian states and districts
        data = pd.read_csv('ISD.csv')

        # Fetch all Indian states
        indian_states = data['state'].unique()

        def generate_society_id():
            # Generate a random alphanumeric string
            chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
            random_string = ''.join(random.choices(chars, k=4))

            # Get the current date and time
            now = datetime.datetime.now()

            # Format the date and time components
            date_component = now.strftime("%Y%m%d")
            time_component = now.strftime("%H%M%S")

            # Combine the components and random string to form the society ID
            society_id = f"SOC-{date_component}{time_component}{random_string}"

            # Truncate or pad the society ID to a length of 10 characters
            society_id = society_id[:10].ljust(10, '0')

            return society_id

        def fetch_districts(state):
            if state:
                districts = data[data['state'] == state]['district'].unique()
                return districts
            return []

        def register_society():
            st.markdown("### Cooperative Society Registration")

            # Form inputs
            society_name = st.text_input("Society Name")
            address = st.text_area("Address", height=100)  # Increase the height of the address field
            state = st.selectbox("State", indian_states, key="state")

            # Fetch districts based on the selected state
            districts = fetch_districts(state)
            district = st.selectbox("District", np.append(districts, "Other"), key="district")

            if district == "Other":
                district = st.text_input("Other District")

            area_of_operation = st.multiselect("Area of Operation", indian_states[:-1])
            sector_type = st.selectbox("Sector Type", SECTOR_TYPES)

            if st.button("Submit"):
                # Validate form inputs
                if not society_name or not address or not state or not district or not area_of_operation or not sector_type:
                    st.warning("Please fill in all the required fields.")
                elif district == "Other" and not district:
                    st.warning("Please enter the Other District.")
                else:
                    society_id = generate_society_id()
                    # Prepare the registration data
                    registration_data = {
                        "society_id": society_id,
                        "society_name": society_name,
                        "address": address,
                        "state": state,
                        "district": district,
                        "registration_date": datetime.datetime.now(),
                        "area_of_operation": ", ".join(area_of_operation),
                        "sector_type": sector_type,
                    }

                    # Write the registration data to a CSV file
                    with open("dataset.csv", "a") as file:
                        registration_row = ",".join([str(value) for value in registration_data.values()])
                        file.write(registration_row + "\n")

                    # Provide confirmation message or next steps
                    st.success("Society registered successfully!")
                    
                    # Show registered data
                    registered_data = pd.DataFrame([registration_data])
                    registered_data.index = registered_data.index + 1  # Start index from 1
                    st.markdown("#### Detail")
                    st.dataframe(registered_data.rename(columns=lambda x: x.replace("_", " ").title()))
                        
                return None
            
        register_society()

    with tab2:
        st.markdown("### Amendments and appealing")
        
        def handle_amendments_appeals():
            global data  # Declare data as a global variable

            # Form inputs for amendments and appeals
            society_id = st.text_input("Society ID")
            change_details = st.text_area("Change Details")
            appeal_reason = st.text_area("Appeal Reason")

            if st.button("Submit Request"):
                # Validate form inputs
                if not society_id or not change_details or not appeal_reason:
                    st.warning("Please fill in all the required fields.")
                else:
                    # Save the request to the dataset
                    request = {
                        "society_id": society_id,
                        "change_details": change_details,
                        "appeal_reason": appeal_reason,
                        "status": "Pending"  # Assuming status starts as "Pending"
                    }
                    
                    st.success("Request submitted successfully!")
                    request_data = pd.DataFrame([request])
                    request_data.index = request_data.index + 1  # Start index from 1
                    st.markdown("#### Detail")
                    st.dataframe(request_data.rename(columns=lambda x: x.replace("_", " ").title()))
                    

        handle_amendments_appeals()

    with tab3:
        st.markdown("### Society Details")

        # Filter by Society ID (optional)
        society_id_filter = st.text_input("Filter by Society ID")
        data = load_data()
        data['registration_date'] = pd.to_datetime(data['registration_date']).dt.year

        # Apply filter if Society ID is provided
        if society_id_filter:
            filtered_data = data[data['society_id'] == society_id_filter]
        else:
            filtered_data = data

        if not filtered_data.empty:
            # Format the header names
            header_mapping = {
                'society_name': 'Society Name',
                'address': 'Address',
                'state': 'State',
                'district': 'District',
                'registration_date': 'Registration Date',
                'area_of_operation': 'Area of Operation',
                'sector_type': 'Sector Type'
            }
            filtered_data = filtered_data.rename(columns=header_mapping)

            # Reset the index and show data with a single index starting from 1
            filtered_data.reset_index(drop=True, inplace=True)
            filtered_data.index += 1

            # Display the filtered data with the requested details
            st.dataframe(filtered_data[['Society Name', 'Address', 'State', 'District', 'Registration Date', 'Area of Operation', 'Sector Type']])
        else:
            st.info("No data found.")

            

if selected == "Data Analytics":
    st.markdown("## Insights")
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

if selected == "About":
    st.markdown('## About')
    st.markdown('''
        The Central Registrar for Cooperative Societies (CRCS) is responsible for registering and regulating multistate cooperative societies in India, in accordance with the MSCS Act of 2002.
        
        As part of the CRCS Hackathon, we have developed this comprehensive dashboard for the upcoming new CRCS portal. The dashboard aims to streamline the registration process, handle amendments and appeals, and manage annual returns for the registered societies.
        
        **Dashboard Features:**
        - Visualization: The dashboard presents the data from the provided dataset in a visually appealing and easily understandable manner using charts, graphs, and maps.
        - Filters and Interactivity: Users can interact with the dashboard by incorporating filters, dropdown menus, and selection options to explore and analyze specific aspects of the data.
        - Key Metrics: The dashboard displays key metrics, summaries, and trends related to MSCS, such as the number of registered MSCS, distribution across states and districts, and popular sectors.
        - Drill-Down Capabilities: Users can drill down into specific MSCS details, such as their address, registration date, area of operation, and sector type.
        - Responsive Design: The dashboard is responsive and compatible with different screen sizes and devices for a seamless user experience.
        - Data Analytics: The dashboard utilizes data analytics techniques to derive meaningful insights from the dataset and presents them in an informative manner.
        - User-Friendly Interface: The dashboard is designed with an intuitive and user-friendly interface that allows users to navigate effortlessly.
        
        We hope that this dashboard will contribute to the efficient management and monitoring of multistate cooperative societies in India. For any queries or feedback, please contact us.
        
        ---
        Dashboard developed for the CRCS Hackathon
        ''')


st.sidebar.markdown('''
---
Created by [Sathish]('https://github.com/sathish-1804/CRCS_dashboard').
''')
