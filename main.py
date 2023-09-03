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
    col1.metric("Total Number of Customers", total_societies, f"+{societies_in_past_30_days}", help="Shows Total Registered Society and changes in the last 30 days")
    col2.metric("Total Number of Orders", f"{data['num_members'].sum()}", "-1", help="Active Members and changes in the last 30 days (**Note**: Contains sample data)")
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
    selected_sectors_all = st.sidebar.checkbox("Select All Sectors", value=True, key="all_sectors_checkbox")
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
    fig.update_layout(height=250, width=400)
    st.plotly_chart(fig, use_container_width=True)

    r1, r2 = st.columns((7, 3))

    with r1:

        st.markdown("#### Pie Chart")

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

   
Import Dataset & Libraries

from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud
     

# Mount Google Drive
drive.mount('/content/drive')
     
Mounted at /content/drive

from google.colab import drive
drive.mount('/content/drive')
     
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

# Import Dataset
file_path = '/content/drive/MyDrive/instagram_data.csv'
# Read the Excel file into a pandas DataFrame
df = pd.read_csv(file_path, encoding='utf-8')
     

df.head()
     
Post Date	Likes	Comments	Hashtags	Engagement
0	2022-11-01	115	3	skincare, naturalconfidence, beesline, n...	118
1	2022-11-04	269	13	skincare, naturalconfidence, beesline, n...	282
2	2022-11-07	201	3	skincare, naturalconfidence, beesline, n...	204
3	2022-11-09	328	3	skincare, sale, offers, singlesday, na...	331
4	2022-11-10	226	12	skincare, sale, offers, singlesday, na...	238
Likes & Comments Over Time

# Convert 'Post Date' column to datetime format
df['Post Date'] = pd.to_datetime(df['Post Date'])

# Create an interactive line plot
fig = px.line(df, x='Post Date', y=['Likes', 'Comments'], title='Fluctuation of Likes and Comments Over Time')

# Display the plot
fig.show()
     
Refining
There are extreme values. Further investigation revealed that these posts with high engagment are not acttually Beesline post, but those associated with influencer sponsorships. For this reason, they must be removed.


# Convert 'Post Date' column to datetime format
df['Post Date'] = pd.to_datetime(df['Post Date'])

# Dates to be removed
dates_to_remove = ['2023-05-31', '2023-05-15', '2023-05-10', '2023-05-03', '2023-05-02', '2023-07-11', '2023-07-19', '2023-07-20', '2023-07-28']
df = df[~df['Post Date'].isin(pd.to_datetime(dates_to_remove))]

# Create an interactive line plot
fig = px.line(df, x='Post Date', y=['Likes', 'Comments'], title='Fluctuation of Likes and Comments Over Time')

# Display the plot
fig.show()
     
Basic Exploration

# Convert 'Post Date' column to datetime format
df['Post Date'] = pd.to_datetime(df['Post Date'])

# Dates to be removed
dates_to_remove = ['2022-05-31', '2022-05-15', '2022-05-10', '2022-05-03', '2022-05-02']

# Create a new dataframe with rows except those with the dates_to_remove
filtered_df = df[~df['Post Date'].isin(pd.to_datetime(dates_to_remove))]

# Display the filtered dataframe
print(filtered_df)
     
     Post Date  Likes  Comments  \
0   2022-11-01    115         3   
1   2022-11-04    269        13   
2   2022-11-07    201         3   
3   2022-11-09    328         3   
4   2022-11-10    226        12   
..         ...    ...       ...   
135 2023-07-16    120         3   
136 2023-07-18     87         0   
139 2023-07-23     64         2   
140 2023-07-25    109         0   
143 2023-07-31    153         9   

                                              Hashtags  Engagement  
0    skincare,   naturalconfidence,   beesline,   n...         118  
1    skincare,   naturalconfidence,   beesline,   n...         282  
2    skincare,   naturalconfidence,   beesline,   n...         204  
3    skincare,   sale,   offers,   singlesday,   na...         331  
4    skincare,   sale,   offers,   singlesday,   na...         238  
..                                                 ...         ...  
135          instant_bright,   beesline,   PassTheGlow         123  
136                         instant_bright,   beesline          87  
139  passtheglow,   _,   beesline,   skincare,   in...          66  
140              beesline,   instantbright,   cleanser         109  
143           beesline,   instantbright,   passtheglow         162  

[130 rows x 5 columns]
Engagment & Hashtags

import pandas as pd
import plotly.graph_objs as go

# 1. Extract daily engagement:
daily_engagement = filtered_df.groupby('Post Date')['Engagement'].sum()

# 2. Identify special occasions and sale periods:
special_occasions = {
    'womensday': {'color': 'red', 'symbol': 'circle'},
    'womensmonth': {'color': 'red', 'symbol': 'circle'},
    'ramadankareem': {'color': 'black', 'symbol': 'star'},
    'ramadan': {'color': 'black', 'symbol': 'star'},
    'instantbright': {'color': 'blue', 'symbol': 'circle'},
    'instant_bright': {'color': 'blue', 'symbol': 'circle'},
    'passtheglow': {'color': 'blue', 'symbol': 'circle'},
    'sale': {'color': 'yellow', 'symbol': 'circle'},
    'offer': {'color': 'yellow', 'symbol': 'circle'}
}
sale_periods = {
    'yellowweek': {'color': 'yellow', 'symbol': 'circle'}
}

# 3. Create interactive plot:
traces = []

# Engagement Line
trace_engagement = go.Scatter(
    x=daily_engagement.index,
    y=daily_engagement.values,
    mode='lines+markers',
    name='Engagement',
    line=dict(color='orange')
)
traces.append(trace_engagement)

# Dotted Horizontal Line at y=232
trace_horizontal_line = go.Scatter(
    x=daily_engagement.index,
    y=[232] * len(daily_engagement.index),
    mode='lines',
    name='Target Engagement',
    line=dict(color='green', dash='dot')
)
traces.append(trace_horizontal_line)

# Create separate traces for special occasions and sales
for occasion, attributes in {**special_occasions, **sale_periods}.items():
    mask = filtered_df['Hashtags'].str.contains(occasion, case=False, na=False)
    special_dates = filtered_df[mask]['Post Date'].tolist()

    trace = go.Scatter(
        x=special_dates,
        y=[daily_engagement[date] for date in special_dates],
        mode='markers',
        marker=dict(size=10, color=attributes['color'], symbol=attributes['symbol']),
        name=occasion
    )
    traces.append(trace)

# Add video dates
video_dates = ['2022-11-10', '2022-11-21', '2022-11-30', '2022-12-09', '2023-02-28', '2023-04-07', '2023-05-08', "2023-05-25", "2023-05-28", '2023-06-02', '2023-06-05', '2023-06-07', '2023-06-09', "2023-06-12", "2023-06-18", '2023-06-25', '2023-06-27', '2023-07-02', '2023-07-13']


annotations = []
for date in video_dates:
    annotations.append(
        {
            "x": date,
            "y": daily_engagement[pd.Timestamp(date)],
            "xref": "x",
            "yref": "y",
            "text": "Video",
            "showarrow": True,
            "arrowhead": 2,
            "arrowsize": 1,
            "arrowwidth": 2,
            "arrowcolor": "#636363",
            "ax": 0,
            "ay": -30
        }
    )

layout = go.Layout(
    title='Daily Engagement with Special Occasions, Sales, and Videos Highlighted',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Engagement'),
    hovermode='closest',
    updatemenus=[
        dict(
            type="buttons",
            showactive=True,
            buttons=[
                dict(
                    label="Show Videos",
                    method="relayout",
                    args=["annotations", annotations]
                ),
                dict(
                    label="Hide Videos",
                    method="relayout",
                    args=["annotations", []]
                )
            ]
        )
    ]
)

fig = go.Figure(data=traces, layout=layout)
fig.show()

     
Engagement Monthly Over Time & Trend

import pandas as pd
import plotly.express as px
import numpy as np

# Assuming you have your filtered_df already loaded and ready

# Convert 'Post Date' column to datetime
filtered_df['Post Date'] = pd.to_datetime(filtered_df['Post Date'])

# Aggregate by month and compute the mean engagement
avg_monthly_engagement = filtered_df.groupby(filtered_df['Post Date'].dt.to_period("M")).mean()['Engagement'].reset_index()

# Convert back the 'Post Date' from period to datetime for better plot compatibility
avg_monthly_engagement['Post Date'] = avg_monthly_engagement['Post Date'].dt.strftime('%Y-%m')

# Calculate the trend line
z = np.polyfit(x=range(len(avg_monthly_engagement)), y=avg_monthly_engagement['Engagement'], deg=1)
p = np.poly1d(z)
avg_monthly_engagement['Trend'] = p(range(len(avg_monthly_engagement)))

# Plot
fig = px.line(avg_monthly_engagement, x='Post Date', y='Engagement', title='Average Monthly Engagement', labels={'Engagement': 'Average Engagement', 'Post Date': 'Month'})

# Add trend line
fig.add_traces(go.Scatter(x=avg_monthly_engagement['Post Date'], y=avg_monthly_engagement['Trend'], mode='lines', name='Trend Line', line=dict(dash='dash')))

fig.show()
     
<ipython-input-16-67d7f5ac688c>:11: FutureWarning:

The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.


correlation = filtered_df['Likes'].corr(filtered_df['Comments'])
print(f"The correlation between likes and comments is: {correlation:.2f}")
     
The correlation between likes and comments is: 0.64

import numpy as np

# Assuming you have your filtered_df already loaded and ready

# Convert 'Post Date' column to datetime
filtered_df['Post Date'] = pd.to_datetime(filtered_df['Post Date'])

# Aggregate by month and compute the mean engagement
avg_monthly_engagement = filtered_df.groupby(filtered_df['Post Date'].dt.to_period("M")).mean()['Engagement'].reset_index()

# Calculate the trend line
z = np.polyfit(x=range(len(avg_monthly_engagement)), y=avg_monthly_engagement['Engagement'], deg=1)

# Extract the slope (rate of change of engagement over time)
rate_of_change = z[0]
print("Rate of increase in engagement by month:", rate_of_change)
     
Rate of increase in engagement by month: 7.101889483065963
<ipython-input-25-9f30a1b26b62>:9: FutureWarning:

The default value of numeric_only in DataFrameGroupBy.mean is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.


total_comments = filtered_df['Comments'].sum()
print(f"Total Amount of Comments: {total_comments}")

avg_comment_per_post = filtered_df['Comments'].mean()
print(f"Average Comment per Post: {avg_comment_per_post:.2f}")

most_commented_value = filtered_df['Comments'].max()
date_most_commented = filtered_df[filtered_df['Comments'] == most_commented_value]['Post Date'].iloc[0]

print(f"Most Commented Post Date: {date_most_commented}")
print(f"Number of Comments in the Most Commented Post: {most_commented_value}")

least_commented_value = filtered_df['Comments'].min()
date_least_commented = filtered_df[filtered_df['Comments'] == least_commented_value]['Post Date'].iloc[0]

print(f"Least Commented Post Date: {date_least_commented}")
print(f"Number of Comments in the Least Commented Post: {least_commented_value}")
     
Total Amount of Comments: 1153
Average Comment per Post: 8.87
Most Commented Post Date: 2023-05-26 00:00:00
Number of Comments in the Most Commented Post: 59
Least Commented Post Date: 2022-11-26 00:00:00
Number of Comments in the Least Commented Post: 0

# Total Amount of Likes
total_likes = filtered_df['Likes'].sum()
print(f"Total Amount of Likes: {total_likes}")

# Average Likes per Post
avg_likes_per_post = filtered_df['Likes'].mean()
print(f"Average Likes per Post: {avg_likes_per_post:.2f}")

# Most Liked Post & the Date of This Post
most_liked_value = filtered_df['Likes'].max()
date_most_liked = filtered_df[filtered_df['Likes'] == most_liked_value]['Post Date'].iloc[0]
print(f"Most Liked Post Date: {date_most_liked}")
print(f"Number of Likes in the Most Liked Post: {most_liked_value}")

# Least Liked Post & the Date for It
least_liked_value = filtered_df['Likes'].min()
date_least_liked = filtered_df[filtered_df['Likes'] == least_liked_value]['Post Date'].iloc[0]
print(f"Least Liked Post Date: {date_least_liked}")
print(f"Number of Likes in the Least Liked Post: {least_liked_value}")
     
Total Amount of Likes: 29060
Average Likes per Post: 223.54
Most Liked Post Date: 2023-05-25 00:00:00
Number of Likes in the Most Liked Post: 1808
Least Liked Post Date: 2023-04-22 00:00:00
Number of Likes in the Least Liked Post: 42

# Total Amount of Engagement
total_engagement = filtered_df['Engagement'].sum()
print(f"Total Amount of Engagement: {total_engagement}")

# Average Engagement per Post
avg_engagement_per_post = filtered_df['Engagement'].mean()
print(f"Average Engagement per Post: {avg_engagement_per_post:.2f}")

# Post with the Highest Engagement & Its Date
highest_engagement_value = filtered_df['Engagement'].max()
date_highest_engagement = filtered_df[filtered_df['Engagement'] == highest_engagement_value]['Post Date'].iloc[0]
print(f"Post with the Highest Engagement Date: {date_highest_engagement}")
print(f"Engagement in the Post with the Highest Engagement: {highest_engagement_value}")

# Post with Least Engagement & Its Date
least_engagement_value = filtered_df['Engagement'].min()
date_least_engagement = filtered_df[filtered_df['Engagement'] == least_engagement_value]['Post Date'].iloc[0]
print(f"Post with the Least Engagement Date: {date_least_engagement}")
print(f"Engagement in the Post with the Least Engagement: {least_engagement_value}")
     
Total Amount of Engagement: 30213
Average Engagement per Post: 232.41
Post with the Highest Engagement Date: 2023-05-25 00:00:00
Engagement in the Post with the Highest Engagement: 1863
Post with the Least Engagement Date: 2023-04-22 00:00:00
Engagement in the Post with the Least Engagement: 43
WordCloud (Based on Engagement)

# Function to replace the specific hashtags
def combine_hashtags(hashtag_str):
    hashtag_str = hashtag_str.replace('instant_bright', 'instantbright')
    return hashtag_str

# Apply the function to the Hashtags column
filtered_df['Hashtags'] = filtered_df['Hashtags'].apply(combine_hashtags)
     

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming you already have the 'filtered_df' dataset loaded.

# Extract all hashtags and engagements from the dataframe
hashtag_list = []
engagement_list = []

for idx, row in filtered_df.iterrows():
    hashtags = row['Hashtags'].split(', ')
    engagement = row['Engagement']
    for hashtag in hashtags:
        hashtag = hashtag.strip().lower()  # Convert to lowercase and strip spaces
        if hashtag == 'instant_bright':
            hashtag = 'instantbright'  # Consolidate
        hashtag_list.append(hashtag)
        engagement_list.append(engagement)

# Create a DataFrame from the lists
df_hashtags = pd.DataFrame({
    'Hashtag': hashtag_list,
    'Engagement': engagement_list
})

# Group by hashtag and sum the engagements
hashtag_engagement = df_hashtags.groupby('Hashtag')['Engagement'].sum().sort_values(ascending=False)

# Remove specified hashtags
hashtags_to_remove = ['natureinourcosmetics', 'naturalconfidence', 'passtheglow', 'beesline']
hashtag_engagement = hashtag_engagement[~hashtag_engagement.index.isin(hashtags_to_remove)]

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(hashtag_engagement)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
     


# Extract the top 5 hashtags and their engagement points
top_5_hashtags = hashtag_engagement.head(5)

print("Top 5 Hashtags and Their Engagement Points:")
print(top_5_hashtags)
     
Top 5 Hashtags and Their Engagement Points:
Hashtag
skincare         25073
instantbright     5996
offers            3602
sale              3602
moisturize        2907
Name: Engagement, dtype: int64

d# Print the list of hashtags used in the word cloud
hashtags_in_wordcloud = hashtag_engagement.index.tolist()

for hashtag in hashtags_in_wordcloud:
    print(hashtag)
     
skincare
instantbright
offers
sale
moisturize
yellowweek
ramadan
rollon
womensmonth
brightening
singlesday
sunscreen
sensitivezone
beegan
glowserum
micellarwater
deodorant
morningroutine
cream
lipcare
skin
ramadankareem
eyecream
bright
wildbeeconservation
natural
beeday
holiday
skincancerawarnessmonth
skincancer
protection
guardiansofthewild
youngguardians
womensday
women
facemasks
worldbeeday
lipbalm
hydration
mothersday
cleanser
newyear
_
instant_affirmation
festiveseason
launch
earthday
Post Frequency & Engagement

import pandas as pd
import plotly.graph_objs as go

# Assuming 'filtered_df' is already loaded

# 1. Aggregate data by Month:
filtered_df['Month-Year'] = filtered_df['Post Date'].dt.to_period("M")
monthly_data = filtered_df.groupby('Month-Year').agg({'Engagement': 'sum', 'Post Date': 'count'}).rename(columns={'Post Date': 'Number of Posts'})

# 2. Plotting:
trace1 = go.Bar(
    x=monthly_data.index.astype(str),
    y=monthly_data['Number of Posts'],
    name='Number of Posts',
    yaxis='y1'
)

trace2 = go.Scatter(
    x=monthly_data.index.astype(str),
    y=monthly_data['Engagement'],
    name='Engagement',
    yaxis='y2'
)

layout = go.Layout(
    title='Frequency of Posts vs Engagement per Month',
    xaxis=dict(title='Month-Year'),
    yaxis=dict(title='Number of Posts'),
    yaxis2=dict(title='Engagement', overlaying='y', side='right'),
    hovermode='closest'
)

fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()

# 3. Correlation Test:
correlation = monthly_data['Number of Posts'].corr(monthly_data['Engagement'])
print(f"Pearson correlation coefficient between frequency of posts and engagement: {correlation:.2f}")
     
Pearson correlation coefficient between frequency of posts and engagement: 0.87
            

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
    st.markdown("## Data Visulaization")


container = st.container()
col1,col2 = st.columns(2)



@st.cache_resource
def load_data(file):
    """
    Load data from a file (CSV or Excel).

    Parameters:
        file (File): The file to load.

    Returns:
        DataFrame: The loaded data.
    """
    file_extension = file.name.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(file)
    elif file_extension in ["xls", "xlsx"]:
        data = pd.read_excel(file)
    else:
        st.warning("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    return data


def select_columns(df):
    st.write("### Select Columns")
    all_columns = df.columns.tolist()
    #options_key = "_".join(all_columns)
    selected_columns = st.multiselect("Select columns", options=all_columns)
    
    if selected_columns:
        sub_df = df[selected_columns]
        st.write("### Sub DataFrame")
        st.write(sub_df.head())
    else:
        st.warning("Please select at least one column.")

def select_and_rename_column(df):
    st.write("### Select and Rename Columns")
    
    # Select columns to rename
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to rename", options=all_columns)
    
    # Rename the selected columns
    for column in selected_columns:
        new_column_name = st.text_input(f"Enter new name for column '{column}'", value=column)
        if column != new_column_name:
            df.rename(columns={column: new_column_name}, inplace=True)
            st.write(f"Column '{column}' renamed as '{new_column_name}' successfully!")
    
    return df    


def show_missing_values_percentage(df):
    st.write("### Missing Values Percentage")
    
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().sum() / len(df) * 100
    
    # Create a DataFrame to store the missing values percentage
    missing_df = pd.DataFrame({'Column': missing_percentage.index, 'Missing Percentage': missing_percentage.values})
    
    # Display the missing values percentage DataFrame
    st.write("Percentage of missing values",missing_df)


#aggregation funtion
def agg(df):
    # Allow the user to select columns for aggregation
    aggregation_columns = st.multiselect("Select columns for aggregation", options=df.columns)
    
    # Allow the user to select an aggregation function
    aggregation_function = st.selectbox("Select an aggregation function", options=["Sum", "Mean", "Median"])
    
    # Perform the aggregation
    if aggregation_columns:
        if aggregation_function == "Sum":
            aggregated_values = sub_df[aggregation_columns].sum()
        elif aggregation_function == "Mean":
            aggregated_values = sub_df[aggregation_columns].mean()
        elif aggregation_function == "Median":
            aggregated_values = sub_df[aggregation_columns].median()
        
        # Display the aggregated values
        st.write(f"Aggregated {aggregation_function} for {aggregation_columns}")
        st.write(aggregated_values)    

#remove duplicats
def remove_duplicates(df):
    st.write("### Remove Duplicates")
    
    # Select columns for identifying duplicates
    columns = st.multiselect("Select columns for identifying duplicates", options=df.columns)
    
    if columns:
        # Remove duplicates based on selected columns
        df.drop_duplicates(subset=columns, inplace=True)
        
        st.write("Duplicates removed successfully!")
        
    return df
#search and replace a value in column
def search_and_replace(df):
    st.write("### Search and Replace")
    
    # Select a column to search and replace
    column = st.selectbox("Select a column", options=df.columns)
    
    if column:
        # Get the search string from the user
        search_string = st.text_input("Enter the search string")
        
        # Get the replace value from the user
        replace_value = st.text_input("Enter the replace value")
        
        # Perform the search and replace operation
        if search_string in df[column].values:
            df[column] = df[column].replace(search_string, replace_value)
            st.write("Search and replace completed!")
            st.write(df[column])

        else:
            st.warning("The search string is not present in the selected column.")
        

#Change columns datatypes 
import streamlit as st
import pandas as pd

def change_column_data_types(df):
    st.write("### Change Column Data Types")
    
    # Select columns to change data types
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect("Select columns to change data types", options=all_columns)
    
    # Get the new data types from the user
    new_data_types = {}
    for column in selected_columns:
        st.write(f"Column: {column}")
        current_data_type = df[column].dtype
        st.write(f"Current Data Type: {current_data_type}")
        new_data_type = st.selectbox("Select new data type", options=['object', 'int', 'float', 'datetime', 'boolean'])
        new_data_types[column] = new_data_type
    
    # Create a copy of the DataFrame to modify
    modified_df = df.copy()
    
    # Change the data types of selected columns
    for column, data_type in new_data_types.items():
        try:
            if data_type == 'object':
                modified_df[column] = modified_df[column].astype(str)
            elif data_type == 'int':
                modified_df[column] = pd.to_numeric(modified_df[column], errors='coerce', downcast='integer')
            elif data_type == 'float':
                modified_df[column] = pd.to_numeric(modified_df[column], errors='coerce', downcast='float')
            elif data_type == 'datetime':
                modified_df[column] = pd.to_datetime(modified_df[column], errors='coerce')
            elif data_type == 'boolean':
                modified_df[column] = modified_df[column].astype(bool)
            
            st.write(f"Column '{column}' data type changed to '{data_type}' successfully!")
        except Exception as e:
            st.error(f"Error occurred while changing data type of column '{column}': {str(e)}")
    
    return modified_df
def groupby_aggregate_data(sub_df):
    st.write("### Grouping and Aggregating Data")
    st.write(sub_df.head())
    
    # Get the list of columns from the DataFrame
    columns = sub_df.columns.tolist()

    # Get the categorical columns for grouping
    group_columns = st.multiselect("Select categorical columns for grouping", columns)

    # Get the numerical columns for aggregation
    numerical_columns = st.multiselect("Select numerical columns for aggregation", columns)

    # Get the aggregation functions from the user
    #aggregation_functions = st.multiselect("Select aggregation functions", ['sum', 'mean', 'median', 'min', 'max'])
    
    # Create the aggregation dictionary
    #aggregation = {col: func for col in numerical_columns for func in aggregation_functions}

    # Perform grouping and aggregation
    if group_columns and numerical_columns:
        grouped_dff = sub_df.groupby(group_columns)[numerical_columns].agg(['sum', 'mean', 'median', 'min', 'max'])
        grouped_df = grouped_dff.reset_index()  # Reset index to display category names
       
        st.write("### Grouped and Aggregated Data")
        st.write(grouped_df)
        #fig = px.bar(grouped_df, x=grouped_df.index, y=['sum'], barmode='group')
    else:
        st.warning("Please select at least one categorical column, one numerical column, and one aggregation function.")
  
       
def analyze_data(data):

    container = st.container()
    col1,col2 = st.columns(2)
    
    with container:
         st.write("File Header",data.head())
    with col1:
         st.write("Columns in you file are ",data.columns)
    st.write("### Select Columns to make your Data Set for Analysis")
    
    with col2:
        st.write("Data Types " ,data.dtypes)

        all_columns = [str(col) for col in data.columns]
        options_key = "_".join(all_columns)
        selected_columns = st.multiselect("Select columns", options=all_columns)    
    if selected_columns:
        sub_df = data[selected_columns]
        sub_df = select_and_rename_column(sub_df)
        st.write("### Sub DataFrame")
        st.write(sub_df.head())

        remove_duplicates(sub_df)
        
        change_column_type_df = change_column_data_types(sub_df)
        st.write("Columns Types are changed",change_column_type_df)
        st.write("Description")
        st.write(change_column_type_df.describe().T)
        st.write("Data Rank")
        st.write(change_column_type_df.rank())

        st.subheader("Sort Data")
        sort_column = st.selectbox("Select column for sorting", change_column_type_df.columns)
        sorted_df = change_column_type_df.sort_values(by=sort_column)
        st.write(sorted_df)

        #show_missing_values_percentage(sub_df)

        st.write(corr(change_column_type_df))
        
        show_missing_values(change_column_type_df)
        show_percent_missing(change_column_type_df)
        show_unique_values(change_column_type_df)
        show_standard_deviation(change_column_type_df)
        show_data_shape(change_column_type_df)
        show_data_correlation(change_column_type_df)
        filter_rows(change_column_type_df)
    
        groupby_aggregate_data(sub_df)
    
        

        search_and_replace(sub_df)



    else:
        st.warning("Please select at least one column.")


def show_file_header(data):
    st.write("File Header")
    st.write(data.head())

def sort_data(data):
    # Sort the data by a selected column
    sort_column = st.selectbox("Select column to sort by", data.columns)
    sorted_df = data.sort_values(by=sort_column)
    return sorted_df


def show_sorted_data(sorted_df):
    st.write("Sort Data")
    st.write(sorted_df)


def show_missing_values(data):
    #col1 = st.beta_column()
    st.write("Missing Values")
    st.write(data.isnull().sum())

def show_percent_missing(data):
    st.write("Missing Percentage")
    st.write(data.isna().mean().mul(100))



def show_unique_values(data):
    #col2 = st.beta_column()
    st.write("Unique Values")
    st.write(data.nunique())


def show_standard_deviation(data):
    #col1 = st.beta_column()
    st.write("Standard Deviation")
    st.write(data.std(numeric_only=True))


def show_data_shape(data):
    #col1, col2 = st.beta_columns(2)
    st.write("Number of rows")
    st.write(data.shape[0])
    st.write("Number of columns")
    st.write(data.shape[1])


def show_data_correlation(data):
    #col1 = st.beta_column()
    st.write("Data Correlation")
    st.write(data.corr(numeric_only=True))

def corr(data):
    st.write("Data correlation")
    st.write(data.corr(numeric_only=True).style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))  


def filter_rows(data):
    
    column_name = st.selectbox("Select a column to filter", data.columns)
    value = st.text_input("Enter the filter value")
    # Filter the rows based on the converted column
    if value == "":
        filtered_data = data[data[column_name].isnull()]
    elif data[column_name].dtype == 'float':
          filtered_data = data[data[column_name] >= float(value)]
    else:      
        filtered_data = data[data[column_name].astype(str).str.contains(value, case=False)]
    st.write("Filtered Data")
    st.write(filtered_data)    


def create_chart(chart_type, data, x_column, y_column):

    container.write(" # Data Visualization # ")
    if chart_type == "Bar":
    
        st.header("Bar Chart")
        
        color_column = st.sidebar.selectbox("Select column for color ", data.columns,key="color_name")
        #pattern_column = st.sidebar.selectbox("Select column for pattern ", data.columns)
        if color_column:
           fig = px.bar(data, x=x_column, y=y_column,color=color_column,barmode="group")
           st.plotly_chart(fig)
        else:
           fig = px.bar(data, x=x_column, y=y_column,barmode="group")
           st.plotly_chart(fig)   

    elif chart_type == "Line":
        st.header("Line Chart")
        fig = px.line(data, x=x_column, y=y_column)
        st.plotly_chart(fig)

    elif chart_type == "Scatter":
        st.header("Scatter Chart")
        size_column = st.sidebar.selectbox("Select column for size ", data.columns)
        color_column = st.sidebar.selectbox("Select column for color ", data.columns)
        if color_column:
            
           fig = px.scatter(data, x=x_column, y=y_column,color=color_column,size=size_column)

        else:
            fig = px.scatter(data, x=x_column, y=y_column) 
        st.plotly_chart(fig)        

    elif chart_type == "Histogram":
        st.header("Histogram Chart")
        color_column = st.sidebar.selectbox("Select column for color ", data.columns)
        fig = px.histogram(data, x=x_column, y=y_column,color = color_column)
        st.plotly_chart(fig)
        

    elif chart_type == "Pie":
        st.header("Pie Chart")

        color_column = st.sidebar.selectbox("Select column for color ", data.columns)
        if color_column:
            fig = px.pie(data, names=x_column, values=y_column, color=color_column)
            st.plotly_chart(fig)
        else:
            fig = px.pie(data, names=x_column, values=y_column)
            st.plotly_chart(fig)
    
    

def main():

  
    image = Image.open("beesline.png")
    container.image(image, width=200)
    container.write(" #   Beesline Data Analysis and Visualization # ")
    
    st.sidebar.image(image, width=50)
    file_option = st.sidebar.radio("Data Source", options=["Upload Local File", "Enter Online Dataset"])
    file = None
    data = None

    if file_option == "Upload Local File":
        file = st.sidebar.file_uploader("Upload a data set in CSV or EXCEL format", type=["csv", "excel"])

    elif file_option == "Enter Online Dataset":
        online_dataset = st.sidebar.text_input("Enter the URL of the online dataset")
        if online_dataset:
            try:
                response = requests.get(online_dataset)
                if response.ok:
                    data = pd.read_csv(online_dataset)
                else:
                    st.warning("Unable to fetch the dataset from the provided link.")
            except:
                st.warning("Invalid URL or unable to read the dataset from the provided link.")

    options = st.sidebar.radio('Pages', options=['Data Analysis', 'Data visualization'])

    if file is not None:
        data = load_data(file)

    if options == 'Data Analysis':
        if data is not None:
            analyze_data(data)
        else:
            st.warning("No file or empty file")

    elif options == 'Data visualization':
        if data is not None:
            # Create a sidebar for user options
            st.sidebar.title("Chart Options")


            st.write("### Select Columns")
            all_columns = data.columns.tolist()
            options_key = "_".join(all_columns)
            selected_columns = st.sidebar.multiselect("Select columns", options=all_columns)
            if selected_columns:
                sub_df = data[selected_columns]


                chart_type = st.sidebar.selectbox("Select a chart type", ["Bar", "Line", "Scatter", "Histogram", "Pie"])

                x_column = st.sidebar.selectbox("Select the X column", sub_df.columns)

                y_column = st.sidebar.selectbox("Select the Y column", sub_df.columns)

                create_chart(chart_type, sub_df, x_column, y_column)

    
       

if __name__ == "__main__":
    main()
