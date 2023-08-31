
# CRCS Dashboard
This is a Streamlit application for the CRCS (Cooperative Registration and Compliance System) dashboard. The dashboard provides various functionalities for managing cooperative societies and analyzing registration data.

## Features:
  **Dashboard**: Provides an overview of registered societies and key metrics such as registered societies count, active members count, and events organized. Users can apply filters to analyze registration trends and sector distribution.
  
  **Registration**: Allows users to register new cooperative societies by providing necessary details such as society name, address, state, district, area of operation, and sector type. The registration data is stored in a CSV file.
  
  **Amendments and Appeals**: Users can submit requests for amendments or appeals related to registered societies. They need to provide the society ID, change details, and appeal reason. The requests are saved in the dataset with a pending status.
  
  **Society Details**: Users can view details of registered societies by filtering using the society ID. The information displayed includes society name, address, state, district, registration date, area of operation, and sector type.
  
  **Data Analytics**: Provides insights into the registered societies' data. Users can apply filters based on state, sector, and district to analyze the data. The number of registered societies and key statistics are displayed.

## Installation
To run the CRCS Dashboard locally, follow these steps:

```
Clone the repository:
git clone https://github.com/sathish-1804/crcs_dashboardv2.git

Change the directory to the cloned repository:
cd your_repository

Install the required dependencies:
pip install -r requirements.txt
```

To launch the CRCS Dashboard, run the following command:
```
streamlit run app.py
```

The dashboard will be accessible at http://localhost:8501 in your web browser.


**Note**: 
Initially dashboard is connected with mongodb server but due to server issue and other technically issue, I've decided to do with the local dataset
The dashboard uses a CSV file (datasetv2.csv) as the data source. Make sure to have the file in the same directory as the app.py script.
The dashboard supports registration, filtering, and analysis based on the data available in the CSV file. Ensure that the file contains the required columns for proper functioning.
The dashboard may require additional configuration or modifications based on specific requirements or use cases.
