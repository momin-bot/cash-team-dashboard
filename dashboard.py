import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import os
from datetime import timedelta
import plotly.express as px

# Set page config with light theme (Relative path for icon)
st.set_page_config(page_title="CashTeam Dashboard", layout="wide", page_icon="unnamed.png")

# Header
st.title("CashTeam Dashboard")

# --- Constants & Config ---
CREDENTIALS_FILE = 'credentials.json'
SPREADSHEET_ID = "1hJVuerNSCLtWhECv7q1yFT0IRjsX853lqvJ3ryCjXbk"
DRIVE_FOLDER_NAME = "CashTeamData" # Folder to look for in Drive
MASTER_CSV_NAME = "nonce_sales_master.csv"

# --- Credentials ---
def get_credentials():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    if os.path.exists(CREDENTIALS_FILE):
        return Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
    elif "gcp_service_account" in st.secrets:
        return Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    return None

# --- Drive Manager Class ---
class DriveManager:
    def __init__(self, creds):
        self.service = build('drive', 'v3', credentials=creds)
    
    def find_folder(self, folder_name):
        """Find folder ID by name."""
        query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None

    def find_file(self, file_name, parent_id=None):
        """Find file ID by name, optionally in a folder."""
        query = f"name='{file_name}' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if files:
            return files[0]['id']
        return None

    def download_csv(self, file_id):
        """Download CSV content as DataFrame."""
        request = self.service.files().get_media(fileId=file_id)
        file_content = io.BytesIO(request.execute())
        return pd.read_csv(file_content)

    def upload_csv(self, df, file_name, folder_id):
        """Upload DataFrame as CSV to Drive (Overwrite or Create)."""
        try:
            csv_buffer = io.BytesIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0) # IMPORTANT: Reset buffer position
            
            file_metadata = {'name': file_name, 'parents': [folder_id]}
            media = MediaIoBaseUpload(csv_buffer, mimetype='text/csv', resumable=True)
            
            # Check if exists to update or create
            existing_id = self.find_file(file_name, folder_id)
            
            if existing_id:
                # Update existing
                self.service.files().update(
                    fileId=existing_id,
                    media_body=media
                ).execute()
            else:
                # Create new
                self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
        except Exception as e:
            st.error(f"❌ Upload Failed! Error: {e}")
            st.info("💡 Tip: Check if the Service Account has **Editor** access to the 'CashTeamData' folder in Google Drive.")
            # If we have detailed error content, show it
            if hasattr(e, 'content'):
                st.code(e.content.decode('utf-8'))
            raise e

# --- Functions ---

@st.cache_data(ttl=600)
def get_google_sheet_data(sheet_name):
    """Fetch data from a specific worksheet in the Google Sheet."""
    try:
        creds = get_credentials()
        if not creds:
            st.error("Credentials not found.")
            return pd.DataFrame()

        client = gspread.authorize(creds)
        sh = client.open_by_key(SPREADSHEET_ID)
        worksheet = sh.worksheet(sheet_name)
        data = worksheet.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error fetching sheet '{sheet_name}': {e}")
        return pd.DataFrame()

def sync_data_with_drive(uploaded_files):
    """
    1. Downloads Master CSV from Drive.
    2. Appends new Uploaded CSVs.
    3. Deduplicates.
    4. Uploads updated Master CSV back to Drive.
    Returns the unified DataFrame.
    """
    creds = get_credentials()
    if not creds:
        st.error("Missing Credentials")
        return pd.DataFrame()
    
    drive = DriveManager(creds)
    
    # 1. Find/Create Folder
    folder_id = drive.find_folder(DRIVE_FOLDER_NAME)
    if not folder_id:
        st.warning(f"Folder '{DRIVE_FOLDER_NAME}' not found in Drive. Please create it and share with the service account.")
        # Fallback: Just return uploaded data if no folder
        return load_local_processing(uploaded_files)

    # 2. Load Master Master from Drive
    master_id = drive.find_file(MASTER_CSV_NAME, folder_id)
    if master_id:
        with st.spinner("Downloading Master Database from Drive..."):
            master_df = drive.download_csv(master_id)
    else:
        master_df = pd.DataFrame()

    # 3. Load & Append New Files
    new_data = []
    if uploaded_files:
        for f in uploaded_files:
            try:
                new_data.append(pd.read_csv(f))
            except Exception as e:
                st.error(f"Error reading {f.name}: {e}")
    
    if new_data:
        new_df = pd.concat(new_data, ignore_index=True)
        combined_df = pd.concat([master_df, new_df], ignore_index=True)
    else:
        combined_df = master_df

    if combined_df.empty:
        return pd.DataFrame()

    # 4. Deduplicate (Logic: 'id' is unique key)
    if 'id' in combined_df.columns:
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['id'], keep='first') # Keep existing (first) or new? keeping first usually safer for history
        after = len(combined_df)
        
        # If new data was added, Upload back to Drive
        if new_data and (after > len(master_df) or (len(master_df) == 0 and not master_df.empty)): 
             # Simplest check: just upload if we have new files. 
             # Efficient check: only if rows increased or it's a fresh file.
             pass 
        
        if uploaded_files: # Always sync back if user uploaded something, to be safe
            with st.spinner("Syncing updated database to Google Drive..."):
                drive.upload_csv(combined_df, MASTER_CSV_NAME, folder_id)
                st.sidebar.success(f"✅ Database Updated! Total Records: {len(combined_df)}")
    
    return combined_df

def load_local_processing(uploaded_files):
    """Fallback if no Drive connection"""
    if not uploaded_files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in uploaded_files]
    return pd.concat(dfs, ignore_index=True)

def process_sales_data(df):
    """Apply transformations (Time, Machine ID lookup) to the loaded DF."""
    if df.empty:
        return df

    # Time Correction
    if 'created_at_utc' in df.columns:
        df['created_at_utc'] = pd.to_datetime(df['created_at_utc'])
        df['Correct_Time'] = df['created_at_utc'] - timedelta(hours=5)
        df['Date'] = df['Correct_Time'].dt.date
    
    # Machine Lookup
    machines_df = get_google_sheet_data("machines")
    if not machines_df.empty and 'atm.id' in df.columns:
        # machine_ids_in_sheet = set(machines_df.iloc[:, 1].dropna().astype(str).tolist()) # Dangerous relying on index
        # Let's try to find column by name 'Machine ID'
        id_col = next((c for c in machines_df.columns if 'machine' in c.lower() and 'id' in c.lower()), None)
        if id_col:
             machine_ids_in_sheet = set(machines_df[id_col].dropna().astype(str).tolist())
             df['Sales_Type'] = df['atm.id'].apply(lambda x: "BS-Sales" if str(x) in machine_ids_in_sheet else "Other-Sales")
             df['BS-Sales'] = (df['Sales_Type'] == "BS-Sales").astype(int)
             df['Other-Sales'] = (df['Sales_Type'] == "Other-Sales").astype(int)
    
    return df

# --- Main Dashboard Logic ---

# Sidebar: File Uploader
st.sidebar.header("Data Connection")
st.sidebar.info("📂 **Storage**: Google Drive (Smart Append)")

uploaded_files = st.sidebar.file_uploader("Upload New Sales CSVs", type="csv", accept_multiple_files=True)

# Main Data Loader
# We load data if: 1. Files uploaded OR 2. Drive has data (Auto-load on startup)
# To avoid excessive API calls on every interaction, you might want to cache this heavily, 
# but for now we'll trigger it.

sales_df = sync_data_with_drive(uploaded_files)
sales_df = process_sales_data(sales_df)

if not sales_df.empty:
    cim_df = get_google_sheet_data("CIM")
    
    # Sidebar Filters
    st.sidebar.header("Filter Dashboard")
    
    # Ensure Date column exists
    if 'Date' not in sales_df.columns:
        st.error("Data missing 'created_at_utc' column. Cannot filter by date.")
        st.stop()

    min_date = sales_df['Date'].min()
    max_date = sales_df['Date'].max()
    
    date_option = st.sidebar.selectbox(
        "Select Time Period",
        ["Custom Range", "Today", "Yesterday", "Last 7 Days", "Last 30 Days", "This Month", "All Time"],
        index=6
    )
    
    from datetime import date
    today = date.today()
    
    if date_option == "Today":
        start_date, end_date = today, today
    elif date_option == "Yesterday":
        start_date, end_date = today - timedelta(days=1), today - timedelta(days=1)
    elif date_option == "Last 7 Days":
        start_date, end_date = today - timedelta(days=7), today
    elif date_option == "Last 30 Days":
        start_date, end_date = today - timedelta(days=30), today
    elif date_option == "This Month":
        start_date, end_date = today.replace(day=1), today
    elif date_option == "All Time":
        start_date, end_date = min_date, max_date
    else:
        # Custom Range
        date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

    # Apply filter
    filtered_df = sales_df[(sales_df['Date'] >= start_date) & (sales_df['Date'] <= end_date)]

    # --- Top Metrics ---
    # Get deposits data for metrics (filtered by date range)
    deposits_df = get_google_sheet_data("deposits")
    total_deposits = 0
    if not deposits_df.empty:
        date_col_dep = [col for col in deposits_df.columns if 'date' in col.lower() and 'ref' not in col.lower()]
        total_col = [col for col in deposits_df.columns if 'total' in col.lower()]
        if total_col and date_col_dep:
            deposits_df[total_col[0]] = deposits_df[total_col[0]].astype(str).replace(r'[\$,]', '', regex=True)
            deposits_df[total_col[0]] = pd.to_numeric(deposits_df[total_col[0]], errors='coerce').fillna(0)
            deposits_df[date_col_dep[0]] = pd.to_datetime(deposits_df[date_col_dep[0]], errors='coerce')
            # Filter by date range
            deposits_filtered = deposits_df[
                (deposits_df[date_col_dep[0]].dt.date >= start_date) & 
                (deposits_df[date_col_dep[0]].dt.date <= end_date)
            ]
            total_deposits = deposits_filtered[total_col[0]].sum()
    
    # Get forecast data for metrics (filtered by date range)
    forecast_df_metrics = get_google_sheet_data("Forecast")
    total_forecast = 0
    if not forecast_df_metrics.empty:
        date_col_for = [col for col in forecast_df_metrics.columns if 'date' in col.lower() and 'ref' not in col.lower()]
        med_forecast_col = [col for col in forecast_df_metrics.columns if 'med' in col.lower() and 'forecast' in col.lower()]
        if med_forecast_col and date_col_for:
            forecast_df_metrics[med_forecast_col[0]] = forecast_df_metrics[med_forecast_col[0]].astype(str).replace(r'[\$,]', '', regex=True)
            forecast_df_metrics[med_forecast_col[0]] = pd.to_numeric(forecast_df_metrics[med_forecast_col[0]], errors='coerce').fillna(0)
            forecast_df_metrics[date_col_for[0]] = pd.to_datetime(forecast_df_metrics[date_col_for[0]], errors='coerce')
            # Filter by date range
            forecast_filtered = forecast_df_metrics[
                (forecast_df_metrics[date_col_for[0]].dt.date >= start_date) & 
                (forecast_df_metrics[date_col_for[0]].dt.date <= end_date)
            ]
            total_forecast = forecast_filtered[med_forecast_col[0]].sum()
    
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    total_fiat = filtered_df['fiat'].sum() if 'fiat' in filtered_df.columns else 0
    
    # Calculate fiat sums for BS and Other sales
    if 'Sales_Type' in filtered_df.columns and 'fiat' in filtered_df.columns:
        bs_fiat = filtered_df[filtered_df['Sales_Type'] == 'BS-Sales']['fiat'].sum()
        other_fiat = filtered_df[filtered_df['Sales_Type'] == 'Other-Sales']['fiat'].sum()
    else:
        bs_fiat = 0
        other_fiat = 0

    m1.metric("Total Sales", f"${total_fiat:,.2f}")
    m2.metric("BS-Sales", f"${bs_fiat:,.2f}")
    m3.metric("Other-Sales", f"${other_fiat:,.2f}")
    m4.metric("Total Deposits", f"${total_deposits:,.2f}")
    m5.metric("Total Forecast", f"${total_forecast:,.2f}")
    m6.metric("Transactions", f"{len(filtered_df):,}")

    # --- Charts ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trends", "Sales by State", "Sales vs Deposits", "Geographic (CIM)", "Data Explorer"])

    with tab1:
        st.subheader("Sales Trend (BS vs Other)")
        if 'Date' in filtered_df.columns and 'Sales_Type' in filtered_df.columns and 'fiat' in filtered_df.columns:
            trend_df = filtered_df.groupby(['Date', 'Sales_Type'])['fiat'].sum().reset_index(name='Amount')
            fig = px.line(trend_df, x='Date', y='Amount', color='Sales_Type', markers=True, title="Daily Sales Amount by Type")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Sales by State")
        # Lookup state from machines sheet using atm.id -> MachineID
        machines_df_state = get_google_sheet_data("machines")
        
        if not machines_df_state.empty and 'atm.id' in filtered_df.columns and 'fiat' in filtered_df.columns:
            # Find MachineID column
            machine_id_col = [c for c in machines_df_state.columns if 'machine' in c.lower() and 'id' in c.lower()]
            state_col_machines = [c for c in machines_df_state.columns if c.lower() == 'state']
            
            if machine_id_col and state_col_machines:
                # Create lookup dictionary: MachineID -> State
                machines_lookup = machines_df_state[[machine_id_col[0], state_col_machines[0]]].copy()
                machines_lookup.columns = ['MachineID', 'State']
                machines_lookup['MachineID'] = machines_lookup['MachineID'].astype(str)
                
                # Map state to sales data
                sales_with_state = filtered_df.copy()
                sales_with_state['atm.id'] = sales_with_state['atm.id'].astype(str)
                sales_with_state = sales_with_state.merge(machines_lookup, left_on='atm.id', right_on='MachineID', how='left')
                
                # Aggregate by state
                state_sales = sales_with_state.groupby('State')['fiat'].sum().reset_index()
                state_sales.columns = ['State', 'Total Sales']
                state_sales = state_sales[state_sales['State'].notna() & (state_sales['Total Sales'] > 0)]
                state_sales = state_sales.sort_values('Total Sales', ascending=False)
                
                # Bar chart
                fig_state = px.bar(state_sales, x='State', y='Total Sales', title="Sales by State", 
                                  color='Total Sales', color_continuous_scale='Blues')
                st.plotly_chart(fig_state, use_container_width=True)
                
                # Show table
                st.dataframe(state_sales, use_container_width=True)
            else:
                st.warning(f"Could not find MachineID or State columns in machines sheet.")
        else:
            st.warning("Missing data for state lookup.")

    with tab3:
        st.subheader("Sales vs Deposits vs Forecast")
        # Get deposits and forecast data
        deposits_df_chart = get_google_sheet_data("deposits")
        forecast_df = get_google_sheet_data("Forecast")
        
        if not deposits_df_chart.empty and 'Date' in filtered_df.columns:
            # Aggregate daily sales (fiat) - normalize date to datetime
            daily_sales = filtered_df.copy()
            daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
            daily_sales = daily_sales.groupby('Date')['fiat'].sum().reset_index()
            daily_sales.rename(columns={'fiat': 'Sales'}, inplace=True)
            
            # Process deposits
            date_col = [col for col in deposits_df_chart.columns if 'date' in col.lower() and 'ref' not in col.lower()]
            total_col = [col for col in deposits_df_chart.columns if 'total' in col.lower()]
            
            if date_col and total_col:
                deposits_clean = deposits_df_chart[[date_col[0], total_col[0]]].copy()
                deposits_clean.columns = ['Date', 'Deposits']
                # Clean currency formatting
                deposits_clean['Deposits'] = deposits_clean['Deposits'].astype(str).replace(r'[\$,]', '', regex=True)
                deposits_clean['Deposits'] = pd.to_numeric(deposits_clean['Deposits'], errors='coerce').fillna(0)
                deposits_clean['Date'] = pd.to_datetime(deposits_clean['Date'], errors='coerce')
                deposits_clean = deposits_clean.dropna(subset=['Date'])
                # Filter by date range
                deposits_clean = deposits_clean[
                    (deposits_clean['Date'].dt.date >= start_date) & 
                    (deposits_clean['Date'].dt.date <= end_date)
                ]
                
                # Process forecast
                forecast_clean = pd.DataFrame()
                if not forecast_df.empty:
                    forecast_date_col = [col for col in forecast_df.columns if 'date' in col.lower() and 'ref' not in col.lower()]
                    med_forecast_col = [col for col in forecast_df.columns if 'med' in col.lower() and 'forecast' in col.lower()]
                    
                    if forecast_date_col and med_forecast_col:
                        forecast_clean = forecast_df[[forecast_date_col[0], med_forecast_col[0]]].copy()
                        forecast_clean.columns = ['Date', 'Forecast']
                        forecast_clean['Forecast'] = forecast_clean['Forecast'].astype(str).replace(r'[\$,]', '', regex=True)
                        forecast_clean['Forecast'] = pd.to_numeric(forecast_clean['Forecast'], errors='coerce').fillna(0)
                        forecast_clean['Date'] = pd.to_datetime(forecast_clean['Date'], errors='coerce')
                        forecast_clean = forecast_clean.dropna(subset=['Date'])
                        # Filter by date range
                        forecast_clean = forecast_clean[
                            (forecast_clean['Date'].dt.date >= start_date) & 
                            (forecast_clean['Date'].dt.date <= end_date)
                        ]
                
                # Merge all three datasets
                merged = pd.merge(daily_sales, deposits_clean, on='Date', how='outer')
                if not forecast_clean.empty:
                    merged = pd.merge(merged, forecast_clean, on='Date', how='outer')
                
                merged = merged.fillna(0).sort_values('Date')
                
                # Melt for plotting
                value_cols = ['Sales', 'Deposits']
                if 'Forecast' in merged.columns:
                    value_cols.append('Forecast')
                melted = merged.melt(id_vars=['Date'], value_vars=value_cols, var_name='Type', value_name='Amount')
                
                fig_compare = px.line(melted, x='Date', y='Amount', color='Type', markers=True, 
                                      title="Daily Sales vs Deposits vs Forecast",
                                      color_discrete_map={'Sales': '#1f77b4', 'Deposits': '#2ca02c', 'Forecast': '#ff8c00'})
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.warning("Could not find 'Date' or 'Total' columns in deposits sheet.")
        else:
            st.info("No deposits data available or sales data missing Date column.")

    with tab4:
        st.subheader("Cash In Machines (CIM) by State")
        if not cim_df.empty:
            state_col = 'State' if 'State' in cim_df.columns else ([c for c in cim_df.columns if 'state' in c.lower()] + [None])[0]
            balance_col = 'Actual Balance' if 'Actual Balance' in cim_df.columns else ([c for c in cim_df.columns if 'actual' in c.lower() or 'balance' in c.lower()] + [None])[0]
            
            if state_col and balance_col:
                cim_df[balance_col] = cim_df[balance_col].astype(str).replace(r'[\$,#N/A]', '', regex=True)
                cim_df[balance_col] = pd.to_numeric(cim_df[balance_col], errors='coerce').fillna(0)
                
                state_summary = cim_df.groupby(state_col)[balance_col].sum().reset_index()
                state_summary.columns = ['State', 'Total Cash']
                state_summary = state_summary[state_summary['Total Cash'] > 0].sort_values('Total Cash', ascending=False)
                
                fig_cim = px.bar(state_summary, x='State', y='Total Cash', title="Cash Balance by State", color='Total Cash', color_continuous_scale='Greens')
                st.plotly_chart(fig_cim, use_container_width=True)
            else:
                st.warning(f"Could not find State or Balance columns.")
        else:
            st.info("Connect CIM sheet to see geographic breakdown.")

    with tab5:
        st.subheader("Raw Data (Transformed)")
        st.dataframe(filtered_df, use_container_width=True)

    st.divider()
    st.info("💡 **Persistence Active**: Data is automatically saved to and loaded from your Google Drive folder 'CashTeam Data'.")

else:
    st.info("👋 **Welcome!** The dashboard is initializing. If no data appears, please upload a Sales CSV to start the database.")
