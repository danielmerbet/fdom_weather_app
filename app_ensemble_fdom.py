#globals().clear()
#from flask import Flask, render_template
#from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # Import for date formatting
import requests
from datetime import datetime
import pandas as pd
import os
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FuncFormatter
from joblib import load

#app = Flask(__name__)
#scheduler = BackgroundScheduler()

lat = 41.97
lon = 2.38
model = "gfs_seamless"#"bom_access_global_ensemble" #"gfs025" #"gfs_seamless" #"icon_seamless" # "gfs_seamless"
if model == "gfs_seamless":
  members = 31
if model == "gfs025":
  members = 31
if model == "icon_seamless":
  members = 40
if model == "bom_access_global_ensemble":
  members = 18

def fetch_data():
    # Step 1: Fetch data from the API
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,cloud_cover,surface_pressure,shortwave_radiation,et0_fao_evapotranspiration",
        "timezone": "Europe/Berlin", #auto
        "models": model
    }
    response = requests.get(url, params=params); response
    data = response.json()

    # Step 2: Extract timestamps
    timestamps = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
    
    # Step 3: Prepare data for all ensemble members
    ensemble_data = {"Time": timestamps}  # Start with the timestamps
    
    for variable in ["temperature_2m", "precipitation", "relative_humidity_2m",
                     "wind_speed_10m", "cloud_cover", "surface_pressure",
                     "shortwave_radiation", "et0_fao_evapotranspiration"]:
        for member in range(members):  # ensemble members
            member_key = f"{variable}_member{str(member).zfill(2)}"
            #member 0 is not called the same way
            if member == 0:
              member_key = f"{variable}"
              ensemble_data[f"{variable}_member{member}"] = data["hourly"][member_key]
            if member_key in data["hourly"]:  # Check if the member exists
                ensemble_data[f"{variable}_member{member}"] = data["hourly"][member_key]
    
    # Step 4: Convert to DataFrame in one go
    df = pd.DataFrame(ensemble_data)
    df.set_index("Time", inplace=True)

    # Filter to show only data from the current hour onwards
    #current_time = datetime.now()
    #df = df[df.index >= current_time]
    
    return df

def fetch_data_soil():
    variables = [
        "soil_temperature_0_to_10cm", "soil_temperature_10_to_40cm",
        "soil_temperature_40_to_100cm", "soil_temperature_100_to_200cm",
        "soil_moisture_0_to_10cm", "soil_moisture_10_to_40cm",
        "soil_moisture_40_to_100cm", "soil_moisture_100_to_200cm"
    ]
    
    ensemble_data = {"Time": None}  # Initialize with Time key to merge later
    
    for variable in variables:
        # Step 1: Fetch data for the current variable
        url = "https://ensemble-api.open-meteo.com/v1/ensemble"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": variable,
            "timezone": "Europe/Berlin",  # Use auto if necessary
            "models": model
        }
        
        response = requests.get(url, params=params)
        
        # Check if the response is successful
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data for {variable}: {response.status_code} {response.text}")
        
        data = response.json()
        
        # Extract timestamps only once
        if ensemble_data["Time"] is None:
            ensemble_data["Time"] = [datetime.fromisoformat(t) for t in data["hourly"]["time"]]
        
        # Add data for all ensemble members of the current variable
        for member in range(members):  # Ensemble members
            if member == 0:
                member_key = variable
            else:
                member_key = f"{variable}_member{str(member).zfill(2)}"
            
            if member_key in data["hourly"]:  # Check if the key exists
                ensemble_data[f"{variable}_member{member}"] = data["hourly"][member_key]
    
    # Step 2: Convert to DataFrame
    df = pd.DataFrame(ensemble_data)
    df.set_index("Time", inplace=True)
    
    return df

def arrange_predictions(df_soil):
    all_predictions = pd.DataFrame()
    #members = members  # Define the number of members if not passed as a parameter
    
    loaded_rf = load("data/random_forest_fdom.joblib")
    
    # Loop through each member (0 to 29)
    for i in range(members):
        # Fetch and select relevant data for the current member
        df_rf = df_soil[
            [
                f'soil_temperature_0_to_10cm_member{i}',
                f'soil_temperature_10_to_40cm_member{i}',
                f'soil_temperature_40_to_100cm_member{i}',
                f'soil_temperature_100_to_200cm_member{i}',
                f'soil_moisture_0_to_10cm_member{i}',
                f'soil_moisture_10_to_40cm_member{i}',
                f'soil_moisture_40_to_100cm_member{i}',
                f'soil_moisture_100_to_200cm_member{i}',
            ]
        ]
        
        # Rename columns for simplicity
        df_rf.columns = ['st7', 'st28', 'st100', 'st255', 'sm7', 'sm28', 'sm100', 'sm255']
        
        # Resample hourly data to daily and calculate the mean
        df_rf_daily = df_rf.resample('D').mean()
        
        # Apply the trained model
        predictions = loaded_rf.predict(df_rf_daily)
        
        # Store predictions in the DataFrame
        all_predictions[f'member{i}'] = predictions
    
    # Return the aggregated predictions DataFrame
    return all_predictions

def plot_data(df, predictions):
    label_sizes = 3
    cm = 1 / 2.54
    now = datetime.now()  # Get the current time
    fig, axs = plt.subplots(5, 1, figsize=(10 * cm, 10 * cm), sharex=True)  # Add an additional subplot for predictions
    fig.tight_layout(pad=0.5)
    #fig.subplots_adjust(hspace=0.1)

    # Define a function to format y-axis
    def format_y_axis(ax, step):
        ax.tick_params(axis='y', labelsize=label_sizes, color="gray", width=0.3)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.3)
        ax.spines['bottom'].set_color('gray')

    # Custom x-axis formatter function
    def custom_date_formatter(x, pos):
        date = mdates.num2date(x)
        if date.hour == 0 and date.minute == 0:  # Show only the date at midnight
            return date.strftime('%-d %b')  # Format as '5 Dec'
        elif date.hour == 12 and date.minute == 0:  # Show only the time at noon
            return date.strftime('%H:%M')
        else:
            return ""

    # Plot 1: Predictions
    unique_dates = df.index.normalize().unique()
    axs[0].plot(unique_dates, predictions, lw=0.5, color="red")
    axs[0].axvline(now, color="gray", linestyle="solid", lw=0.5)
    axs[0].set_ylabel("fDOM (QSU)", fontsize=label_sizes, labelpad=0)
    #axs[0].tick_params(axis='x', labelbottom=True, color="gray", width=0.3)
    axs[0].tick_params(axis='x', labelbottom=True, rotation=0, labelsize=label_sizes, color="gray", width=0.3)
    axs[0].xaxis.set_major_formatter(FuncFormatter(custom_date_formatter))
    axs[0].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))  # Major ticks at 00:00 and 12:00
    format_y_axis(axs[0], step=5)
    axs[0].grid(axis='y', linewidth=0.2, color='gray')
    axs[0].set_xlabel("Time (UTC +1)", fontsize=label_sizes, labelpad=1)

    # Plot 2: Temperature Ensemble Members
    for member in range(members):
        col_name = f"temperature_2m_member{member}"
        if col_name in df.columns:
            axs[1].plot(df.index, df[col_name], lw=0.5, alpha=0.7, color="black")
    axs[1].axvline(now, color="gray", linestyle="solid", lw=0.5)
    axs[1].set_ylabel("Temperature (°C)", fontsize=label_sizes, labelpad=0)
    axs[1].tick_params(axis='x', labelbottom=False, color="gray", width=0.3)
    format_y_axis(axs[1], step=5)
    axs[1].grid(axis='y', linewidth=0.2, color='gray')

    # Plot 3: Precipitation Ensemble Members
    for member in range(members):
        col_name = f"precipitation_member{member}"
        if col_name in df.columns:
            axs[2].plot(df.index, df[col_name], lw=0.5, alpha=0.7, color="blue")
    axs[2].axvline(now, color="gray", linestyle="solid", lw=0.5)
    axs[2].set_ylabel("Precipitation (mm)", fontsize=label_sizes, labelpad=0)
    axs[2].tick_params(axis='x', labelbottom=False, color="gray", width=0.3)
    format_y_axis(axs[2], step=5)
    axs[2].grid(axis='y', linewidth=0.2, color='gray')

    # Plot 4: Humidity Ensemble Members
    for member in range(members):
        col_name = f"relative_humidity_2m_member{member}"
        if col_name in df.columns:
            axs[3].plot(df.index, df[col_name], lw=0.5, alpha=0.7, color="purple")
    axs[3].axvline(now, color="gray", linestyle="solid", lw=0.5)
    axs[3].set_ylabel("Humidity (%)", fontsize=label_sizes, labelpad=0)
    axs[3].tick_params(axis='x', labelbottom=False, color="gray", width=0.3)
    format_y_axis(axs[3], step=20)
    axs[3].grid(axis='y', linewidth=0.2, color='gray')

    # Plot 5: Wind Speed Ensemble Members
    for member in range(members):
        col_name = f"wind_speed_10m_member{member}"
        if col_name in df.columns:
            axs[4].plot(df.index, df[col_name], lw=0.5, alpha=0.7, color="green")
    axs[4].axvline(now, color="gray", linestyle="solid", lw=0.5)
    axs[4].set_ylabel("Wind Speed (km/h)", fontsize=label_sizes, labelpad=0)
    axs[4].tick_params(axis='x', rotation=0, labelsize=label_sizes, color="gray", width=0.3)
    format_y_axis(axs[4], step=5)
    axs[4].grid(axis='y', linewidth=0.2, color='gray')

    # Apply custom x-axis formatter to the bottom subplot
    axs[4].xaxis.set_major_formatter(FuncFormatter(custom_date_formatter))
    axs[4].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))  # Major ticks at 00:00 and 12:00

    # Add the main x-axis label
    axs[4].set_xlabel("Time (UTC +1)", fontsize=label_sizes, labelpad=1)

    # Save the plot
    plt.savefig("static/plot.png", dpi=300)
    plt.close()
    
# Fetch and plot data, called every hour by the scheduler
def fetch_and_plot():
    df = fetch_data()
    df_soil = fetch_data_soil()
    df_soil_all = arrange_predictions(df_soil)
    plot_data(df, df_soil_all)

# Run the initial fetch and plot
fetch_and_plot()

# Schedule the fetch_and_plot function to run every hour
#scheduler.add_job(fetch_and_plot, "interval", hours=1)
#scheduler.start()

# Flask route to display the plot
#@app.route("/")
#def index():
    # Update plot before rendering
#    fetch_and_plot()
#    return render_template("index.html")

# Shutdown scheduler gracefully when app context is terminated
#@app.teardown_appcontext
#def shutdown_scheduler(exception=None):
#    try:
#        if scheduler.running:
#            scheduler.shutdown()
#    except SchedulerNotRunningError:
#        pass

#if __name__ == '__main__':
    #app.run(debug=True)
    #app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)), debug=True)
