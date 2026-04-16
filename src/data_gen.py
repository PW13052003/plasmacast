# Import the necessary modules
import requests
import pandas as pd
import numpy as np
import holidays
import os

# Define 5 plasma centers with real coordinates
CENTERS = {
    "center_philly":  {"city": "Philadelphia", "lat": 39.9526, "lon": -75.1652, "base_donors": 60},
    "center_nyc":     {"city": "New York City", "lat": 40.7128, "lon": -74.0060, "base_donors": 312},
    "center_la":      {"city": "Los Angeles",   "lat": 34.0522, "lon": -118.2437, "base_donors": 146},
    "center_chicago": {"city": "Chicago",       "lat": 41.8781, "lon": -87.6298, "base_donors": 101},
    "center_houston": {"city": "Houston",       "lat": 29.7604, "lon": -95.3698, "base_donors": 86},
}

START_DATE = "2022-01-01"
END_DATE   = "2023-12-31"

def fetch_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,precipitation_sum"
    }
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temp_max": data["daily"]["temperature_2m_max"],
        "precipitation": data["daily"]["precipitation_sum"]
    })
    df["date"] = pd.to_datetime(df["date"])
    return df

def calculate_donor_count(base_donors, temp, precipitation, day_of_week, is_holiday):
    count = base_donors

    # Day of week effect
    day_multipliers = {
        0: 1.1,   # Monday
        1: 1.15,  # Tuesday
        2: 1.15,  # Wednesday
        3: 1.1,   # Thursday
        4: 0.95,  # Friday
        5: 0.75,  # Saturday
        6: 0.60,  # Sunday
    }
    count *= day_multipliers[day_of_week]

    # Weather effects
    if temp < 0:
        count *= 0.70       # very cold, people stay home
    elif temp < 10:
        count *= 0.85       # chilly
    elif temp > 35:
        count *= 0.80       # extreme heat

    if precipitation > 20:
        count *= 0.75       # heavy rain
    elif precipitation > 5:
        count *= 0.90       # light rain

    # Holiday effect
    if is_holiday:
        count *= 0.50

    # Add realistic randomness (+/- 10%)
    noise = np.random.uniform(0.90, 1.10)
    count *= noise

    return int(count)

def generate_dataset():
    us_holidays = holidays.US()
    all_data = []

    for center_id, info in CENTERS.items():
        print(f"Fetching weather for {info['city']}...")
        weather_df = fetch_weather(info["lat"], info["lon"], START_DATE, END_DATE)

        for _, row in weather_df.iterrows():
            date = row["date"]
            temp = row["temp_max"]
            precipitation = row["precipitation"]
            day_of_week = date.dayofweek
            is_holiday = date in us_holidays

            donor_count = calculate_donor_count(
                base_donors=info["base_donors"],
                temp=temp,
                precipitation=precipitation,
                day_of_week=day_of_week,
                is_holiday=is_holiday
            )

            all_data.append({
                "date": date,
                "center_id": center_id,
                "city": info["city"],
                "donor_count": donor_count,
                "temp_max": temp,
                "precipitation": precipitation,
                "day_of_week": day_of_week,
                "is_holiday": int(is_holiday)
            })

    df = pd.DataFrame(all_data)
    df = df.sort_values(["center_id", "date"]).reset_index(drop=True)
    os.makedirs("src/data", exist_ok=True)
    df.to_csv("src/data/donor_data.csv", index=False)
    print(f"Dataset saved! {len(df)} rows generated.")
    return df


if __name__ == "__main__":
    generate_dataset()