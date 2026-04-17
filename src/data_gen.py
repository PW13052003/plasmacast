# =====================================================================================
# Project:      PlasmaCast — Plasma Donor Demand Forecasting
# Contributor:  Puranjay Wadhera (GitHub: @PW13052003)
# File:         data_gen.py
# Purpose:      Generates a synthetic donor dataset for 10 US plasma centers.
#               Real historical weather data is fetched from the Open-Meteo API
#               and combined with US federal holiday data and population-scaled
#               donor baselines to produce realistic daily donor count estimates.
# Language:     Python 3.12
# Libraries:    requests, pandas, numpy, holidays, os
# APIs:         Open-Meteo Historical Weather API (https://archive-api.open-meteo.com)
# GitHub:       https://github.com/PW13052003/plasmacast/blob/main/src/data_gen.py
# =====================================================================================


# Import the necessary modules
import requests
import pandas as pd
import numpy as np
import holidays
import os


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# The 10 largest US cities by population, sourced from the US Census Bureau.
# Vintage 2024 estimates (census.gov). Coordinates are sourced from the same dataset.
# base_donors represents the average daily donor count, scaled proportionally to each city's population.
# Philadelphia is the baseline reference city with 60 base donors per day.
CENTERS = {
    "center_nyc":          {"city": "New York City", "lat": 40.66, "lon": -73.94, "base_donors": 318},
    "center_la":           {"city": "Los Angeles",   "lat": 34.02, "lon": -118.41, "base_donors": 145},
    "center_chicago":      {"city": "Chicago",       "lat": 41.84, "lon": -87.68,  "base_donors": 102},
    "center_houston":      {"city": "Houston",       "lat": 29.79, "lon": -95.39,  "base_donors": 90},
    "center_phoenix":      {"city": "Phoenix",       "lat": 33.57, "lon": -112.09, "base_donors": 63},
    "center_philly":       {"city": "Philadelphia",  "lat": 40.01, "lon": -75.13,  "base_donors": 60},
    "center_san_antonio":  {"city": "San Antonio",   "lat": 29.46, "lon": -98.52,  "base_donors": 57},
    "center_san_diego":    {"city": "San Diego",     "lat": 32.81, "lon": -117.14, "base_donors": 53},
    "center_dallas":       {"city": "Dallas",        "lat": 32.79, "lon": -96.77,  "base_donors": 50},
    "center_jacksonville": {"city": "Jacksonville",  "lat": 30.34, "lon": -81.66,  "base_donors": 38},
}


# 4 years of data (2020-2023) give us a ~75/25 train/test split.
# Data from 2020-2022 constitutes the training dataset and data from 2023 constitutes the testing dataset.
START_DATE = "2020-01-01"
END_DATE   = "2023-12-31"


def fetch_weather(lat, lon, start_date, end_date):
    """
        Fetches real historical daily weather data from the Open-Meteo archive API.

        Args:
            lat (float): Latitude of the plasma center city.
            lon (float): Longitude of the plasma center city.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.

        Returns:
            pd.DataFrame: Daily weather data with columns:
                          date, temp_max (°C), precipitation (mm).
    """
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
    """
        Calculates a synthetic daily donor count by applying real-world behavioral multipliers to a city's population-scaled baseline donor count.

        Args:
            base_donors (int): Population-scaled average donor count for the city.
            temp (float): Max daily temperature in Celsius.
            precipitation (float): Total daily precipitation in mm.
            day_of_week (int): Day of week as integer (0=Monday, 6=Sunday).
            is_holiday (bool): Whether the date is a US federal holiday.

        Returns:
            int: Simulated donor count for that day.
    """
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
    """
        Orchestrates the full data generation pipeline:-
            1. Loops through all 10 plasma centers
            2. Fetches real weather data for each city
            3. Calculates synthetic donor counts for every day
            4. Assembles and saves the final dataset as a CSV

        Args:
            NONE

        Returns:
            pd.DataFrame: The complete donor dataset.
    """
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
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "donor_data.csv"), index=False)
    print(f"Dataset saved! {len(df)} rows generated.")
    return df


if __name__ == "__main__":
    generate_dataset()