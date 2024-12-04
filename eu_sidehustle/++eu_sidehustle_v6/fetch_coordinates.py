import fetch_training_data_methods as methods
import pandas as pd

# List of cities to fetch data for
cities = [
    "New York", "London", "Tokyo", "Paris", "Beijing", "Shanghai", "Los Angeles",
    "Moscow", "Dubai", "Singapore", "Hong Kong", "Berlin", "Madrid", "Rome",
    "Sydney", "Toronto", "Bangkok", "Seoul", "Mumbai", "Istanbul", "Mexico City",
    "São Paulo", "Buenos Aires", "Cairo", "Jakarta", "Lagos", "Delhi", "Kuala Lumpur",
    "Melbourne", "Rio de Janeiro", "Chicago", "San Francisco", "Houston", "Miami",
    "Boston", "Washington, D.C.", "Amsterdam", "Zurich", "Geneva", "Frankfurt",
    "Barcelona", "Munich", "Dubai", "Cape Town", "Johannesburg", "Abu Dhabi",
    "Doha", "Osaka", "Nagoya", "Brussels", "Lisbon", "Stockholm", "Vienna",
    "Helsinki", "Warsaw", "Dublin", "Copenhagen", "Oslo", "Athens", "Budapest",
    "Prague", "Edinburgh", "Manchester", "Glasgow", "Birmingham", "Cardiff",
    "Mumbai", "Chennai", "Bangalore", "Hyderabad", "Ahmedabad", "Pune", "Kolkata",
    "Karachi", "Lahore", "Islamabad", "Manila", "Hanoi", "Ho Chi Minh City",
    "Riyadh", "Jeddah", "Tel Aviv", "Jerusalem", "Beirut", "Tehran", "Baghdad",
    "Kuwait City", "Casablanca", "Rabat", "Nairobi", "Accra", "Addis Ababa",
    "Kinshasa", "Algiers", "Tunis", "Santiago", "Bogotá", "Lima", "Caracas",
    "Tunis"
]


# Fetching lat and lon for each city and creating the DataFrame
data = {"city": [], "lat": [], "lon": []}

for city in cities:
    lat, lon = methods.fetch_lat_lon(city_name=city)
    data["city"].append(city)
    data["lat"].append(lat)
    data["lon"].append(lon)

# Create DataFrame with an index column
city_coordinates = pd.DataFrame(data)
city_coordinates.index.name = "index"

# Write DataFrame to CSV
city_coordinates.to_csv("city_coordinates.csv")

print("City coordinates have been saved to 'city_coordinates.csv'.")
