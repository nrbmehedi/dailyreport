import pandas as pd

# Load your dataset (replace with your file path)
df = pd.read_excel("biddata.xlsx")   # or pd.read_csv("biddata.csv")

# Define mapping: original → new column names
rename_map = {
    "Booking ID": "booking_id",
    "Driver Fare": "bid_amount",
    "Car Type": "car_type",
    "Driver Phone No": "driver_phone_no",
    "Bid Time": "created_at",
    "Trip Type": "trip_type"   # Added
}

# Keep only mapped columns and rename
df = df.rename(columns=rename_map)[list(rename_map.values())]

# Save to CSV
df.to_csv("biddata.csv", index=False)

print("✅ File saved as biddata.csv")
