#!/usr/bin/env python3
# Script to generate synthetic financial data with OHLC and technical indicators

import csv
import datetime
import random
import math

# Output file path
output_file = "MNQ_03-25_Data_03-2025_1806.csv"

# Initialize the starting values
start_time = datetime.datetime(2025, 2, 16, 21, 31, 0)
start_open = 22261.75
start_high = 22263.75
start_low = 22261.00
start_close = 22261.00
start_adx = 50
start_volume = 169
start_ema = 22261
start_atr = 2.75

# Parameters for price movement simulation
volatility = 0.0004  # Controls the amount of random price movement
trend = 0.0002       # Positive for uptrend, negative for downtrend
mean_reversion = 0.3  # Strength of mean reversion to control price range

# Function to generate realistic price movements
def generate_prices(prev_open, prev_high, prev_low, prev_close):
    # Random walk with drift and mean reversion
    random_factor = random.gauss(0, 1)
    price_change = trend + volatility * random_factor - mean_reversion * (prev_close - start_close) / 100
    
    # Generate new prices
    new_open = round(prev_close + random.uniform(-0.5, 1.0), 2)
    new_close = round(new_open + price_change * 10, 2)
    
    # Ensure High and Low are consistent with Open and Close
    max_price = max(new_open, new_close)
    min_price = min(new_open, new_close)
    
    # High should be above both Open and Close
    new_high = round(max_price + random.uniform(0.25, 2.0), 2)
    
    # Low should be below both Open and Close
    new_low = round(min_price - random.uniform(0, 0.75), 2)
    
    return new_open, new_high, new_low, new_close

# Generate data for 6500 rows
data = []
header = ["Timestamp", "Open", "High", "Low", "Close", "ADX_ADX", "VOL_Volume", "EMA_EMA", "EMA_EMA", "ATR_ATR"]
data.append(header)

# First row with initial values
first_row = [
    start_time.strftime("%Y-%m-%d %H:%M:%S"),
    start_open,
    start_high,
    start_low,
    start_close,
    start_adx,
    start_volume,
    start_ema,
    start_ema,
    start_atr
]
data.append(first_row)

# Generate the remaining rows
current_time = start_time
current_open = start_open
current_high = start_high
current_low = start_low
current_close = start_close
current_adx = start_adx
current_volume = start_volume
current_ema = start_ema
current_atr = start_atr

for i in range(1, 6500):
    # Increment time by 1 minute
    current_time = current_time + datetime.timedelta(minutes=1)
    
    # Generate new OHLC values
    current_open, current_high, current_low, current_close = generate_prices(
        current_open, current_high, current_low, current_close
    )
    
    # Update technical indicators
    current_adx = min(100, current_adx + random.uniform(-0.5, 1))
    current_volume = current_volume + random.randint(1, 10)
    
    # EMA typically follows close price with some smoothing
    current_ema = round(current_close - random.uniform(0, 5))
    
    # ATR typically increases with volatility
    current_atr = round(current_atr + random.uniform(-0.05, 0.1), 2)
    current_atr = max(1.0, current_atr)  # Ensure ATR doesn't go too low
    
    # Create row and add to data
    row = [
        current_time.strftime("%Y-%m-%d %H:%M:%S"),
        current_open,
        current_high,
        current_low,
        current_close,
        round(current_adx),
        round(current_volume),
        current_ema,
        current_ema,
        current_atr
    ]
    data.append(row)

# Write data to CSV file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

print(f"Successfully generated {len(data)-1} rows of synthetic financial data in {output_file}")