def create_synthetic_data(filename="synthetic_data.csv", num_bars=1000):
    """
    Create synthetic OHLC data for testing purposes
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    print(f"Creating synthetic data file: {filename}")
    
    # Start date
    start_date = datetime.now() - timedelta(days=num_bars//1440)
    
    # Generate synthetic price series (random walk)
    price = 15000.0  # Starting price
    volatility = 25.0  # Daily volatility
    
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    
    for i in range(num_bars):
        # Generate timestamp
        bar_time = start_date + timedelta(minutes=i)
        dates.append(bar_time)
        
        # Generate OHLC data
        daily_vol = volatility * np.sqrt(1/1440)  # Scale to per-minute
        
        # Random walk for close price
        price_change = np.random.normal(0, daily_vol)
        new_price = price * (1 + price_change/100)
        
        # Generate open, high, low based on close
        open_price = price
        high_price = max(open_price, new_price) * (1 + abs(np.random.normal(0, daily_vol/2)/100))
        low_price = min(open_price, new_price) * (1 - abs(np.random.normal(0, daily_vol/2)/100))
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(new_price)
        
        # Update price for next iteration
        price = new_price
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Created {num_bars} bars of synthetic data")
    return filename

# Add this to the if __name__ block before the argparse section:
if __name__ == "__main__":
    # Check if synthetic data should be created
    import os
    if not os.path.exists("synthetic_data.csv"):
        create_synthetic_data()
    
    # Rest of the code (argparse section from above)