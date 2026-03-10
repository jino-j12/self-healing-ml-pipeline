import pandas as pd
import time

def stream_transactions(file_path, batch_size=500):

    data = pd.read_csv(file_path)

    total_rows = len(data)

    for start in range(0, total_rows, batch_size):

        end = start + batch_size
        batch = data.iloc[start:end]

        print(f"\nStreaming transactions {start} → {end}")

        yield batch

        time.sleep(2)   # simulate real-time delay