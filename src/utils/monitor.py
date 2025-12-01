import time
import os
import sys
from datetime import datetime

LOG_FILE = "logs/audio_downloader.log"
TEMP_DIR = "data/audio/temp"
PROCESSED_DIR = "data/audio/processed"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_last_log_line():
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            if lines:
                return lines[-1].strip()
    except FileNotFoundError:
        return "Log file not found."
    return "Waiting for logs..."

def count_files(directory):
    if not os.path.exists(directory):
        return 0
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def get_temp_file_size():
    if not os.path.exists(TEMP_DIR):
        return "No temp dir"

    files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if os.path.isfile(os.path.join(TEMP_DIR, f))]
    if not files:
        return "No active download"

    # Get the largest file (likely the one being downloaded)
    largest_file = max(files, key=os.path.getsize)
    size_mb = os.path.getsize(largest_file) / (1024 * 1024)
    return f"{os.path.basename(largest_file)}: {size_mb:.2f} MB"

def main():
    print("Starting monitor... Press Ctrl+C to exit.")
    time.sleep(1)

    start_processed_count = count_files(PROCESSED_DIR)
    start_time = time.time()

    try:
        while True:
            clear_screen()
            current_time = datetime.now().strftime("%H:%M:%S")
            elapsed_time = time.time() - start_time

            current_processed_count = count_files(PROCESSED_DIR)
            new_songs = current_processed_count - start_processed_count

            print(f"=== Audio Pipeline Monitor [{current_time}] ===")
            print(f"Runtime: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
            print(f"Total Processed Songs: {current_processed_count}")
            print(f"Songs Downloaded this Session: {new_songs}")
            print("-" * 40)
            print(f"Last Log Activity:")
            print(f"  {get_last_log_line()}")
            print("-" * 40)
            print(f"Current Download Status:")
            print(f"  {get_temp_file_size()}")
            print("-" * 40)
            print("Press Ctrl+C to stop monitoring (script will continue running)")

            time.sleep(2)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")

if __name__ == "__main__":
    main()
