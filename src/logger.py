from datetime import datetime

def log_event(message):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_message = f"[{timestamp}] {message}"

    print(log_message)

    with open("logs/pipeline_log.txt", "a") as f:
        f.write(log_message + "\n")