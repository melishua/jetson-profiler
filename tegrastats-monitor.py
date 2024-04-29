"""
Goal: Run tegrastats on Jetson

Example:
    python3 tegrastats-monitor.py --start_signal "" --end_signal "" --logfile ""
"""
import os
import time
import subprocess
from datetime import datetime
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor signals to start and stop tegrastats.")
    parser.add_argument("--start_signal", default="START_SIGNAL", help="File name for the start signal.")
    parser.add_argument("--end_signal", default="END_SIGNAL", help="File name for the end signal.")
    parser.add_argument("--logfile", default="tegrastats", help="Base file name for the logfile.")
    return parser.parse_args()

def run_helper_script(command, *args):
    """Run the helper script with the given command and args."""
    cmd = ['./util/check_file_in_docker_container.sh', command] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()

def get_first_container_id():
    """Get the first running container ID."""
    return run_helper_script('get_container_id')

def check_file_in_container(container_id, file_path):
    """Check if a file exists in the specified Docker container."""
    result = run_helper_script('check_file', container_id, file_path)
    return result == ""

def get_logfile_name(base_name):
    """Generate a logfile name with a timestamp if the logfile already exists."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.log" if os.path.exists(f"{base_name}.log") else f"{base_name}.log"

def run_tegrastats(logfile):
    """Start the tegrastats daemon process."""
    subprocess.run(['tegrastats', '--logfile', logfile, '--start'])

def stop_tegrastats():
    """Stop the tegrastats daemon process."""
    subprocess.run(['tegrastats', '--stop'])

def cleanup_files(*files):
    """Remove specified files."""
    for file in files:
        try:
            os.remove(file)
            print(f"Removed {file}")
        except FileNotFoundError:
            print(f"{file} not found for removal.")

def main():
    args = parse_arguments()
    logfile = args.logfile
    start_signal = args.start_signal
    end_signal = args.end_signal
    
    container_id = get_first_container_id()
    print(f"(first running container ID): {container_id}")
    if container_id is None:
        print("No running containers found.")
        print("Start jetson-container first.")
        print("Bye!")
        return
    
    # Loop until the start signal file is detected
    counter = 600
    while True:
        if check_file_in_container(container_id, start_signal):
            print("Start signal detected. Starting tegrastats...")
            logfile = get_logfile_name(logfile)
            run_tegrastats(logfile)
            break
        else:
            print("Waiting for start signal...")
            time.sleep(1)  # Check every 1 seconds
            counter -= 1
        
        if counter == 0:
            print("Start signal not detected after 10min...")
            print("Please start the experiment!")
            print("Exiting for now...")
            print("Bye!")
            return

    # Loop to monitor the end signal
    try:
        print_counter = 0
        while True:
            if check_file_in_container(container_id, end_signal):
                print("End signal detected. Stopping tegrastats...")
                stop_tegrastats()
                # cleanup_files(start_signal, end_signal)
                break
            else:
                if print_counter % 12 == 0:
                    print_counter = 0
                    print("Monitoring for end signal...")
                time.sleep(5)  # Check every 5 seconds
                print_counter += 1
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected.")
        print("Stopping tegrastats...")
        stop_tegrastats()
        print("Exiting for now...")
        print("Bye!")
    except Exception as e:
        print(f"Exception: {e}")
        print("Stopping tegrastats...")
        stop_tegrastats()
        print("Exiting for now...")
        print("Bye!")
if __name__ == "__main__":
    main()
