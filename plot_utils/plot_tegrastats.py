import argparse
import csv
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse and plot tegrastats power usage data.")
    parser.add_argument("--input_file", default="tegrastats.log", help="Tegrastats output file.")
    parser.add_argument("--image_file", default="power_graph.pdf", help="Output file for the plot.")
    parser.add_argument("--csv", action="store_true", help="Save extracted values to CSV.")
    parser.add_argument("--remove_original", action="store_true", help="Remove original tegrastats log files after saving to CSV.")
    parser.add_argument("--plot_title", default="Jetson Nano Power Usage", help="Title of the plot.")
    return parser.parse_args()

def read_and_process_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            timestamp_match = re.search(r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
            
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(1), "%m-%d-%Y %H:%M:%S")
                vdd_matches = re.findall(r"(VDD_\w+ \d+mW)", line)
                vdd_data = {}
                for vdd in vdd_matches:
                    key, value = vdd.split()
                    vdd_data[key] = int(value[:-2])
                data.append((timestamp, vdd_data))
                print(f"Timestamp: {timestamp}, VDD Values: {vdd_data}")
    return data

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Summed VDD (mW)'])
        writer.writerows(data)
    print(f"Data saved to {filename}")

def plot_data(data, output_file, title):
    # timestamps = [entry[0] for entry in data]
    
    first_timestamp = data[0][0]
    relative_seconds = [(entry[0] - first_timestamp).total_seconds() for entry in data]

    vdd_keys = set()
    for _, vdds in data:
        vdd_keys.update(vdds.keys())
    
    sum_vdd = [0] * len(data)

    plt.figure(figsize=(14, 7))

    for key in vdd_keys:
        vdd_values = [vdds.get(key, 0) for _, vdds in data]
        plt.plot(relative_seconds, vdd_values, label=key)
        sum_vdd = [sum_vdd[i] + vdd_values[i] for i in range(len(sum_vdd))]

    
    # Plotting the sum of all VDD values
    plt.plot(relative_seconds, sum_vdd, label='Sum of VDDs', color='k', linestyle='--')

    plt.xlabel('Timestamp')
    plt.ylabel('Power (mW)')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend(loc='upper right') 
    plt.tight_layout()
    
    # Save output if specified
    if output_file is not None:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")


def main():
    args = parse_arguments()
    data = read_and_process_file(args.input_file)
    plot_data(data, args.image_file, args.plot_title)
    
    if args.csv:
        csv_filename = args.input_file.replace('.log', '.csv')
        save_to_csv(data, csv_filename)
        if args.remove_original:
            os.remove(args.input_file)
            print(f"Original file {args.input_file} removed.")

if __name__ == "__main__":
    main()
