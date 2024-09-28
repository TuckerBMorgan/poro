import numpy as np

def read_file(filepath):
    """Reads a file and returns its lines as a list"""
    with open(filepath, 'r') as file:
        return file.readlines()

def calculate_percent_error(value1, value2):
    """Calculates the percent error between two numbers"""
    if value1 == value2 == 0:
        return 0.0
    # handle masking of -inf values
    if value1 == float('-inf') or value2 == float('-inf'):
        return 0.0
    try:
        return abs((value1 - value2) / value1) * 100
    except ZeroDivisionError:
        return np.inf

def process_files(rust_file, python_file):
    """Processes two files, calculates percentage error per zone"""
    rust_lines = read_file(rust_file)
    python_lines = read_file(python_file)

    if len(rust_lines) != len(python_lines):
        print("Error: Files have different numbers of lines.")
        return

    current_zone = None
    zone_errors = {}
    
    for rust_line, python_line in zip(rust_lines, python_lines):
        rust_line = rust_line.strip()
        python_line = python_line.strip()

        # Track zone if line starts with $
        if rust_line.startswith('$'):
            current_zone = rust_line
            zone_errors[current_zone] = []
            continue
        
        if python_line.startswith('$'):
            continue  # Ignore python zone markers, just rely on rust_file

        try:
            rust_value = float(rust_line)
            python_value = float(python_line)
        except ValueError:
            # Skip lines that are not numerical
            continue

        # Calculate percent error
        percent_error = calculate_percent_error(rust_value, python_value)
        zone_errors[current_zone].append(percent_error)

    # Calculate total percent error for each zone
    for zone, errors in zone_errors.items():
        if errors:
            avg_error = sum(errors) / len(errors)
        else:
            avg_error = 0
        print(f"Zone: {zone}, Total % Error: {avg_error:.2f}%")

if __name__ == "__main__":
    rust_file = "rust_checkfile.txt"  # Replace with your rust file path
    python_file = "python_checkfile.txt"  # Replace with your python file path

    process_files(rust_file, python_file)
