import csv

def get_unique_values_from_csv_column(file_path, column_name):
    """
    Reads a CSV file, finds a specified column, and prints its unique values.

    Args:
        file_path (str): The path to the CSV file.
        column_name (str): The name of the column to parse.
    """
    try:
        # Open the CSV file for reading
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            # Create a CSV reader that uses a semicolon delimiter
            reader = csv.reader(csvfile, delimiter=';')

            # Read the header to find the index of the desired column
            header = next(reader)
            try:
                # Get the index of the column with the specified name
                col_index = header.index(column_name)
            except ValueError:
                print(f"Error: Column '{column_name}' not found in the CSV file.")
                return

            # Use a set to store unique values, which is very efficient
            unique_values = set()
            for row in reader:
                if len(row) > col_index:
                    # Add the value from the specified column to the set
                    unique_values.add(row[col_index])

            # Print the sorted unique values
            print(f"Unique values in the '{column_name}' column:")
            for value in sorted(list(unique_values)):
                print(value)

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except StopIteration:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

file_to_parse = '/Users/abiralshakya/Documents/Research/GraphVectorTopological/list_magnetic_compounds_20250602.csv'

column_to_parse = 'MSG'

get_unique_values_from_csv_column(file_to_parse, column_to_parse)