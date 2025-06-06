import csv

def get_unique_values_from_csv_column(file_path, column_name):
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            header = next(reader)
            try:
                col_index = header.index(column_name)
            except ValueError:
                print(f"Error: Column '{column_name}' not found in the CSV file.")
                return
            unique_values = set()
            for row in reader:
                if len(row) > col_index:
                    unique_values.add(row[col_index])

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