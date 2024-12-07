import pandas as pd

# Load the uploaded CSV files (assuming they are in the same directory as the script)
file1_path = 'aitools.csv'
file2_path = 'futuretools.csv'

# Read the CSV files into dataframes
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Concatenate the dataframes
df_combined = pd.concat([df1, df2], ignore_index=True)

# Create the new columns based on the required format
df_combined['Question'] = "what is " + df_combined['tool']
df_combined['Answer'] = df_combined['tool'] + ": " + df_combined['tool_description'] + "; \nYou can find more information about this tool at " + df_combined['tool_mage_url']

# Create the new CSV with only the 'Question' and 'Answer' columns
output_df = df_combined[['Question', 'Answer']]

# Save the result to a new CSV file in the same directory
output_csv_path = 'consolidated_qa_pairs.csv'
output_df.to_csv(output_csv_path, index=False)

print(f"CSV file has been saved to {output_csv_path}")
