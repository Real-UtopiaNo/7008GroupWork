import pandas as pd

def extract_qa_pairs(input_file, output_file):
    # Read the contents of the input text file
    with open(input_file, 'r',encoding='utf-8') as file:
        content = file.readlines()

    # Extract Q and A pairs, ensuring to remove duplicates based on the question (Q)
    qa_pairs = {}
    for i in range(len(content)):
        line = content[i].strip()
        if line.startswith('Q:'):
            question = line[2:].strip()  # Remove 'Q:' prefix
            # The corresponding answer is the next line
            if i + 1 < len(content) and content[i + 1].startswith('A:'):
                answer = content[i + 1][2:].strip()  # Remove 'A:' prefix
                qa_pairs[question] = answer  # Store unique questions with their answers

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(qa_pairs.items()), columns=['Question', 'Answer'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

# Example usage
input_file_path = 'edtech.txt'  # Path to your input text file
output_file_path = 'Edtech_QA_pairs.csv'  # Path to save the output CSV file
extract_qa_pairs(input_file_path, output_file_path)
input_file_path = 'artech.txt'  # Path to your input text file
output_file_path = 'Artech_QA_pairs.csv'  # Path to save the output CSV file
extract_qa_pairs(input_file_path, output_file_path)