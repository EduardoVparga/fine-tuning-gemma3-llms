import pandas as pd
import json
from pathlib import Path

# Parámetros configurables
SEED = 310       # Semilla para la mezcla aleatoria
N_TRAIN = 5000  # Número de registros que irán al archivo train.json

# Define paths using pathlib for better portability and robustness
RAW_DATA_PATH = Path("./raw_data")
DATA_PATH = Path("./data")

def create_dataset(df: pd.DataFrame) -> list:
    """
    Transforms the DataFrame into a list of dictionaries in the required format.
    Each element contains a key 'messages' that groups messages with roles:
    - system: the system message.
    - user: the user message.
    - assistant: the expected assistant response.
    
    Uses itertuples for better performance compared to iterrows.
    """
    return [
        {
            "messages": [
                {"role": "system", "content": row.sys_msg},
                {"role": "user", "content": row.text},
                {"role": "assistant", "content": f"<label>{row.target}</label>"},
            ]
        }
        for row in df.itertuples(index=False)
    ]

def main():
    """
    Main function that:
    - Reads the input CSV file.
    - Renames and adds necessary columns.
    - Shuffles the dataset with a fixed random seed.
    - Splits into train and test sets based on N_TRAIN.
    - Saves each subset in JSON files (train.json and test.json).
    """
    # Define the system message to be included in each record
    system_message = (
        "You are a sentiment analysis system. Classify input text into one of the following categories:\n\n"
        "- Negative: Expresses dissatisfaction, frustration, or negative emotions.\n"
        "- Neutral: Lacks strong emotions or is ambiguous.\n"
        "- Positive: Expresses satisfaction, enthusiasm, or approval.\n\n"
        "Analyze the overall sentiment based on tone, keywords, and context. Return the result in the following format:\n"
        "<label>[Negative|Neutral|Positive]</label>"
    )

    # Build the full path to the CSV file and read the selected data
    csv_file = RAW_DATA_PATH / "Tweets.csv"
    try:
        df = pd.read_csv(csv_file, usecols=["airline_sentiment", "text"])
    except Exception as e:
        print(f"Error reading the file {csv_file}: {e}")
        return

    # Rename column for clarity
    df.rename(columns={"airline_sentiment": "target"}, inplace=True)
    
    # Add the system message column to all records
    df["sys_msg"] = system_message

    # Mezclar aleatoriamente el DataFrame con la semilla definida
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Dividir el DataFrame en train y test
    train_df = df.iloc[:N_TRAIN].copy()
    test_df = df.iloc[N_TRAIN:].copy()

    # Crear los datasets en el formato deseado
    train_dataset = create_dataset(train_df)
    test_dataset = create_dataset(test_df)

    # Construir las rutas de salida
    train_file = DATA_PATH / "train.json"
    test_file = DATA_PATH / "test.json"

    # Guardar train.json
    try:
        with open(train_file, "w", encoding="utf-8") as file:
            json.dump(train_dataset, file, ensure_ascii=False, indent=4)
        print(f"Train dataset saved successfully in {train_file}")
    except Exception as e:
        print(f"Error saving the file {train_file}: {e}")

    # Guardar test.json
    try:
        with open(test_file, "w", encoding="utf-8") as file:
            json.dump(test_dataset, file, ensure_ascii=False, indent=4)
        print(f"Test dataset saved successfully in {test_file}")
    except Exception as e:
        print(f"Error saving the file {test_file}: {e}")

if __name__ == "__main__":
    main()
