import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """
    Clean and preprocess text data
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def load_and_clean_summaries(file_path):
    """
    Load movie summaries from the provided file and clean them
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()
    
    # Parse the data
    summaries = {}
    for line in data:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            movie_id, summary = parts
            summaries[movie_id] = clean_text(summary)
    
    return summaries

def extract_metadata(file_path):
    """
    Extract genre information from the metadata file
    """
    metadata = pd.read_csv(file_path, sep='\t', header=None)
    
    # Rename columns based on the README description
    metadata.columns = ['wikipedia_id', 'freebase_id', 'name', 'release_date', 
                        'box_office', 'runtime', 'languages', 'countries', 'genres']
    
    # Extract genres
    movie_genres = {}
    for _, row in metadata.iterrows():
        movie_id = str(row['wikipedia_id'])
        genres = row['genres'].split(',') if isinstance(row['genres'], str) else []
        
        # Clean genre names from the Freebase ID:name format
        clean_genres = []
        for genre in genres:
            if ':' in genre:
                # Extract just the name part after the Freebase ID
                clean_genres.append(genre.split(':')[-1])
        
        movie_genres[movie_id] = clean_genres
    
    return movie_genres

def create_dataset(summaries, genres, output_file):
    """
    Create a cleaned dataset with movie ID, summary, and genres
    """
    data = []
    
    for movie_id in summaries:
        if movie_id in genres:
            data.append({
                'movie_id': movie_id,
                'summary': summaries[movie_id],
                'genres': ','.join(genres[movie_id])
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"Dataset created and saved to {output_file}")
    return df

def train_test_split(df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets
    """
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    train_df.to_csv('train_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)
    
    print(f"Data split into train ({len(train_df)} samples) and test ({len(test_df)} samples) sets.")
    return train_df, test_df

if __name__ == "__main__":
    # Define file paths
    plot_summaries_path = "data\plot_summaries.txt"
    metadata_path = "movie.metadata.tsv"
    output_file = "cleaned_movie_data.csv"
    
    # Process data
    print("Loading and cleaning summaries...")
    summaries = load_and_clean_summaries(plot_summaries_path)
    
    print("Extracting metadata...")
    genres = extract_metadata(metadata_path)
    
    print("Creating dataset...")
    df = create_dataset(summaries, genres, output_file)
    
    print("Splitting into train and test sets...")
    train_df, test_df = train_test_split(df)
    
    print("Data preprocessing completed.")
