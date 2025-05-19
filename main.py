import os
import sys
import pandas as pd
from data_preprocessing import clean_text
from translation_audio import TranslationAudioService
from genre_prediction import MovieGenrePredictor
from utils import play_audio, display_data_stats, check_file_paths

class MovieAnalysisSystem:
    def __init__(self):
        """Initialize the movie analysis system"""
        self.translation_service = TranslationAudioService()
        self.genre_predictor = MovieGenrePredictor()
        
        # Try to load the trained model if it exists
        try:
            if os.path.exists('models/model.pkl'):
                print("Loading pre-trained genre prediction model...")
                self.genre_predictor.load_model()
                self.model_loaded = True
            else:
                print("No pre-trained model found. You'll need to train the model first.")
                self.model_loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_menu(self):
        """Display the main menu"""
        
        print("=" * 50)
        print("MOVIE SUMMARY ANALYSIS SYSTEM")
        print("=" * 50)
        print("1. Process and Clean Data")
        print("2. Train Genre Prediction Model")
        print("3. Translate and Generate Audio for Sample Movies")
        print("4. Enter a Movie Summary for Analysis")
        print("5. Display Dataset Statistics")
        print("6. Exit")
        print("=" * 50)
    
    def process_data(self):
        """Process and clean the movie data"""
        self.clear_screen()
        print("Processing and Cleaning Data...")
        
        paths = check_file_paths()
        
        # Import here to avoid circular imports
        from data_preprocessing import (
            load_and_clean_summaries,
            extract_metadata,
            create_dataset,
            train_test_split
        )
        
        print(f"\nLoading data from:\n- {paths['plot_summaries']}\n- {paths['metadata']}")
        
        try:
            # Process summaries
            print("Loading and cleaning summaries...")
            summaries = load_and_clean_summaries(paths['plot_summaries'])
            print(f"Processed {len(summaries)} summaries.")
            
            # Extract metadata
            print("Extracting metadata...")
            genres = extract_metadata(paths['metadata'])
            print(f"Extracted genres for {len(genres)} movies.")
            
            # Create dataset
            print("Creating dataset...")
            output_file = "cleaned_movie_data.csv"
            df = create_dataset(summaries, genres, output_file)
            print(f"Created dataset with {len(df)} movies.")
            
            # Split data
            print("Splitting data into train and test sets...")
            train_df, test_df = train_test_split(df)
            
            print("\nData processing completed successfully!")
            
        except Exception as e:
            print(f"Error processing data: {e}")
        
        input("\nPress Enter to continue...")
    
    def train_model(self):
        """Train the genre prediction model"""
        self.clear_screen()
        print("Training Genre Prediction Model...")
        
        if not os.path.exists('train_data.csv') or not os.path.exists('test_data.csv'):
            print("Error: Training and test data files not found.")
            print("Please process the data first (Option 1).")
            input("\nPress Enter to continue...")
            return
        
        try:
            # Load data
            train_df, test_df = self.genre_predictor.load_data('train_data.csv', 'test_data.csv')
            
            # Preprocess data
            X_train, y_train, X_test, y_test = self.genre_predictor.preprocess_data(train_df, test_df)
            
            # Train model
            self.genre_predictor.train_model(X_train, y_train)
            self.model_loaded = True
            
            # Evaluate model
            print("\nEvaluating model performance...")
            metrics = self.genre_predictor.evaluate_model(X_test, y_test)
            
            # Plot confusion matrix
            print("Plotting confusion matrices...")
            self.genre_predictor.plot_confusion_matrix(X_test, y_test)
            print("Confusion matrices saved as 'confusion_matrices.png'")
            
            # Save model
            self.genre_predictor.save_model()
            
        except Exception as e:
            print(f"Error training model: {e}")
        
        input("\nPress Enter to continue...")
    
    def process_sample_movies(self):
        """Translate and generate audio for sample movies"""
        self.clear_screen()
        print("Translating and Generating Audio for Sample Movies...")
        
        if not os.path.exists('cleaned_movie_data.csv'):
            print("Error: Cleaned movie data file not found.")
            print("Please process the data first (Option 1).")
            input("\nPress Enter to continue...")
            return
        
        try:
            num_samples = int(input("Enter number of movies to process (recommended: 50): "))
            
            processed_data = self.translation_service.process_movie_summaries(
                'cleaned_movie_data.csv', 
                num_samples=num_samples
            )
            
            print(f"\nSuccessfully processed {len(processed_data)} movie summaries.")
            print(f"Translations saved in 'translations' directory.")
            print(f"Audio files saved in 'audio' directory.")
            
        except Exception as e:
            print(f"Error processing sample movies: {e}")
        
        input("\nPress Enter to continue...")
    
    def analyze_summary(self):
        """Analyze a user-provided movie summary"""
        self.clear_screen()
        print("Movie Summary Analysis")
        print("=" * 50)
        
        # Get user input
        print("Enter a movie summary (or type 'exit' to return to main menu):")
        summary = input("> ")
        
        if summary.lower() == 'exit':
            return
        
        # Clean the summary
        cleaned_summary = clean_text(summary)
        
        while True:
            self.clear_screen()
            print("Movie Summary Analysis")
            print("=" * 50)
            print(f"Original Summary: {summary}")
            print(f"Cleaned Summary: {cleaned_summary}")
            print("\nOptions:")
            print("1. Convert Summary to Audio")
            print("2. Predict Genre")
            print("3. Return to Main Menu")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                # Convert to audio
                print("\nAvailable Languages:")
                for idx, lang in enumerate(self.translation_service.supported_languages.keys(), 1):
                    print(f"{idx}. {lang.capitalize()}")
                
                lang_choice = input("\nSelect language (1-4): ")
                
                try:
                    lang_idx = int(lang_choice) - 1
                    languages = list(self.translation_service.supported_languages.keys())
                    selected_language = languages[lang_idx]
                    
                    print(f"\nTranslating and generating audio in {selected_language}...")
                    translated_text, audio_file = self.translation_service.translate_and_speak(
                        cleaned_summary, selected_language
                    )
                    
                    print(f"Translation: {translated_text}")
                    print(f"Playing audio...")
                    play_audio(audio_file)
                    
                except Exception as e:
                    print(f"Error: {e}")
                
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                # Predict genre
                if not self.model_loaded:
                    print("\nError: Genre prediction model not loaded.")
                    print("Please train the model first (Option 2).")
                else:
                    try:
                        print("\nPredicting genres...")
                        predicted_genres = self.genre_predictor.predict_genres(cleaned_summary)
                        
                        if predicted_genres:
                            # Clean up genre names for display
                            cleaned_genres = [g.replace('}', '') for g in predicted_genres]
                            # Remove duplicates
                            unique_genres = list(set(cleaned_genres))
                            print(f"Predicted Genres: {', '.join(unique_genres)}")
                        else:
                            print("No genres could be predicted for this summary.")
                    except Exception as e:
                        print(f"Error predicting genres: {e}")
                
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                # Return to main menu
                return
            
            else:
                print("Invalid choice. Please try again.")
                input("\nPress Enter to continue...")
    
    def display_statistics(self):
        """Display statistics about the dataset"""
        self.clear_screen()
        print("Dataset Statistics")
        print("=" * 50)
        
        if not os.path.exists('cleaned_movie_data.csv'):
            print("Error: Cleaned movie data file not found.")
            print("Please process the data first (Option 1).")
        else:
            try:
                display_data_stats('cleaned_movie_data.csv')
                print("\nGenre distribution plot saved as 'genre_distribution.png'")
                print("Summary length distribution saved as 'summary_length_distribution.png'")
            except Exception as e:
                print(f"Error displaying statistics: {e}")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Run the main program loop"""
        while True:
            self.display_menu()
            choice = input("Enter your choice (1-6): ")
            
            if choice == '1':
                self.process_data()
            elif choice == '2':
                self.train_model()
            elif choice == '3':
                self.process_sample_movies()
            elif choice == '4':
                self.analyze_summary()
            elif choice == '5':
                self.display_statistics()
            elif choice == '6':
                print("Thank you for using the Movie Summary Analysis System!")
                sys.exit(0)
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")

if __name__ == "__main__":
    system = MovieAnalysisSystem()
    system.run()

def preprocess_data(self, train_df, test_df=None):
    # Before creating the MultiLabelBinarizer, clean the genre lists
    train_df['genres'] = train_df['genres'].apply(
        lambda genres: [g.strip().replace('}', '') for g in genres]
    )
    if test_df is not None:
        test_df['genres'] = test_df['genres'].apply(
            lambda genres: [g.strip().replace('}', '') for g in genres]
        )
        
 
