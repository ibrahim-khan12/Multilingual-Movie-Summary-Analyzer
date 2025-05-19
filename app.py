from flask import Flask, render_template, request, jsonify, send_file, session
import os
import pandas as pd
import json
import tempfile
from werkzeug.utils import secure_filename
from data_preprocessing import clean_text
from translation_audio import TranslationAudioService
from genre_prediction import MovieGenrePredictor
from utils import display_data_stats
from data_preprocessing import (
    load_and_clean_summaries,
    extract_metadata,
    create_dataset,
    train_test_split
)

app = Flask(__name__)
app.secret_key = 'movie_analysis_system_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize services
translation_service = TranslationAudioService()
genre_predictor = MovieGenrePredictor()

# Try to load the trained model if it exists
model_loaded = False
if os.path.exists('models/model.pkl'):
    try:
        genre_predictor.load_model()
        model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/process_data', methods=['GET', 'POST'])
def process_data():
    if request.method == 'GET':
        return render_template('process_data.html')
    
    try:
        # Process data
        plot_summaries_path = request.form.get('plot_summaries_path')
        metadata_path = request.form.get('metadata_path')
        
        # Process summaries
        summaries = load_and_clean_summaries(plot_summaries_path)
        
        # Extract metadata
        genres = extract_metadata(metadata_path)
        
        # Create dataset
        output_file = "cleaned_movie_data.csv"
        df = create_dataset(summaries, genres, output_file)
        
        # Split data
        train_df, test_df = train_test_split(df)
        
        return jsonify({
            'success': True, 
            'message': f'Data processing completed successfully! Created dataset with {len(df)} movies.'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    global model_loaded
    
    if request.method == 'GET':
        return render_template('train_model.html', model_loaded=model_loaded)
    
    if not os.path.exists('train_data.csv') or not os.path.exists('test_data.csv'):
        return jsonify({'success': False, 'error': 'Training and test data files not found. Please process the data first.'})
    
    try:
        # Load data
        train_df, test_df = genre_predictor.load_data('train_data.csv', 'test_data.csv')
        
        # Preprocess data
        X_train, y_train, X_test, y_test = genre_predictor.preprocess_data(train_df, test_df)
        
        # Train model
        genre_predictor.train_model(X_train, y_train)
        model_loaded = True
        
        # Evaluate model
        metrics = genre_predictor.evaluate_model(X_test, y_test)
        
        # Plot confusion matrix
        genre_predictor.plot_confusion_matrix(X_test, y_test)
        
        # Save model
        genre_predictor.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'metrics': metrics,
            'confusion_matrix_path': '/static/img/confusion_matrices.png'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/translate_audio', methods=['GET', 'POST'])
def translate_audio():
    if request.method == 'GET':
        return render_template('translate_audio.html', languages=translation_service.supported_languages)
    
    if not os.path.exists('cleaned_movie_data.csv'):
        return jsonify({'success': False, 'error': 'Cleaned movie data file not found. Please process the data first.'})
    
    try:
        num_samples = int(request.form.get('num_samples', 5))
        
        # Process in background (would be better with Celery for production)
        processed_data = translation_service.process_movie_summaries(
            'cleaned_movie_data.csv', 
            num_samples=num_samples
        )
        
        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(processed_data)} movie summaries.',
            'processed': len(processed_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze_summary', methods=['GET', 'POST'])
def analyze_summary():
    if request.method == 'GET':
        return render_template('analyze_summary.html', 
                              languages=translation_service.supported_languages,
                              model_loaded=model_loaded)
    
    summary = request.form.get('summary', '')
    action = request.form.get('action', '')
    
    if not summary:
        return jsonify({'success': False, 'error': 'Please enter a summary'})
    
    # Clean the summary
    cleaned_summary = clean_text(summary)
    
    if action == 'translate':
        language = request.form.get('language', 'spanish')
        try:
            translated_text, audio_file = translation_service.translate_and_speak(cleaned_summary, language)
            
            # Save to a temp file that we can serve
            filename = f"translation_{language}.mp3"
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(audio_file, 'rb') as src_file, open(temp_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
            
            return jsonify({
                'success': True,
                'translated_text': translated_text,
                'audio_url': f'/uploads/{filename}'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    elif action == 'predict':
        if not model_loaded:
            return jsonify({'success': False, 'error': 'Model not loaded. Please train the model first.'})
        
        try:
            predicted_genres = genre_predictor.predict_genres(cleaned_summary)
            
            if predicted_genres:
                # Clean up genre names for display
                cleaned_genres = [g.replace('}', '') for g in predicted_genres]
                # Remove duplicates
                unique_genres = list(set(cleaned_genres))
                return jsonify({
                    'success': True,
                    'predicted_genres': unique_genres
                })
            else:
                return jsonify({
                    'success': True,
                    'predicted_genres': []
                })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid action'})

@app.route('/statistics')
def statistics():
    if not os.path.exists('cleaned_movie_data.csv'):
        return render_template('statistics.html', error='Cleaned movie data file not found. Please process the data first.')
    
    try:
        # Generate statistics
        stats = display_data_stats('cleaned_movie_data.csv', return_dict=True)
        
        # Make the stats JSON serializable
        for key in stats:
            if isinstance(stats[key], pd.DataFrame):
                stats[key] = stats[key].to_dict()
        
        return render_template('statistics.html', stats=stats)
    except Exception as e:
        return render_template('statistics.html', error=str(e))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)
