import os
import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QTextEdit, QComboBox, QFileDialog, QMessageBox, 
                            QProgressBar, QTabWidget, QScrollArea, QSpinBox, QSplitter,
                            QGroupBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import shutil

# Import from existing modules
from data_preprocessing import clean_text
from translation_audio import TranslationAudioService
from genre_prediction import MovieGenrePredictor
from utils import play_audio, display_data_stats, check_file_paths

# Worker thread for background processing
class Worker(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, task, *args, **kwargs):
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        try:
            result = self.task(*self.args, progress_callback=self.progress_signal, 
                              status_callback=self.status_signal, **self.kwargs)
            self.finished_signal.emit(True, "Operation completed successfully!")
        except Exception as e:
            self.finished_signal.emit(False, f"Error: {str(e)}")

class MovieAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.translation_service = TranslationAudioService()
        self.genre_predictor = MovieGenrePredictor()
        
        # Check if model exists
        if os.path.exists('models/model.pkl'):
            self.status_message("Loading pre-trained genre prediction model...")
            self.genre_predictor.load_model()
            self.model_loaded = True
        else:
            self.model_loaded = False
            
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Movie Analysis System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4285F4;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #3275E4;
            }
            QPushButton:pressed {
                background-color: #2559C7;
            }
            QLabel {
                font-size: 14px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dcdcdc;
                border-radius: 5px;
                margin-top: 15px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit {
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
        """)
        
        # Create main tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.data_tab = QWidget()
        self.model_tab = QWidget()
        self.translation_tab = QWidget()
        self.analysis_tab = QWidget()
        self.stats_tab = QWidget()
        
        # Add tabs to widget
        self.tabs.addTab(self.data_tab, "Data Processing")
        self.tabs.addTab(self.model_tab, "Model Training")
        self.tabs.addTab(self.translation_tab, "Translation & Audio")
        self.tabs.addTab(self.analysis_tab, "Summary Analysis")
        self.tabs.addTab(self.stats_tab, "Statistics")
        
        # Setup each tab
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_translation_tab()
        self.setup_analysis_tab()
        self.setup_stats_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def setup_data_tab(self):
        layout = QVBoxLayout()
        
        # Info section
        info_group = QGroupBox("Data Processing Information")
        info_layout = QVBoxLayout()
        info_text = QLabel("This section allows you to process raw movie data files and prepare them for analysis.")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Data paths section
        paths_group = QGroupBox("Data Paths")
        paths_layout = QVBoxLayout()
        
        # Get current paths
        paths = check_file_paths()
        
        # Summaries path
        summaries_layout = QHBoxLayout()
        summaries_label = QLabel("Plot Summaries:")
        self.summaries_path = QLabel(paths['plot_summaries'])
        self.summaries_path.setStyleSheet("font-style: italic;")
        summaries_browse = QPushButton("Browse")
        summaries_browse.clicked.connect(lambda: self.browse_file('summaries'))
        summaries_layout.addWidget(summaries_label)
        summaries_layout.addWidget(self.summaries_path, 1)
        summaries_layout.addWidget(summaries_browse)
        paths_layout.addLayout(summaries_layout)
        
        # Metadata path
        metadata_layout = QHBoxLayout()
        metadata_label = QLabel("Metadata:")
        self.metadata_path = QLabel(paths['metadata'])
        self.metadata_path.setStyleSheet("font-style: italic;")
        metadata_browse = QPushButton("Browse")
        metadata_browse.clicked.connect(lambda: self.browse_file('metadata'))
        metadata_layout.addWidget(metadata_label)
        metadata_layout.addWidget(self.metadata_path, 1)
        metadata_layout.addWidget(metadata_browse)
        paths_layout.addLayout(metadata_layout)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Process button and progress bar
        process_group = QGroupBox("Process Data")
        process_layout = QVBoxLayout()
        
        self.data_progress = QProgressBar()
        self.data_progress.setValue(0)
        process_layout.addWidget(self.data_progress)
        
        self.data_status = QLabel("Ready to process data")
        self.data_status.setWordWrap(True)
        process_layout.addWidget(self.data_status)
        
        self.process_btn = QPushButton("Process and Clean Data")
        self.process_btn.clicked.connect(self.process_data)
        process_layout.addWidget(self.process_btn)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # Add stretcher
        layout.addStretch()
        
        # Apply layout
        self.data_tab.setLayout(layout)
        
    def setup_model_tab(self):
        layout = QVBoxLayout()
        
        # Info section
        info_group = QGroupBox("Model Training Information")
        info_layout = QVBoxLayout()
        info_text = QLabel("Train the genre prediction model using processed data. The model will learn to predict movie genres based on summaries.")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Training section
        train_group = QGroupBox("Model Training")
        train_layout = QVBoxLayout()
        
        self.model_progress = QProgressBar()
        self.model_progress.setValue(0)
        train_layout.addWidget(self.model_progress)
        
        self.model_status = QLabel("Ready to train model")
        self.model_status.setWordWrap(True)
        train_layout.addWidget(self.model_status)
        
        self.train_btn = QPushButton("Train Genre Prediction Model")
        self.train_btn.clicked.connect(self.train_model)
        train_layout.addWidget(self.train_btn)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Results section
        results_group = QGroupBox("Model Evaluation Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(150)
        results_layout.addWidget(self.results_text)
        
        # Add placeholder for confusion matrix image
        self.confusion_matrix_label = QLabel("Confusion matrix will appear here after training")
        self.confusion_matrix_label.setAlignment(Qt.AlignCenter)
        self.confusion_matrix_label.setMinimumHeight(300)
        results_layout.addWidget(self.confusion_matrix_label)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Add stretcher
        layout.addStretch()
        
        # Apply layout
        self.model_tab.setLayout(layout)
        
    def setup_translation_tab(self):
        layout = QVBoxLayout()
        
        # Info section
        info_group = QGroupBox("Translation & Audio Generation Information")
        info_layout = QVBoxLayout()
        info_text = QLabel("Process movie summaries by translating them to different languages and generating audio files.")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()
        
        samples_layout = QHBoxLayout()
        samples_label = QLabel("Number of movies to process:")
        self.samples_input = QSpinBox()
        self.samples_input.setRange(1, 1000)
        self.samples_input.setValue(50)
        samples_layout.addWidget(samples_label)
        samples_layout.addWidget(self.samples_input)
        options_layout.addLayout(samples_layout)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Translation progress
        progress_group = QGroupBox("Translation Progress")
        progress_layout = QVBoxLayout()
        
        self.translation_progress = QProgressBar()
        self.translation_progress.setValue(0)
        progress_layout.addWidget(self.translation_progress)
        
        self.translation_status = QLabel("Ready to process")
        self.translation_status.setWordWrap(True)
        progress_layout.addWidget(self.translation_status)
        
        self.translate_btn = QPushButton("Start Translation & Audio Generation")
        self.translate_btn.clicked.connect(self.process_translations)
        progress_layout.addWidget(self.translate_btn)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Results section
        results_group = QGroupBox("Processing Results")
        results_layout = QVBoxLayout()
        
        self.translation_results = QTextEdit()
        self.translation_results.setReadOnly(True)
        self.translation_results.setMinimumHeight(200)
        results_layout.addWidget(self.translation_results)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Add stretcher
        layout.addStretch()
        
        # Apply layout
        self.translation_tab.setLayout(layout)
        
    def setup_analysis_tab(self):
        layout = QVBoxLayout()
        
        # Input section
        input_group = QGroupBox("Enter Movie Summary")
        input_layout = QVBoxLayout()
        
        input_label = QLabel("Enter a movie summary for analysis:")
        input_layout.addWidget(input_label)
        
        self.summary_input = QTextEdit()
        self.summary_input.setPlaceholderText("Type or paste a movie summary here...")
        self.summary_input.setMinimumHeight(150)
        input_layout.addWidget(self.summary_input)
        
        clean_layout = QHBoxLayout()
        self.clean_btn = QPushButton("Clean Text")
        self.clean_btn.clicked.connect(self.clean_summary)
        clean_layout.addWidget(self.clean_btn)
        clean_layout.addStretch()
        input_layout.addLayout(clean_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Analysis section - use a splitter for resizable sections
        analysis_splitter = QSplitter(Qt.Horizontal)
        
        # Genre prediction
        genre_widget = QWidget()
        genre_layout = QVBoxLayout(genre_widget)
        genre_group = QGroupBox("Genre Prediction")
        genre_box_layout = QVBoxLayout()
        
        self.predict_btn = QPushButton("Predict Genres")
        self.predict_btn.clicked.connect(self.predict_genres)
        genre_box_layout.addWidget(self.predict_btn)
        
        genre_results_label = QLabel("Predicted Genres:")
        genre_box_layout.addWidget(genre_results_label)
        
        self.genre_results = QTextEdit()
        self.genre_results.setReadOnly(True)
        self.genre_results.setMinimumHeight(100)
        genre_box_layout.addWidget(self.genre_results)
        
        genre_group.setLayout(genre_box_layout)
        genre_layout.addWidget(genre_group)
        analysis_splitter.addWidget(genre_widget)
        
        # Translation and audio
        audio_widget = QWidget()
        audio_layout = QVBoxLayout(audio_widget)
        audio_group = QGroupBox("Translation & Audio")
        audio_box_layout = QVBoxLayout()
        
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Select Language:")
        self.lang_combo = QComboBox()
        for lang in self.translation_service.supported_languages.keys():
            self.lang_combo.addItem(lang.capitalize())
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        audio_box_layout.addLayout(lang_layout)
        
        self.translate_audio_btn = QPushButton("Translate & Generate Audio")
        self.translate_audio_btn.clicked.connect(self.translate_and_speak)
        audio_box_layout.addWidget(self.translate_audio_btn)
        
        translation_label = QLabel("Translation:")
        audio_box_layout.addWidget(translation_label)
        
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setMinimumHeight(100)
        audio_box_layout.addWidget(self.translation_text)
        
        self.play_audio_btn = QPushButton("Play Audio")
        self.play_audio_btn.setEnabled(False)
        self.play_audio_btn.clicked.connect(self.play_current_audio)
        audio_box_layout.addWidget(self.play_audio_btn)
        
        audio_group.setLayout(audio_box_layout)
        audio_layout.addWidget(audio_group)
        analysis_splitter.addWidget(audio_widget)
        
        # Add splitter to main layout
        layout.addWidget(analysis_splitter)
        
        # Add stretcher
        layout.addStretch()
        
        # Apply layout
        self.analysis_tab.setLayout(layout)
        
    def setup_stats_tab(self):
        layout = QVBoxLayout()
        
        # Info section
        info_group = QGroupBox("Dataset Statistics")
        info_layout = QVBoxLayout()
        info_text = QLabel("View statistics and visualizations of the processed movie dataset.")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Control section
        control_group = QGroupBox("Generate Statistics")
        control_layout = QVBoxLayout()
        
        self.stats_btn = QPushButton("Generate Dataset Statistics")
        self.stats_btn.clicked.connect(self.show_statistics)
        control_layout.addWidget(self.stats_btn)
        
        self.stats_status = QLabel("Ready to generate statistics")
        control_layout.addWidget(self.stats_status)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Results display - using a tab widget for different visualizations
        vis_tabs = QTabWidget()
        
        # General stats tab
        general_tab = QWidget()
        general_layout = QVBoxLayout()
        self.general_stats_text = QTextEdit()
        self.general_stats_text.setReadOnly(True)
        general_layout.addWidget(self.general_stats_text)
        general_tab.setLayout(general_layout)
        vis_tabs.addTab(general_tab, "General Statistics")
        
        # Genre distribution tab
        genre_tab = QWidget()
        genre_layout = QVBoxLayout()
        self.genre_plot_label = QLabel("Genre distribution chart will appear here")
        self.genre_plot_label.setAlignment(Qt.AlignCenter)
        self.genre_plot_label.setMinimumHeight(300)
        genre_layout.addWidget(self.genre_plot_label)
        genre_tab.setLayout(genre_layout)
        vis_tabs.addTab(genre_tab, "Genre Distribution")
        
        # Summary length tab
        length_tab = QWidget()
        length_layout = QVBoxLayout()
        self.length_plot_label = QLabel("Summary length chart will appear here")
        self.length_plot_label.setAlignment(Qt.AlignCenter)
        self.length_plot_label.setMinimumHeight(300)
        length_layout.addWidget(self.length_plot_label)
        length_tab.setLayout(length_layout)
        vis_tabs.addTab(length_tab, "Summary Length")
        
        layout.addWidget(vis_tabs)
        
        # Apply layout
        self.stats_tab.setLayout(layout)
        
    # Utility functions
    def status_message(self, message):
        self.statusBar().showMessage(message)
        
    def show_message(self, title, message, icon=QMessageBox.Information):
        msg = QMessageBox()
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.exec_()
        
    def browse_file(self, file_type):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {file_type} file", "", "All Files (*)")
        if file_path:
            if file_type == 'summaries':
                self.summaries_path.setText(file_path)
            elif file_type == 'metadata':
                self.metadata_path.setText(file_path)
        
    # Functionality implementations
    def process_data(self):
        self.data_progress.setValue(0)
        self.data_status.setText("Starting data processing...")
        
        def process_task(progress_callback, status_callback):
            from data_preprocessing import (
                load_and_clean_summaries,
                extract_metadata,
                create_dataset,
                train_test_split
            )
            
            # Get file paths from GUI
            paths = {
                'plot_summaries': self.summaries_path.text(),
                'metadata': self.metadata_path.text()
            }
            
            # Process summaries
            status_callback.emit("Loading and cleaning summaries...")
            progress_callback.emit(10)
            summaries = load_and_clean_summaries(paths['plot_summaries'])
            
            # Extract metadata
            status_callback.emit(f"Processed {len(summaries)} summaries. Extracting metadata...")
            progress_callback.emit(30)
            genres = extract_metadata(paths['metadata'])
            
            # Create dataset
            status_callback.emit(f"Extracted genres for {len(genres)} movies. Creating dataset...")
            progress_callback.emit(60)
            output_file = "cleaned_movie_data.csv"
            df = create_dataset(summaries, genres, output_file)
            
            # Split data
            status_callback.emit(f"Created dataset with {len(df)} movies. Splitting data...")
            progress_callback.emit(80)
            train_df, test_df = train_test_split(df)
            
            status_callback.emit("Data processing completed!")
            progress_callback.emit(100)
            return True
        
        # Create worker thread
        self.worker = Worker(process_task)
        self.worker.progress_signal.connect(self.data_progress.setValue)
        self.worker.status_signal.connect(self.data_status.setText)
        self.worker.finished_signal.connect(self.process_data_finished)
        self.worker.start()
        
        # Disable button while processing
        self.process_btn.setEnabled(False)
        
    def process_data_finished(self, success, message):
        self.process_btn.setEnabled(True)
        if success:
            self.show_message("Data Processing", "Data processed successfully!")
        else:
            self.show_message("Error", message, QMessageBox.Critical)
            
    def train_model(self):
        self.model_progress.setValue(0)
        self.model_status.setText("Starting model training...")
        
        if not os.path.exists('train_data.csv') or not os.path.exists('test_data.csv'):
            self.show_message("Error", "Training and test data files not found. Please process the data first.", QMessageBox.Critical)
            return
            
        def train_task(progress_callback, status_callback):
            # Load data
            status_callback.emit("Loading data...")
            progress_callback.emit(10)
            train_df, test_df = self.genre_predictor.load_data('train_data.csv', 'test_data.csv')
            
            # Preprocess data
            status_callback.emit("Preprocessing data...")
            progress_callback.emit(30)
            X_train, y_train, X_test, y_test = self.genre_predictor.preprocess_data(train_df, test_df)
            
            # Train model
            status_callback.emit("Training model (this may take a while)...")
            progress_callback.emit(50)
            self.genre_predictor.train_model(X_train, y_train)
            self.model_loaded = True
            
            # Evaluate model
            status_callback.emit("Evaluating model performance...")
            progress_callback.emit(80)
            metrics = self.genre_predictor.evaluate_model(X_test, y_test)
            
            # Plot confusion matrix
            status_callback.emit("Plotting confusion matrices...")
            progress_callback.emit(90)
            self.genre_predictor.plot_confusion_matrix(X_test, y_test)
            
            # Save model
            status_callback.emit("Saving model...")
            progress_callback.emit(95)
            self.genre_predictor.save_model()
            
            status_callback.emit("Model training completed!")
            progress_callback.emit(100)
            
            return metrics
        
        # Create worker thread
        self.worker = Worker(train_task)
        self.worker.progress_signal.connect(self.model_progress.setValue)
        self.worker.status_signal.connect(self.model_status.setText)
        self.worker.finished_signal.connect(self.train_model_finished)
        self.worker.start()
        
        # Disable button while processing
        self.train_btn.setEnabled(False)
        
    def train_model_finished(self, success, message):
        self.train_btn.setEnabled(True)
        if success:
            # Display metrics in results text
            self.results_text.clear()
            self.results_text.append("Model Training Results:\n")
            self.results_text.append(f"Model trained and saved successfully!")
            
            # Check for and display confusion matrix image
            if os.path.exists('confusion_matrices.png'):
                pixmap = QPixmap('confusion_matrices.png')
                pixmap = pixmap.scaled(800, 400, Qt.KeepAspectRatio)
                self.confusion_matrix_label.setPixmap(pixmap)
            
            self.show_message("Model Training", "Model trained successfully!")
        else:
            self.show_message("Error", message, QMessageBox.Critical)
            
    def process_translations(self):
        self.translation_progress.setValue(0)
        self.translation_status.setText("Starting translation process...")
        
        if not os.path.exists('cleaned_movie_data.csv'):
            self.show_message("Error", "Cleaned movie data file not found. Please process the data first.", QMessageBox.Critical)
            return
            
        num_samples = self.samples_input.value()
        
        def translation_task(progress_callback, status_callback):
            status_callback.emit(f"Processing {num_samples} movie summaries...")
            progress_callback.emit(10)
            
            processed_data = self.translation_service.process_movie_summaries(
                'cleaned_movie_data.csv', 
                num_samples=num_samples,
                progress_callback=progress_callback,
                status_callback=status_callback
            )
            
            status_callback.emit(f"Successfully processed {len(processed_data)} movie summaries.")
            progress_callback.emit(100)
            
            return processed_data
        
        # Create worker thread
        self.worker = Worker(translation_task)
        self.worker.progress_signal.connect(self.translation_progress.setValue)
        self.worker.status_signal.connect(self.translation_status.setText)
        self.worker.finished_signal.connect(self.translation_finished)
        self.worker.start()
        
        # Disable button while processing
        self.translate_btn.setEnabled(False)
        
    def translation_finished(self, success, message):
        self.translate_btn.setEnabled(True)
        if success:
            self.translation_results.clear()
            self.translation_results.append("Translation Results:\n")
            self.translation_results.append(message + "\n")
            self.translation_results.append("Translations saved in 'translations' directory.")
            self.translation_results.append("Audio files saved in 'audio' directory.")
            
            self.show_message("Translation Process", "Translation and audio generation completed successfully!")
        else:
            self.show_message("Error", message, QMessageBox.Critical)
            
    def clean_summary(self):
        text = self.summary_input.toPlainText()
        if not text:
            self.show_message("Error", "Please enter a summary to clean", QMessageBox.Warning)
            return
            
        cleaned = clean_text(text)
        self.summary_input.setPlainText(cleaned)
        self.status_message("Summary cleaned")
        
    def predict_genres(self):
        text = self.summary_input.toPlainText()
        if not text:
            self.show_message("Error", "Please enter a summary for prediction", QMessageBox.Warning)
            return
            
        if not self.model_loaded:
            self.show_message("Error", "Model not loaded. Please train the model first.", QMessageBox.Critical)
            return
            
        try:
            self.status_message("Predicting genres...")
            predicted_genres = self.genre_predictor.predict_genres(text)
            
            self.genre_results.clear()
            if predicted_genres:
                # Clean up genre names for display
                cleaned_genres = [g.replace('}', '') for g in predicted_genres]
                # Remove duplicates
                unique_genres = list(set(cleaned_genres))
                self.genre_results.append(", ".join(unique_genres))
            else:
                self.genre_results.append("No genres could be predicted for this summary.")
                
            self.status_message("Genre prediction completed")
        except Exception as e:
            self.show_message("Error", f"Error predicting genres: {str(e)}", QMessageBox.Critical)
            
    def translate_and_speak(self):
        text = self.summary_input.toPlainText()
        if not text:
            self.show_message("Error", "Please enter a summary for translation", QMessageBox.Warning)
            return
            
        language = self.lang_combo.currentText().lower()
        
        try:
            self.status_message(f"Translating to {language} and generating audio...")
            translated_text, audio_file = self.translation_service.translate_and_speak(text, language)
            
            self.translation_text.clear()
            self.translation_text.append(translated_text)
            
            self.current_audio_file = audio_file
            self.play_audio_btn.setEnabled(True)
            
            self.status_message("Translation and audio generation completed")
        except Exception as e:
            self.show_message("Error", f"Error translating: {str(e)}", QMessageBox.Critical)
            
    def play_current_audio(self):
        if hasattr(self, 'current_audio_file') and self.current_audio_file:
            try:
                self.status_message("Playing audio...")
                play_audio(self.current_audio_file)
                self.status_message("Audio playback completed")
            except Exception as e:
                self.show_message("Error", f"Error playing audio: {str(e)}", QMessageBox.Critical)
        else:
            self.show_message("Error", "No audio file available. Generate audio first.", QMessageBox.Warning)
            
    def show_statistics(self):
        if not os.path.exists('cleaned_movie_data.csv'):
            self.show_message("Error", "Cleaned movie data file not found. Please process the data first.", QMessageBox.Critical)
            return
            
        try:
            self.stats_status.setText("Generating statistics...")
            
            # Get stats data
            stats_data = display_data_stats('cleaned_movie_data.csv', gui_mode=True)
            
            # Display text stats
            self.general_stats_text.clear()
            self.general_stats_text.append("Dataset Statistics:\n")
            self.general_stats_text.append(f"Total number of movies: {stats_data['total_movies']}\n")
            self.general_stats_text.append(f"Number of unique genres: {stats_data['unique_genres']}\n")
            self.general_stats_text.append(f"Average summary length: {stats_data['avg_length']:.2f} words\n")
            self.general_stats_text.append(f"Longest summary: {stats_data['max_length']} words\n")
            self.general_stats_text.append(f"Shortest summary: {stats_data['min_length']} words\n\n")
            
            self.general_stats_text.append("Top 10 most common genres:\n")
            for genre, count in stats_data['top_genres']:
                self.general_stats_text.append(f"- {genre}: {count}\n")
            
            # Load and display genre distribution image
            if os.path.exists('genre_distribution.png'):
                pixmap = QPixmap('genre_distribution.png')
                pixmap = pixmap.scaled(800, 400, Qt.KeepAspectRatio)
                self.genre_plot_label.setPixmap(pixmap)
            
            # Load and display summary length distribution image
            if os.path.exists('summary_length_distribution.png'):
                pixmap = QPixmap('summary_length_distribution.png')
                pixmap = pixmap.scaled(800, 400, Qt.KeepAspectRatio)
                self.length_plot_label.setPixmap(pixmap)
                
            self.stats_status.setText("Statistics generated successfully")
        except Exception as e:
            self.show_message("Error", f"Error generating statistics: {str(e)}", QMessageBox.Critical)

def main():
    app = QApplication(sys.argv)
    window = MovieAnalysisGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
