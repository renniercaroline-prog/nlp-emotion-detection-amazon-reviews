"""
GoEmotions model processor for Amazon reviews emotion analysis
Processes text data and outputs emotion scores for all 27 emotions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class GoEmotionsProcessor:
    def __init__(self):
        """
        Initialize the GoEmotions model for emotion detection
        Uses the 27-emotion classification model from Google Research
        """
        # Use the most accurate GoEmotions model
        self.model_name = "SamLowe/roberta-base-go_emotions"
        print(f"Loading GoEmotions model: {self.model_name}")
        
        try:
            # Initialize the emotion classification pipeline
            self.emotion_pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True,
                device=-1  # Use CPU, change to 0 if you have GPU
            )
            
            # Test model and get emotion labels
            test_result = self.emotion_pipeline("This is a test")
            self.emotion_labels = [item['label'] for item in test_result[0]]
            print(f"Model loaded successfully. Detected {len(self.emotion_labels)} emotions.")
            print(f"Emotions: {self.emotion_labels}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def process_text_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """
        Process a batch of texts and return emotion scores
        
        Args:
            texts: List of text strings to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of dictionaries containing emotion scores for each text
        """
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"Processing {len(texts)} texts in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            print(f"Processing batch {batch_num}/{total_batches}")
            
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                # Handle missing or empty text
                if pd.isna(text) or str(text).strip() == "":
                    emotion_scores = {label: 0.0 for label in self.emotion_labels}
                else:
                    # Truncate text to model's token limit (approximately 2000 characters)
                    text_truncated = str(text)[:2000]
                    
                    try:
                        # Get emotion scores from model
                        emotion_result = self.emotion_pipeline(text_truncated)
                        emotion_scores = {item['label']: item['score'] for item in emotion_result[0]}
                    except Exception as e:
                        print(f"Error processing text: {e}")
                        # Return zeros if processing fails
                        emotion_scores = {label: 0.0 for label in self.emotion_labels}
                
                batch_results.append(emotion_scores)
            
            results.extend(batch_results)
        
        print("Emotion processing complete")
        return results
    
    def process_reviews_dataframe(self, df: pd.DataFrame, text_col: str = 'Text') -> pd.DataFrame:
        """
        Process a DataFrame of reviews and add emotion columns
        
        Args:
            df: Input DataFrame with review text
            text_col: Name of column containing text to analyze
            
        Returns:
            DataFrame with original data plus 27 emotion score columns
        """
        print(f"Processing {len(df)} reviews for emotion detection")
        
        # Validate input
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in DataFrame")
        
        # Process all texts
        emotion_results = self.process_text_batch(df[text_col].tolist())
        
        # Convert results to DataFrame
        emotions_df = pd.DataFrame(emotion_results)
        
        # Verify we got all expected emotion columns
        missing_emotions = set(self.emotion_labels) - set(emotions_df.columns)
        if missing_emotions:
            print(f"Warning: Missing emotion columns: {missing_emotions}")
            # Add missing columns with zeros
            for emotion in missing_emotions:
                emotions_df[emotion] = 0.0
        
        # Reorder columns to match emotion labels order
        emotions_df = emotions_df[self.emotion_labels]
        
        # Combine with original DataFrame
        result_df = pd.concat([df.reset_index(drop=True), emotions_df.reset_index(drop=True)], axis=1)
        
        print(f"Added {len(self.emotion_labels)} emotion columns to dataset")
        return result_df
    
    def save_results(self, df: pd.DataFrame, filename: str):
        """
        Save processed results to CSV file
        
        Args:
            df: DataFrame with emotion scores
            filename: Output filename
        """
        output_path = Path('/Users/carolinerennier/Desktop/Emotion_Project/Data/Processed') / filename
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Output shape: {df.shape}")


def process_emotion_analysis():
    """
    Main function to run emotion analysis on processed reviews
    """
    print("GOEMOTIONS EMOTION ANALYSIS")
    
    # File paths
    input_file = "/Users/carolinerennier/Desktop/Emotion_Project/Data/Processed/reviews_cleaned_full.csv"
    output_file = "reviews_with_emotions.csv"
    
    try:
        # Load the processed reviews
        print(f"\n1. Loading processed reviews from {input_file}")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} reviews with columns: {list(df.columns)}")
        
        # Initialize emotion processor
        print(f"\n2. Initializing GoEmotions model...")
        processor = GoEmotionsProcessor()
        
        # Process emotions
        print(f"\n3. Processing emotions for all reviews...")
        df_with_emotions = processor.process_reviews_dataframe(df)
        
        # Save results
        print(f"\n4. Saving results...")
        processor.save_results(df_with_emotions, output_file)
        
        # Show summary
        emotion_cols = processor.emotion_labels
        print(f"\nPROCESSING COMPLETE")
        print(f"Original columns: {len(df.columns)}")
        print(f"Final columns: {len(df_with_emotions.columns)}")
        print(f"Added emotion columns: {len(emotion_cols)}")
        print(f"Emotion columns: {emotion_cols}")
        
        # Show sample emotion scores for first few reviews
        print(f"\nSample emotion scores (first 3 reviews):")
        sample_emotions = df_with_emotions[emotion_cols].head(3)
        for idx, row in sample_emotions.iterrows():
            top_emotions = row.nlargest(3)
            print(f"Review {idx+1} top emotions: {dict(top_emotions)}")
        
        print(f"\nOutput file: /Users/carolinerennier/Desktop/Emotion_Project/Data/Processed/{output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        print("Make sure you have run the data preprocessing step first")
    except Exception as e:
        print(f"Error during emotion processing: {e}")
        print("Check your input file and try again")


if __name__ == "__main__":
    # Run the emotion analysis
    process_emotion_analysis()