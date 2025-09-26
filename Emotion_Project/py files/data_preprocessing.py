"""
Data preprocessing utilities for Amazon reviews emotion analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Tuple, List, Optional

class ReviewDataPreprocessor:
    def __init__(self, data_path: str):
        """
        Initialize the preprocessor with data path
        
        Args:
            data_path: Path to the raw data file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        print(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} reviews")
        return self.df
    
    def clean_text(self, text: str) -> str:
        """
        Clean review text by removing HTML and normalizing whitespace
        
        Args:
            text: Raw review text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
            
        # Convert to string just in case
        text = str(text)
        
        # Remove HTML tags - they don't add value for emotion analysis
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace - multiple spaces become single spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def preprocess_reviews(self, 
                          text_col: str = 'Text',
                          rating_col: str = 'Score',
                          min_text_length: int = 10,
                          max_text_length: int = 1000) -> pd.DataFrame:
        """
        Preprocess the reviews data by cleaning text, filtering, and creating features
        
        Args:
            text_col: Name of the text column
            rating_col: Name of the rating column
            min_text_length: Minimum text length to keep (filter out too short)
            max_text_length: Maximum text length to keep (filter out too long)
            
        Returns:
            Preprocessed DataFrame with cleaned data and new features
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Starting preprocessing...")
        original_count = len(self.df)
        
        # Work on a copy to avoid modifying original
        df_processed = self.df.copy()
        
        # Validate that required columns exist
        if text_col not in df_processed.columns:
            raise ValueError(f"Text column '{text_col}' not found in data")
        if rating_col not in df_processed.columns:
            raise ValueError(f"Rating column '{rating_col}' not found in data")
        
        # Clean the review text
        print("Cleaning text...")
        df_processed[text_col] = df_processed[text_col].apply(self.clean_text)
        
        # Remove rows with empty text - can't analyze emotions on empty reviews
        df_processed = df_processed[df_processed[text_col] != ""]
        print(f"Removed {original_count - len(df_processed)} empty reviews")
        
        # Filter by text length - too short reviews lack context, too long may be spam
        df_processed['text_length'] = df_processed[text_col].str.len()
        length_filter = (
            (df_processed['text_length'] >= min_text_length) & 
            (df_processed['text_length'] <= max_text_length)
        )
        df_processed = df_processed[length_filter]
        print(f"Kept {len(df_processed)} reviews with text length {min_text_length}-{max_text_length}")
        
        # Create binary satisfaction labels - this is our target variable
        # Scores 4-5 are satisfied, 1-3 are unsatisfied
        df_processed['satisfied'] = (df_processed[rating_col] >= 4).astype(int)
        print(f"Created binary satisfaction labels:")
        print(df_processed['satisfied'].value_counts())
        
        # Standardize Score to 0-1 range for better model performance
        df_processed['Score_normalized'] = (df_processed[rating_col] - 1) / 4  # Transform 1-5 to 0-1
        print(f"Created normalized score (0-1 range)")
        
        # Remove duplicates based on text content - same review shouldn't appear twice
        df_processed = df_processed.drop_duplicates(subset=[text_col])
        print(f"Removed duplicates, final count: {len(df_processed)} reviews")
        
        # Feature engineering - add these after basic preprocessing
        print("Creating engineered features...")
        
        # Process Time column if it exists (Unix timestamp to datetime features)
        if 'Time' in df_processed.columns:
            df_processed['Time_dt'] = pd.to_datetime(df_processed['Time'], unit='s')
            df_processed['year'] = df_processed['Time_dt'].dt.year
            df_processed['month'] = df_processed['Time_dt'].dt.month
            df_processed['day_of_week'] = df_processed['Time_dt'].dt.dayofweek
            df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
            print("Created temporal features from Time column")
        
        # Text length features (already have text_length, now add more)
        df_processed['text_length_log'] = np.log1p(df_processed['text_length'])
        
        # Create and encode text length categories
        df_processed['text_length_category'] = pd.cut(df_processed['text_length'], 
                                                     bins=[0, 50, 150, 500, float('inf')],
                                                     labels=['very_short', 'short', 'medium', 'long'])
        # Ordinal encode text length categories
        text_length_mapping = {'very_short': 0, 'short': 1, 'medium': 2, 'long': 3}
        df_processed['text_length_encoded'] = df_processed['text_length_category'].map(text_length_mapping)
        
        # Product ID encoding if it exists
        if 'ProductId' in df_processed.columns:
            product_counts = df_processed['ProductId'].value_counts()
            df_processed['product_review_count'] = df_processed['ProductId'].map(product_counts)
            
            # Create and encode product popularity categories
            df_processed['product_popularity'] = pd.cut(df_processed['product_review_count'],
                                                       bins=[0, 5, 20, 100, float('inf')],
                                                       labels=['niche', 'moderate', 'popular', 'very_popular'])
            # Ordinal encode product popularity
            popularity_mapping = {'niche': 0, 'moderate': 1, 'popular': 2, 'very_popular': 3}
            df_processed['product_popularity_encoded'] = df_processed['product_popularity'].map(popularity_mapping)
            print("Created and encoded product popularity features")
        
        # Drop redundant/unuseful columns for model readiness
        columns_to_drop = [
            'Summary', 'UserId', 'ProfileName',  # Original unnecessary columns
            'Time_dt',  # Keep original Time, drop datetime conversion
            'text_length_category',  # Categorical version - keep encoded only
            'product_popularity',  # Categorical version - keep encoded only
            'satisfied',  # Using Score_normalized for regression
            'text_length_encoded',  # Redundant with text_length_log
            'text_length',  # Raw length - text_length_log is more useful
            'Score',  # Using Score_normalized instead
            'Id',
            'Time',
            'ProductId',
            'HelpfulnessNumerator',
            'HelpfulnessDenominator'
            
        ]
        
        # Only drop columns that actually exist
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        if existing_cols_to_drop:
            df_processed = df_processed.drop(columns=existing_cols_to_drop)
            print(f"Dropped redundant/unnecessary columns: {existing_cols_to_drop}")
        
        print(f"Final model-ready dataset with {len(df_processed.columns)} features")
        print(f"Remaining columns: {list(df_processed.columns)}")
        
        print(f"Feature engineering complete - added temporal, text, and product features")
        
        # Drop unnecessary columns that won't help with emotion analysis
        # Keep Text column for reference and potential future analysis
        columns_to_drop = ['Summary', 'UserId', 'ProfileName']
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
        if existing_cols_to_drop:
            df_processed = df_processed.drop(columns=existing_cols_to_drop)
            print(f"Dropped unnecessary columns: {existing_cols_to_drop}")
            print("Kept Text column for reference")
        
        return df_processed
    
    def create_sample(self, df: pd.DataFrame, 
                     sample_size: int = 1000, 
                     random_state: int = 42,
                     stratify: bool = True) -> pd.DataFrame:
        """
        Create a representative sample for testing and development
        
        Args:
            df: DataFrame to sample from
            sample_size: Number of samples to create
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by satisfaction (maintains class balance)
            
        Returns:
            Sampled DataFrame
        """
        if len(df) <= sample_size:
            print(f"Dataset size ({len(df)}) <= sample size ({sample_size}), returning full dataset")
            return df
        
        if stratify and 'satisfied' in df.columns:
            # Stratified sampling maintains the same proportion of satisfied/unsatisfied
            # This prevents bias in our sample
            sample = df.groupby('satisfied', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // 2), random_state=random_state),
                include_groups=False
            )
        else:
            # Simple random sampling if stratification not needed
            sample = df.sample(n=sample_size, random_state=random_state)
        
        print(f"Created sample of {len(sample)} reviews")
        return sample
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to the designated processed folder"""
        save_path = Path('/Users/carolinerennier/Desktop/Emotion_Project/Data/Processed') / filename
        # Create the directory if it doesn't exist
        save_path.parent.mkdir(exist_ok=True)
        
        df.to_csv(save_path, index=False)
        print(f"Saved processed data to {save_path}")


def load_and_preprocess_data(data_file: str, 
                           text_col: str = 'Text',
                           rating_col: str = 'Score',
                           sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function to load and preprocess data in one step
    
    Args:
        data_file: Path to data file
        text_col: Name of text column
        rating_col: Name of rating column
        sample_size: If provided, create a sample of this size
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = ReviewDataPreprocessor(data_file)
    
    # Load the raw data
    df = preprocessor.load_data()
    
    # Clean and preprocess it
    df_processed = preprocessor.preprocess_reviews(text_col, rating_col)
    
    # Create sample if requested
    if sample_size:
        df_processed = preprocessor.create_sample(df_processed, sample_size)
    
    return df_processed


if __name__ == "__main__":
    # Main execution - process the full dataset and save it
    print("AMAZON REVIEWS DATA PREPROCESSING")
    
    # File paths - update these if your data is located elsewhere
    raw_data_file = "/Users/carolinerennier/Desktop/Emotion_Project/Data/Raw/Reviews.csv"
    
    try:
        # Step 1: Load and preprocess the full dataset
        print("\n1. Loading and preprocessing full dataset...")
        preprocessor = ReviewDataPreprocessor(raw_data_file)
        df_raw = preprocessor.load_data()
        df_processed = preprocessor.preprocess_reviews()
        
        # Step 2: Save the cleaned full dataset
        print("\n2. Saving cleaned dataset...")
        preprocessor.save_processed_data(df_processed, "reviews_cleaned_full.csv")
        
        # Step 3: Show summary statistics
        print(f"\nPREPROCESSING COMPLETE")
        print(f"Dataset Summary:")
        print(f"   Original reviews: {len(df_raw):,}")
        print(f"   Cleaned reviews: {len(df_processed):,}")
        print(f"   Average normalized score: {df_processed['Score_normalized'].mean():.3f}")
        print(f"   Score distribution: {dict(pd.cut(df_processed['Score_normalized'], bins=5).value_counts().sort_index())}")
        print(f"   Columns in final dataset: {list(df_processed.columns)}")
        
        print(f"\nFile created: /Users/carolinerennier/Desktop/Emotion_Project/Data/Processed/reviews_cleaned_full.csv")
        print(f"Ready for emotion analysis.")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Check that the raw data file exists at the specified path")