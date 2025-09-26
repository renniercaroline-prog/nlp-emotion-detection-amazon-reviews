import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Load and prepare data
df = pd.read_csv("/Users/carolinerennier/Desktop/Emotion_Project/Data/Processed/reviews_with_emotions.csv")
# df = df.iloc[:10000]  # Uncomment if you want to limit to first 10k rows

# Create binary outcome
df['Score_Binary'] = (df['Score_normalized'] > 0.25).astype(int)

# Define emotion categories
positive_emotions = ["admiration", "joy", "excitement", "gratitude", "love", "optimism", 
                    "pride", "relief", "caring", "approval"]

negative_emotions = ["anger", "annoyance", "disappointment", "disapproval", "disgust", 
                    "embarrassment", "fear", "sadness", "remorse", "grief", "nervousness"]

ambiguous_emotions = ["confusion", "curiosity", "realization", "surprise"]

all_emotions = positive_emotions + negative_emotions + ambiguous_emotions

# Remove unnecessary features and clean data
emotion_cols = [col for col in all_emotions if col in df.columns]
df_clean = df[emotion_cols + ['Score_Binary']].dropna(subset=['Score_Binary'])

def evaluate_emotion_category(data, emotions, category_name):
    """Function to evaluate emotion category performance"""
    # Only select emotions that exist in the data
    existing_emotions = [emotion for emotion in emotions if emotion in data.columns]
    
    if not existing_emotions:
        return {
            'category': category_name,
            'auc': 0.0,
            'n_features': 0,
            'importance': {},
            'model': None
        }
    
    df_category = data[existing_emotions + ['Score_Binary']]
    
    X = df_category[existing_emotions]
    y = df_category['Score_Binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    pred_proba = rf_model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, pred_proba)
    
    # Get importance scores
    importance = dict(zip(existing_emotions, rf_model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'category': category_name,
        'auc': auc_score,
        'n_features': len(existing_emotions),
        'importance': importance,
        'model': rf_model
    }

# ==============================================================================
# PART 1: INDIVIDUAL EMOTION ANALYSIS BY CATEGORY
# ==============================================================================

# Evaluate each emotion category
results_positive = evaluate_emotion_category(df_clean, positive_emotions, "Positive")
results_negative = evaluate_emotion_category(df_clean, negative_emotions, "Negative")
results_ambiguous = evaluate_emotion_category(df_clean, ambiguous_emotions, "Ambiguous")
results_all = evaluate_emotion_category(df_clean, all_emotions, "All Emotions")

# Print category comparison
print("=== EMOTION CATEGORY PERFORMANCE ===")
print(f"Positive emotions only ({results_positive['n_features']} features): AUC = {results_positive['auc']:.3f}")
print(f"Negative emotions only ({results_negative['n_features']} features): AUC = {results_negative['auc']:.3f}")
print(f"Ambiguous emotions only ({results_ambiguous['n_features']} features): AUC = {results_ambiguous['auc']:.3f}")
print(f"All emotions ({results_all['n_features']} features): AUC = {results_all['auc']:.3f}")

# ==============================================================================
# PART 2: AMBIGUOUS EMOTIONS DEEP DIVE
# ==============================================================================

print("\n=== AMBIGUOUS EMOTIONS ANALYSIS ===")

# Individual ambiguous emotion correlations
ambiguous_correlations = {}
for emotion in ambiguous_emotions:
    if emotion in df_clean.columns:
        corr, _ = spearmanr(df_clean[emotion], df_clean['Score_Binary'])
        ambiguous_correlations[emotion] = corr

print("Individual ambiguous emotion correlations with satisfaction:")
for emotion, corr in ambiguous_correlations.items():
    if not np.isnan(corr):
        print(f"{emotion}: {corr:.4f}")

# Ambiguous emotion importance ranking
print("\nAmbiguous emotions ranked by Random Forest importance:")
for emotion, importance in results_ambiguous['importance'].items():
    print(f"{emotion}: {importance:.4f}")

# Test ambiguous emotions + top emotions from other categories
top_positive = list(results_positive['importance'].keys())[:3]
top_negative = list(results_negative['importance'].keys())[:3]

results_mixed = evaluate_emotion_category(
    df_clean,
    ambiguous_emotions + top_positive + top_negative,
    "Ambiguous + Top Others"
)

print(f"\nAmbiguous + top 3 positive + top 3 negative: AUC = {results_mixed['auc']:.3f}")

# ==============================================================================
# PART 3: INTERACTION TERMS ANALYSIS
# ==============================================================================

print("\n=== INTERACTION TERMS ANALYSIS ===")

# Create comprehensive interaction terms
df_interactions = df_clean.copy()

# Check if required emotions exist before creating interactions
if 'confusion' in df_interactions.columns and 'disappointment' in df_interactions.columns:
    df_interactions['confusion_disappointment'] = df_interactions['confusion'] * df_interactions['disappointment']

if 'surprise' in df_interactions.columns and 'admiration' in df_interactions.columns:
    df_interactions['surprise_admiration'] = df_interactions['surprise'] * df_interactions['admiration']

if 'surprise' in df_interactions.columns and 'disappointment' in df_interactions.columns:
    df_interactions['surprise_disappointment'] = df_interactions['surprise'] * df_interactions['disappointment']

if 'curiosity' in df_interactions.columns and 'excitement' in df_interactions.columns:
    df_interactions['curiosity_excitement'] = df_interactions['curiosity'] * df_interactions['excitement']

if 'realization' in df_interactions.columns and 'relief' in df_interactions.columns:
    df_interactions['realization_relief'] = df_interactions['realization'] * df_interactions['relief']

# Create intensity measures
pos_cols = [col for col in ['admiration', 'joy', 'excitement', 'gratitude', 'love', 'optimism', 'pride'] if col in df_interactions.columns]
neg_cols = [col for col in ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'sadness'] if col in df_interactions.columns]
amb_cols = [col for col in ['confusion', 'curiosity', 'realization', 'surprise'] if col in df_interactions.columns]

if pos_cols:
    df_interactions['positive_intensity'] = df_interactions[pos_cols].sum(axis=1)
if neg_cols:
    df_interactions['negative_intensity'] = df_interactions[neg_cols].sum(axis=1)
if amb_cols:
    df_interactions['ambiguous_intensity'] = df_interactions[amb_cols].sum(axis=1)

# Cross-category interactions
if pos_cols and neg_cols:
    df_interactions['positive_negative_conflict'] = (
        df_interactions[pos_cols].sum(axis=1) * df_interactions[neg_cols].sum(axis=1)
    )

# Ratios
if 'positive_intensity' in df_interactions.columns and 'negative_intensity' in df_interactions.columns:
    df_interactions['positive_negative_ratio'] = (
        df_interactions['positive_intensity'] / (df_interactions['negative_intensity'] + 0.001)
    )

if 'ambiguous_intensity' in df_interactions.columns:
    df_interactions['ambiguous_clarity'] = 1 / (df_interactions['ambiguous_intensity'] + 0.001)

# Surprise amplification effects
if 'surprise' in df_interactions.columns:
    if 'negative_intensity' in df_interactions.columns:
        df_interactions['surprise_amplified_negative'] = (
            df_interactions['surprise'] * df_interactions['negative_intensity']
        )
    if 'positive_intensity' in df_interactions.columns:
        df_interactions['surprise_amplified_positive'] = (
            df_interactions['surprise'] * df_interactions['positive_intensity']
        )

# Define interaction terms (only those that were actually created)
interaction_terms = [col for col in df_interactions.columns 
                    if col not in df_clean.columns and col != 'Score_Binary']

# Test interactions only
results_interactions = evaluate_emotion_category(df_interactions, interaction_terms, "Interactions Only")

print(f"Interaction terms only ({len(interaction_terms)} features): AUC = {results_interactions['auc']:.3f}")

# Test original + interactions
results_full = evaluate_emotion_category(
    df_interactions,
    emotion_cols + interaction_terms,
    "Original + Interactions"
)

print(f"Original + interactions ({len(emotion_cols) + len(interaction_terms)} features): AUC = {results_full['auc']:.3f}")

# ==============================================================================
# PART 4: FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

print("\n=== COMPREHENSIVE FEATURE IMPORTANCE ===")

# Get top features from full model
full_importance = results_full['importance']

print("Top 20 features overall:")
top_20_features = list(full_importance.keys())[:20]
for i, (feature, importance) in enumerate(list(full_importance.items())[:20]):
    print(f"{i+1}. {feature}: {importance:.4f}")

# Separate by feature type
original_in_top20 = [f for f in top_20_features if f in emotion_cols]
interactions_in_top20 = [f for f in top_20_features if f in interaction_terms]

print(f"\nIn top 20 features:")
print(f"Original emotions: {len(original_in_top20)}")
print(f"Interaction terms: {len(interactions_in_top20)}")

if interactions_in_top20:
    print("\nTop interaction terms:")
    interaction_importance = {k: v for k, v in full_importance.items() if k in interaction_terms}
    for i, (feature, importance) in enumerate(list(interaction_importance.items())[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")

# ==============================================================================
# PART 5: AMBIGUOUS EMOTIONS INTERACTION ANALYSIS
# ==============================================================================

print("\n=== AMBIGUOUS EMOTIONS INTERACTION EFFECTS ===")

# Test each ambiguous emotion's interaction potential
ambiguous_interaction_results = {}

for ambiguous_emotion in ambiguous_emotions:
    if ambiguous_emotion not in df_interactions.columns:
        continue
        
    # Test interactions with top positive/negative emotions
    interaction_features = []
    temp_df = df_interactions.copy()
    
    for pos_emotion in top_positive:
        if pos_emotion in temp_df.columns:
            interaction_name = f"{ambiguous_emotion}_{pos_emotion}"
            temp_df[interaction_name] = temp_df[ambiguous_emotion] * temp_df[pos_emotion]
            interaction_features.append(interaction_name)
    
    for neg_emotion in top_negative:
        if neg_emotion in temp_df.columns:
            interaction_name = f"{ambiguous_emotion}_{neg_emotion}"
            temp_df[interaction_name] = temp_df[ambiguous_emotion] * temp_df[neg_emotion]
            interaction_features.append(interaction_name)
    
    # Test baseline vs with interactions
    baseline_features = [ambiguous_emotion] + [e for e in top_positive + top_negative if e in temp_df.columns]
    full_features = baseline_features + interaction_features
    
    results_baseline = evaluate_emotion_category(temp_df, baseline_features, f"{ambiguous_emotion}_baseline")
    results_with_int = evaluate_emotion_category(temp_df, full_features, f"{ambiguous_emotion}_with_interactions")
    
    improvement = results_with_int['auc'] - results_baseline['auc']
    
    # Get interaction term importance
    interaction_importance = {k: v for k, v in results_with_int['importance'].items() if k in interaction_features}
    
    ambiguous_interaction_results[ambiguous_emotion] = {
        'baseline_auc': results_baseline['auc'],
        'interaction_auc': results_with_int['auc'],
        'improvement': improvement,
        'interaction_importance': interaction_importance
    }
    
    print(f"\n{ambiguous_emotion} interactions:")
    print(f"  Baseline AUC: {results_baseline['auc']:.4f}")
    print(f"  With interactions AUC: {results_with_int['auc']:.4f}")
    print(f"  Improvement: {improvement:.4f}")
    
    if interaction_importance:
        print("  Top interaction terms:")
        for j, (term, imp) in enumerate(list(interaction_importance.items())[:3]):
            print(f"    {term}: {imp:.4f}")

# ==============================================================================
# PART 6: SUMMARY AND RECOMMENDATIONS
# ==============================================================================

print("\n" + "=" * 60)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("=" * 60)

print(f"\nEmotion Category Performance:")
print(f"  Negative emotions: {results_negative['auc']:.3f} AUC ({results_negative['n_features']} features)")
print(f"  Positive emotions: {results_positive['auc']:.3f} AUC ({results_positive['n_features']} features)")
print(f"  Ambiguous emotions: {results_ambiguous['auc']:.3f} AUC ({results_ambiguous['n_features']} features)")

print(f"\nInteraction Effects:")
print(f"  Interactions alone: {results_interactions['auc']:.3f} AUC")
print(f"  Original + interactions: {results_full['auc']:.3f} AUC")
print(f"  Improvement from interactions: {results_full['auc'] - results_all['auc']:.4f}")

print(f"\nAmbiguous Emotions Insights:")
for emotion, result in ambiguous_interaction_results.items():
    print(f"  {emotion}: {result['improvement']:.4f} improvement from interactions")

# Find best performing ambiguous emotion interactions
if ambiguous_interaction_results:
    best_ambiguous_improvements = {k: v['improvement'] for k, v in ambiguous_interaction_results.items()}
    best_ambiguous = max(best_ambiguous_improvements, key=best_ambiguous_improvements.get)
    
    print(f"\nKey Findings:")
    # Most predictive category
    category_aucs = [results_negative['auc'], results_positive['auc'], results_ambiguous['auc']]
    category_names = ["Negative", "Positive", "Ambiguous"]
    best_category = category_names[np.argmax(category_aucs)]
    print(f"  - Most predictive category: {best_category} emotions")
    print(f"  - Best ambiguous emotion for interactions: {best_ambiguous}")
    print(f"  - Interaction terms in top 20: {len(interactions_in_top20)} out of {len(interaction_terms)}")
    
    if interactions_in_top20:
        best_interaction = list(interaction_importance.keys())[0]
        print(f"  - Most important interaction: {best_interaction}")
    
    print("\nBusiness Recommendations:")
    if results_negative['auc'] > results_positive['auc']:
        print(f"  - Focus on reducing negative emotions ({results_negative['auc']:.3f} AUC) over promoting positive ones ({results_positive['auc']:.3f} AUC)")
    
    if max(best_ambiguous_improvements.values()) > 0.01:
        print(f"  - {best_ambiguous} shows strong interaction effects - monitor this emotion in combination with others")
    
    if interactions_in_top20:
        print("  - Consider composite metrics combining multiple emotions rather than single emotion tracking")
    else:
        print("  - Individual emotions are sufficient - complex combinations add little value")

print("\nAnalysis complete!")
