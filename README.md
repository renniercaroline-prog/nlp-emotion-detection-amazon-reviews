Consumer Satisfaction Prediction with Emotion Detection

Overview

This project predicts customer satisfaction from Amazon Fine Food Reviews by analyzing the emotions hidden in review text. Using Google’s GoEmotions model (27 emotions), it shows how emotions provide a powerful lens for understanding product performance — going beyond star ratings or “liking” scores.

Why It Matters
	•	Emotions are highly predictive: Models reached 94% AUC using emotion features alone.
	•	Negative emotions (annoyance, disappointment, disgust) are stronger predictors of dissatisfaction than positive ones are of satisfaction.
	•	Ambiguous emotions (confusion, surprise) don’t predict well by themselves, but amplify insights when combined with clear signals.
	•	Emotional balance (positive vs. negative ratio) was the single strongest predictor of overall satisfaction.

Together, this creates a new metric: emotional profiles of products. Businesses can use these profiles to:
	•	Spot dissatisfaction earlier than ratings reveal it.
	•	Track emotional balance across launches or reformulations.
	•	Prioritize fixes that reduce negative emotions instead of chasing higher averages.

Method
	1.	Emotion Detection – GoEmotions model applied to 366k reviews.
	2.	Feature Engineering – Built aggregate signals (positive, negative, ambiguous intensity, emotional ratios, surprise interactions).
	3.	Modeling – Random Forest classification with AUC evaluation.

Applications
	•	Customer Experience: Identify hidden pain points that stars/liking scores miss.
	•	Product Development: Detect emotional drift over time as products evolve.
	•	Market Insights: Compare products by emotional signature, not just numeric averages.

Tech Stack
	•	Python: pandas, numpy, HuggingFace Transformers
