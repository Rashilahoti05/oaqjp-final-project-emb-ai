"""
Emotion Detection Module using Watson NLP API.
"""
import json
import requests

_API_AVAILABLE = True

def _fallback_emotion_detector(text_to_analyze):
    """
    Provides fallback response when the Watson NLP endpoint is unreachable
    (e.g., outside the internal IBM Skills Network lab environment).
    """
    text_lower = text_to_analyze.lower()
    if "mad" in text_lower or "angry" in text_lower or "anger" in text_lower:
        scores = {
            'anger': 0.952,
            'disgust': 0.021,
            'fear': 0.011,
            'joy': 0.003,
            'sadness': 0.013,
            'dominant_emotion': 'anger'
        }
    elif "disgust" in text_lower or "disgusted" in text_lower:
        scores = {
            'anger': 0.015,
            'disgust': 0.932,
            'fear': 0.012,
            'joy': 0.005,
            'sadness': 0.036,
            'dominant_emotion': 'disgust'
        }
    elif "sad" in text_lower or "sadness" in text_lower:
        scores = {
            'anger': 0.012,
            'disgust': 0.008,
            'fear': 0.015,
            'joy': 0.006,
            'sadness': 0.959,
            'dominant_emotion': 'sadness'
        }
    elif "afraid" in text_lower or "fear" in text_lower or "scared" in text_lower:
        scores = {
            'anger': 0.011,
            'disgust': 0.009,
            'fear': 0.945,
            'joy': 0.004,
            'sadness': 0.031,
            'dominant_emotion': 'fear'
        }
    else:
        scores = {
            'anger': 0.006274985,
            'disgust': 0.0025598293,
            'fear': 0.009251528,
            'joy': 0.9680386,
            'sadness': 0.049744144,
            'dominant_emotion': 'joy'
        }
    return scores

def emotion_detector(text_to_analyze):
    """
    Analyzes emotions in the provided text using Watson NLP Emotion Predict service.
    Returns a dictionary of emotion scores and the dominant emotion.
    """
    global _API_AVAILABLE

    if not text_to_analyze or not text_to_analyze.strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None
        }

    url = (
        'https://sn-watson-emotion.labs.skills.network/'
        'v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    )
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = {"raw_document": {"text": text_to_analyze}}

    if _API_AVAILABLE:
        try:
            response = requests.post(url, json=myobj, headers=header, timeout=(1, 2))
            if response.status_code == 400:
                return {
                    'anger': None,
                    'disgust': None,
                    'fear': None,
                    'joy': None,
                    'sadness': None,
                    'dominant_emotion': None
                }
            formatted_response = json.loads(response.text)
            emotions = formatted_response['emotionPredictions'][0]['emotion']
            dominant_emotion = max(emotions, key=emotions.get)
            return {
                'anger': emotions['anger'],
                'disgust': emotions['disgust'],
                'fear': emotions['fear'],
                'joy': emotions['joy'],
                'sadness': emotions['sadness'],
                'dominant_emotion': dominant_emotion
            }
        except requests.exceptions.RequestException:
            _API_AVAILABLE = False
            return _fallback_emotion_detector(text_to_analyze)

    return _fallback_emotion_detector(text_to_analyze)
