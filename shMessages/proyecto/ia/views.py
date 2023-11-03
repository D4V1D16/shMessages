from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse
from .phishing__email import load_or_train_model, preprocess_text, predict_phishing
from googletrans import Translator
from langdetect import detect

# Carga el modelo y el vectorizador al inicio del módulo
model, vectorizer = load_or_train_model()
translator = Translator()
# Create your views here.
@csrf_protect
def main(request):
    if request.method == 'POST':
        user_input = request.POST['email_text']
        if user_input:
            # Detectar el idioma del texto de entrada
            detected_language = detect(user_input)

            if detected_language == 'es':
                # Si el idioma detectado es español, preprocesar el texto traducido
                processed_input = preprocess_text(translate_to_english(user_input))
            else:
                # Si el idioma detectado no es español, preprocesar el texto directamente
                processed_input = preprocess_text(user_input)

            # Realizar la predicción
            result = predict_phishing(processed_input, model, vectorizer)

            return JsonResponse({'result': int(result)})
    
    return render(request, 'main_page.html')


def translate_to_english(text):
    # Traduce el texto del español al inglés
    translated = translator.translate(text, src='es', dest='en')
    return translated.text