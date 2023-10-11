from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.http import JsonResponse
from .phishing__email import load_or_train_model, preprocess_text, predict_phishing

# Carga el modelo y el vectorizador al inicio del módulo
model, vectorizer = load_or_train_model()
# Create your views here.
@csrf_protect
def main(request):
    if request.method == 'POST':
        user_input = request.POST['email_text']
        if user_input:
            # Cargar o entrenar el modelo
            
            
            # Preprocesar el texto
            processed_input = preprocess_text(user_input)
            
            # Realizar la predicción
            result = predict_phishing(processed_input, model, vectorizer)
            
            return JsonResponse({'result': int(result)})
    
    return render(request, 'main_page.html')