from django.shortcuts import render
from .forms import textAreaForm
from django.http import HttpResponse


# Create your views here.
def main(request):
    return render(request, 'main_page.html')

def envioEmailPish(request):
    if request.method == 'POST':
        form = textAreaForm(request.POST)

        if form.is_valid():
            text_content = form.cleaned_data['text_area_content']

           

            return HttpResponse(f"Text entered: {text_content}")

    else:
        form = textAreaForm()

    return render(request, 'main_page.html', {'form': form})
