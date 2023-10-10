from django.shortcuts import render
from django import forms



class textAreaForm(forms.Form):
    text_area_content = forms.CharField( widget = forms.Textarea(attrs={'placeholder':'Ingrese el email a analizar aquí'}))