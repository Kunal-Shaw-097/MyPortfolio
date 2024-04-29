from django import forms

class Qform(forms.Form):
    text_input = forms.CharField(label='Enter Text', max_length=100)