from django import forms

class Qform(forms.Form):
    text_input = forms.CharField(label='', max_length=100, required= True)
