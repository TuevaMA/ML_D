from django import forms

class User_bank_Form(forms.Form):
    kod_city = forms.IntegerField(label='Kod_city',initial='3')
    age = forms.IntegerField(label='age',initial='21')
    money = forms.DecimalField(label='money', max_digits=15, decimal_places=2,initial='20000.74')

class User_Haus_Form(forms.Form):
    floorNumber = forms.IntegerField(label='floorNumber',initial='2')
    floorsTotal = forms.DecimalField(label='floorsTotal',max_digits=20, decimal_places=1,initial='5.0')
    totalArea = forms.DecimalField( label='totalArea',max_digits=20, decimal_places=2, initial='50.0')
    kitchenArea = forms.DecimalField(label='kitchenArea',max_digits=20, decimal_places=1, initial='6.0')
    latitude = forms.DecimalField(label='latitude',max_digits=15, decimal_places=8, initial='55.486698')
    longitude = forms.DecimalField(label='longitude',max_digits=15, decimal_places=8, initial='37.59532100')
