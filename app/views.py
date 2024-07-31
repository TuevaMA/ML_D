from django.shortcuts import render
from keras.utils import load_img, img_to_array
import dlib  # pip install cmake,  pip install dlib
from skimage import io  # pip install scikit-image
from scipy.spatial import distance
import time
# Установить Keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import pickle
import numpy as np

from app.forms import User_Haus_Form, User_bank_Form


# Create your views here.
def index(request):
    return render(request, "index.html")

def ML1(request):
    if request.method == 'POST':
        IM_name = request.POST.get('radio')
        # Создаем модель с архитектурой VGG16 и загружаем веса, обученные
        # на наборе данных ImageNet
        model = VGG16(weights='imagenet')
        # Загружаем изображение для распознавания, преобразовываем его в массив
        # numpy и выполняем предварительную обработку
        img_path = './app/static/images/' + IM_name + '.jpg'  # 'ship.jpg' 'cat.jpg' 'plane.jpgэ cat1.jpg
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Запускаем распознавание объекта на изображении
        preds = model.predict(x)
        # Печатаем три класса объекта с самой высокой вероятностью
        # convert the probabilities to class labels
        label = decode_predictions(preds)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]
        poroda = label[1]
        ver = label[2]
        return render(request, "ML1.html", {"IM_name": IM_name, "por": poroda, "ver": ver, })
    else:
        IM_name = 'first'
        return render(request, "ML1.html", {"IM_name": IM_name, "por": "нет", "ver": "нет", })

def ML2(request):
    msg = 'Заполните поля'
    if request.method == "POST":
        form = User_Haus_Form(request.POST)
        if form.is_valid():
            floorNumber = form.cleaned_data.get("floorNumber")
            floorsTotal = form.cleaned_data.get("floorsTotal")
            totalArea = form.cleaned_data.get("totalArea")
            kitchenArea = form.cleaned_data.get("kitchenArea")
            latitude = form.cleaned_data.get("latitude")
            longitude = form.cleaned_data.get("longitude")
            filename = './app/static/File/finalized_model.sav'
             # load the model from disk
            loaded_model = pickle.load(open(filename, 'rb'))
            # прогноз для Самойловой. Вот ее квартира:
            X_test = np.array([[floorNumber, floorsTotal, totalArea, kitchenArea,latitude,longitude,]])  #55.786698
            # Предскажем ей цену
            predicted = loaded_model.predict(X_test)  # предсказываем цену квартиры Самойловой
            msg = 'Предсказала отлично!'
            return render(request, 'ML2.html', {'form': form, 'message': msg, 'predicted': predicted, })
    else:
        form = User_Haus_Form()
        return render(request, 'ML2.html', {'form': form, 'message': msg, })


def ML3(request):
    msg ='Заполните поля'
    if request.method == "POST":
       form = User_bank_Form(request.POST)
       if form.is_valid():
           kod_city = form.cleaned_data.get("kod_city")
           age = form.cleaned_data.get("age")
           money = form.cleaned_data.get("money")
           filename = './app/static/File/finalized_modelNB.sav'  #
           # load the model from disk
           loaded_model = pickle.load(open(filename, 'rb'))
           # прогноз для Самойловой. Она в этом банке
           x_test = np.array([[kod_city, age, money]])  # это Самойлова, она в этом банке  -Париж, возраст, вклад
           predicted = loaded_model.predict(x_test)
           # Предсказали ей уход из банка
           msg = 'Предсказала отлично!'
           return render(request, 'ML3.html', {'form': form, 'message': msg, 'predicted': predicted,})
    else:
        form = User_bank_Form()
        return render(request, 'ML3.html', {'form': form, 'message': msg,})

    def ML4(request):
        if request.method == 'POST':
            IM_name2 = request.POST.get('radio')

            """
            Created on 17:29:28 2020
            @author: Самойлова
            """

            sp = dlib.shape_predictor('D:\STA/ML/app/static/File/shape_predictor_68_face_landmarks.dat')
            facerec = dlib.face_recognition_model_v1(
                'D:\STA/ML/app/static/File/dlib_face_recognition_resnet_model_v1.dat')
            detector = dlib.get_frontal_face_detector()
            # -----------------------------------------
            # img = io.imread('foto_comparison/munerman.jpg')
            img = io.imread('D:\STA/ML/app/static/images/samoilova.jpg')
            IM_name1 = 'samoilova';
            win1 = dlib.image_window()
            win1.clear_overlay()
            win1.set_image(img)
            # --------------------
            dets = detector(img, 1)
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.left(), d.top(), d.right(), d.bottom()))
                shape = sp(img, d)
                win1.clear_overlay()
                win1.add_overlay(d)
                win1.add_overlay(shape)
            # --------------------
            print("something")
            time.sleep(5.0)
            face_descriptor1 = facerec.compute_face_descriptor(img, shape)
            # print(face_descriptor1)
            # ВТОРОЕ ФОТО
            # img = io.imread('foto_comparison/sozykin_webcam.jpg')
            img = io.imread(
                'D:\STA/ML/app/static/images/' + IM_name2 + '.jpg')  # samoilova1.jpg')   #samoilov.jpg')  #samoilova1.jpg
            win2 = dlib.image_window()
            win2.clear_overlay()
            win2.set_image(img)
            dets_webcam = detector(img, 1)
            for k, d in enumerate(dets_webcam):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.left(), d.top(), d.right(), d.bottom()))
                shape = sp(img, d)
                win2.clear_overlay()
                win2.add_overlay(d)
                win2.add_overlay(shape)
            face_descriptor2 = facerec.compute_face_descriptor(img, shape)
            print("something")
            time.sleep(5.0)
            evk = distance.euclidean(face_descriptor1, face_descriptor2)
            print("Евклидово расстояние=",
                  evk)  # Евклидово расстояние меньше 0.6, значит две фотографии принадлежат одному человеку
            # ------------------------------
            import numpy as np
            rect = detector(img)[0]
            predictor = dlib.shape_predictor('D:\STA/ML/app/static/File/shape_predictor_68_face_landmarks.dat')
            sp = predictor(img, rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])

            return render(request, "ML4.html", {"IM_name1": IM_name1, "IM_name2": IM_name2, "evk": evk, })
        else:
            IM_name2 = 'first'
            IM_name1 = 'first'
            return render(request, "ML4.html", {"IM_name1": IM_name1, "IM_name2": IM_name2, "por": "нет", })
