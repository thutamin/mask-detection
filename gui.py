from typing import Sized
import PySimpleGUIQt as sg
from PIL import Image
import os.path
import io
from numpy.core.fromnumeric import resize
import tensorflow
from tensorflow.keras.models import Model
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto

model = tensorflow.keras.models.load_model('modelaug.hdf5')
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
keras_model = tensorflow.keras.models.load_model('modelaug.hdf5')
file_name = "haarcascade_frontalface_alt2.xml"
classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

label = {
            0: {"name": "Mask only on the chin", "color": (51, 153, 255), "id": 0},
            1: {"name": "Mask below the nose", "color": (255, 255, 0), "id": 1},
            2: {"name": "Mask not on", "color": (0, 0, 255), "id": 2},
            3: {"name": "Mask on", "color": (0, 102, 51), "id": 3},
        }

def classify_image(imgpath):
    img = cv2.imread(imgpath)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = image.img_to_array(rgb_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    prediction = np.argmax(preds)
    return label[prediction]["name"]


layout = [
    [sg.Text('', pad=(0,0),key='-EXPAND0-'), sg.Image('titleimg.png'),sg.Text('', pad=(0,0),key='-EXPAND4-')],
    [sg.Text('', pad=(0,0),key='-EXPAND2-'), sg.Image('icon.png'),sg.Text('', pad=(0,0),key='-EXPAND-')],
    [sg.Text("Student Name: "), sg.Input(key='-STUDENTNAME-')],
    [sg.Text("Student Number"), sg.Input(key='-STUDENTNUMBER-')],
    [sg.Button('Submit')]]


window = sg.Window("Mask Master", layout, icon='icon.png').Finalize()
window.maximize()
win1_active = False
win2_active = False
win3_active = False
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        window.close()
        break
    if event == 'Submit' and win1_active == False:
        win1_active = True
        ilayout = [
            [sg.Text('', pad=(0,0),key='-EXPAND0-'), sg.Image('titleimg.png'),sg.Text('', pad=(0,0),key='-EXPAND4-')],
            [sg.Text('', pad=(0,0),key='-EXPAND2-'), sg.Image('icon.png'),sg.Text('', pad=(0,0),key='-EXPAND-')],
            [sg.Text(text='Placeholder', key='-MSG-')],
            [sg.Button('Classification'), sg.Button('Detection')]
        ]
        message = "Welcome: " + values["-STUDENTNAME-"] + ", " + values["-STUDENTNUMBER-"]
        window.Hide()
        window1 = sg.Window("Mask Master", ilayout, icon='icon.png').Finalize()
        window1["-MSG-"].update(message)
        window1.maximize()
        while True:
            ev2, vals2 = window1.Read()
            if ev2 == sg.WIN_CLOSED or ev2 == 'Exit':
                window1.Close()
                win1_active = False
                window.UnHide()
                break
            if ev2 == 'Classification' and win2_active == False:
                window1.Hide()
                win2_active = True
                file_list_column = [
                    [
                        sg.Text("Image Folder"),
                        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
                        sg.FolderBrowse(),
                    ],
                    [
                        sg.Listbox(
                            values=[], enable_events=True, size=(40, 10),
                            key='-FILE_LIST-'
                        )
                    ],
                    [
                        sg.Button('Process'), sg.Text('Classification:', key='-CLASSRESULT-')

                    ]
                ]
                image_viewer_column = [
                    [sg.Text('Choose an image from the selected directory on the left')],
                    [sg.Text(size=(40, 1), key='-TOUT-')],
                    [sg.Text('', pad=(0,0),key='-EXPAND5-'), sg.Image(key='-IMAGE-'), sg.Text('', pad=(0,0),key='-EXPAND6-')],
                ]
                classlayout = [
                    [
                        sg.Column(file_list_column),
                        sg.VSeperator(),
                        sg.Column(image_viewer_column),
                    ]
                ]
                window2 = sg.Window("Mask Master", classlayout, icon='icon.png').Finalize()
                window2.maximize()
                while True:
                    ev3, vals3 = window2.Read()
                    if ev3 == sg.WIN_CLOSED or ev3 == 'Exit':
                        window2.Close()
                        win2_active = False
                        window1.UnHide()
                        break
                    if ev3 == 'Process':
                        result = 'Classification: ' + str(classify_image(filename))
                        window2['-CLASSRESULT-'].update(result)
                        print(result)

                    if ev3 == "-FOLDER-":
                        folder = vals3["-FOLDER-"]
                        try:
                            file_list = os.listdir(folder)
                        except:
                            file_list = []
                        fnames = [
                            f
                            for f in file_list
                            if
                            os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(
                                (".png", ".jpg", ".jpeg", ".tiff", ".bmp"))

                        ]
                        window2["-FILE_LIST-"].update(fnames)
                    elif ev3 == "-FILE_LIST-":
                        try:
                            filename = os.path.join(
                                vals3["-FOLDER-"], vals3["-FILE_LIST-"][0]
                            )
                            window2["-TOUT-"].update(filename)
                            window2["-IMAGE-"].update(filename=filename)
                        except:
                            pass
            if ev2 == 'Detection':
                window1.Hide()
                detlayout = [[sg.Text('', pad=(0,0),key='-EXPAND0-'), sg.Image('titleimg.png'),sg.Text('', pad=(0,0),key='-EXPAND4-')],
                             [sg.Text('', pad=(0,0),key='-EXPAND5-'), sg.Image(filename='', key='-image-'), sg.Text('', pad=(0,0),key='-EXPAND6-')],
                             [sg.Text('', pad=(0,0),key='-EXPAND510-'), sg.Text('', key='-ENTRY-', size=(40, 1)), sg.Text('', pad=(0,0),key='-EXPAND511-')]]
                window3 = sg.Window('Mask Master', detlayout, no_titlebar=False,
                                   location=(0, 0), icon='icon.png').Finalize()
                window3.maximize()
                image_elem = window3['-image-']

                cam = cv2.VideoCapture(0)
                while True:
                    event, values = window3.read(timeout=0)
                    status, frame = cam.read()

                    if event == sg.WIN_CLOSED or event == 'Exit':
                        window3.Close()
                        window1.UnHide()
                        cam.release()
                        cv2.destroyAllWindows()
                        break

                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    faces = classifier.detectMultiScale(rgb)

                    for x, y, w, h in faces:
                        color = (0, 0, 0)
                        rgb_face = rgb[y:y + h + 50, x:x + w + 50]

                        if rgb_face.shape[0] >= 200 and rgb_face.shape[1] >= 200:
                            rgb_face = cv2.resize(rgb_face, (300, 300))
                            rgb_face = rgb_face / 255
                            rgb_face = np.expand_dims(rgb_face, axis=0)
                            rgb_face = rgb_face.reshape((1, 300, 300, 3))
                            pred = np.argmax(keras_model.predict(rgb_face))
                            classification = label[pred]["name"]
                            color = label[pred]["color"]
                            if pred == 3:
                                print("You may enter the builiding")
                                window3['-ENTRY-'].update("You may enter the building")
                            else:
                                window3['-ENTRY-'].update("")

                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, label[pred]["id"])
                            cv2.putText(frame, classification, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                                        cv2.LINE_AA)
                            cv2.putText(frame, f"{len(faces)} detected face", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 0), 2, cv2.LINE_AA)

                    imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
                    image_elem.update(data=imgbytes)
