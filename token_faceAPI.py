from fastai.vision import *
from flask import Flask, json, request

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
import numpy as np
import re
import time
from passporteye import read_mrz
import datetime
from datetime import date as da
from dateutil import relativedelta

from datetime import datetime as dt

import face_recognition
from fuzzywuzzy import fuzz
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)


@app.route("/front", methods=['POST', 'GET'])
def cards():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if request.method == 'POST':
        if 'idtype' not in request.form:
            return json.dumps({"message": 'No idtype found', "success": False, "code": 201})
        idtype = int(request.form['idtype'])
        if 'token' not in request.form:
            return json.dumps({"message": 'No token key found', "success": False, "code": 201})
        token = request.form['token']
        
        df = pd.read_csv('tokens.csv')
        result = df[df['Password'] == token]
        z = result.empty
        if z == True:
            return json.dumps({"message": 'Incorrect Token.Please re-enter token.', "success": False, "code": 201})

        if idtype == 1:                                            # Front Side
            if 'idfront' not in request.files:
                return json.dumps({"message": 'No image of card', "success": False, "code": 201})
            idfront = request.files['idfront']
            idfront.save('./images/' + time_str + '_front_card10.jpg')

            if 'fullname' not in request.form:
                return json.dumps({"message": 'No Name Found', "success": False, "code": 201})
            fullname = request.form['fullname']

            if 'country' not in request.form:
                return json.dumps({"message": 'No country Name Found', "success": False, "code": 201})
            country = request.form['country']
            country = str(country).upper().replace(" ", "")

            if 'idname' not in request.form:
                return json.dumps({"message": 'No idname found', "success": False, "code": 201})
            idname = request.form['idname']
            idname = int(idname)

            if 'p_image' not in request.files:
                return json.dumps({"message": 'No profile image of user', "success": False, "code": 201})
            p_image = request.files['p_image']
            p_image.save('./images/' + time_str + '_p_image10.jpg')

            path = './model_id/'
            learn = load_learner(path, 'id_card_3.1.pkl')
            learn = learn.load('stage-3.1=new')

            img = open_image('./images/' + time_str + '_front_card10.jpg')
            pred_class, pred_idx, outputs = learn.predict(img)
            array = pred_idx.tolist()
            print(pred_class,array)

            if array == 16 or (idname == 1 and country == 'MINNESOTA'):                 # driving card USA (MINNESOTA)
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                     return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                     'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                              "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    ix = 0
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                            {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                             "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                            {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                           "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                            {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                             "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace(".", " ").replace(" ", "")

                dob = re.findall(r"[\d]{1,4}[/-][\d]{1,4}[/-][\d]{1,4}", text)
                dob1 = datetime.datetime.strptime(dob[0], "%m-%d-%Y").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%m-%d-%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%m-%d-%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                             "gender": ''})
                            return json.dumps(
                                    {"data": data, "message": 'User is less than 18 years old', "success": False,
                                     "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,"code": flag})

                a16 = fuzz.partial_ratio(fullname, text)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                            {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                            {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                             "code": flag})

                number = str(re.findall(r"([P|p]{1}[0-9]{12})", text)).replace("[", "").replace("]", "").replace(
                        "'", "")
                if not number:
                    flag.append(16)
                    number = ''
                try:
                    expiry = datetime.datetime.strptime(dob[1], "%m-%d-%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[1], "%m-%d-%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": number, 'issue_date':'', 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                        {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob5 = re.findall(r"[\d]{2}[/-][\d]{4}", text)
                issue = datetime.datetime.strptime(dob5[1], "%m-%Y").strftime("%m/%Y")  # Issue_date
                if not issue:
                    flag.append(24)
                    issue = ''

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict(
                            {"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                            {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                            "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict(
                            {"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                            {"data": data, "message": 'Faces does not matched. Please try again', "success": False,
                             "code": flag})
                else:
                    return json.dumps(
                            {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 13 or(idname == 2 and country == 'INDIA'):                           # indian national card
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                    # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                     'expiry_date': ''})
                    return json.dumps({
                                              "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                              "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    ix = 0
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                            {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                            {"data": data,
                             "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                             "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                            {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                           "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                           "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                            {"data": data,
                             "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                             "success": False, "code": 201})

                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                # removing shadow from image
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).upper().replace(" ", "")
                text = text.replace("/\s+/g, ' '", "")

                date = str(re.findall(r"[\d]{1,4}[/-][\d]{1,4}[/-][\d]{1,4}", text)).replace("]", "").replace("[",
                                                                                                                  "").replace(
                        "'", "")
                try:
                    if date != datetime.datetime.strptime(date, "%d/%m/%Y"):
                        wa12 = datetime.datetime.strptime(date, "%d/%m/%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict(
                                    {"idn": '', 'gender': '', "pincode": '', "issue_date": '', 'expiry_date': ''})
                            return json.dumps(
                                    {"data": data, "message": 'User is less than 18 years old', "success": False,
                                     "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict({"idn": '', 'gender': '', "pincode": '', "issue_date": '', "expiry_date": ''})
                    return json.dumps(
                            {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                             "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                            {"dob": date, "idn": '', 'gender': '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                            {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                             "code": flag})

                number = str(re.findall(r"[0-9]{12}", text)).replace("]", "").replace("[","").replace("'", "")
                if len(number) == 0:
                    number = ''
                    flag.append(16)

                sex = re.findall(r"MALE|FEMALE", text)
                sex = str(sex).replace("[", "").replace("]", "").replace("'", "").replace("'", "")
                if not sex:
                    sex = ''

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict(
                            {"dob": date, "idn": number, "gender": sex, "match": True, "pincode": '', "issue_date": '',
                             'expiry_date': ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True,
                                           "data": data}).replace("svat fa", "").replace("SRR swat", "").replace(
                            "afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                            {"data": data,
                             "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload. ',
                             "success": False, "code": flag})
                else:
                    return json.dumps(
                            {"code": flag, "message": 'ID data extraction failed. Please try again.', "success": False,
                             "data": ''})

            elif array == 10 or (idname == 2 and country == 'FINLAND'):                      #finland identity
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 76:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((11, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{4})", text)


                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{9})", text)).replace("[", "").replace("]", "").replace("'",
                                                                                                        "")  # 9 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict(
                            {"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps({"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 1 or (idname == 2 and country =='ALBANIA'):                             #albania identity
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 70:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace(".", " ").replace("\n",
                                                                                  " ").upper()  # space is not replaced (92 partial ratio)

                dob = re.findall(r"[\d]{1,4}[/-][\d]{1,4}[/-][\d]{1,4}", text)

                dob1 = datetime.datetime.strptime(dob[0], "%d-%m-%Y").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d-%m-%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d-%m-%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)

                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = re.findall(r"[0-9]{9}", text)

                if not number:
                    flag.append(16)
                    number = ''
                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d-%m-%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d-%m-%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": number[0], 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                        return json.dumps({"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict({"dob": dob[0], "idn": number, "issue_date": '', 'expiry_date': '', "pincode": '',
                                 "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d-%m-%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict({"dob": dob[0], "idn": number, "issue_date": issue, 'expiry_date': '', "pincode": '',
                                 "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict(
                        {"dob": dob1, "idn": number[0], 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict(
                        {"dob": dob1, "idn": number[0], 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 2 or (idname == 2 and country =='AUSTRIA'):                                # austria identity
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 70:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n",
                                                                " ").upper()  # space is not replaced (92 partial ratio)

                dob = str(re.findall(r"[\d]{2}[.][\d]{2}[.][\d]{4}", text)).replace("[", "").replace("]",
                                                                                                           "").replace(
                    "'", "")
                dob1 = datetime.datetime.strptime(dob, "%d.%m.%Y").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob != datetime.datetime.strptime(dob, "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob, "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)

                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict({"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"[0-9]{8}", text)).replace("[", "").replace("]", "").replace("'", "")
                if not number:
                    flag.append(16)
                    number = ''

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 7 or (idname == 2 and country =='CHILE'):                              # chile identity
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 60:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace(" ", "").replace("\n","").upper()  # space is not replaced (92 partial ratio)
                print(text)
                dob = re.findall(r"[\d]{1,4}[A-z]{1,3}[\d]{1,4}", text)
                try:
                    c = int(dob[0][-4:])
                    a1 = dt(c, 1, 1)
                    b1 = dt.today()
                    delta = relativedelta.relativedelta(b1, a1)
                    years = delta.years
                    if years < 18:
                        flag.append(19)
                        data = dict({"idn": '', 'gender': '', "pincode": '', "issue_date": '', 'expiry_date': ''})
                        return json.dumps(
                            {"data": data, "message": 'User is less than 18 years old', "success": False, "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict({"idn": '', 'gender': '', "pincode": '', "issue_date": '', "expiry_date": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)

                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict({"dob": dob[0], "idn": '', "issue_date": dob[1], 'expiry_date': dob[2], "pincode": '',
                                 "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"[0-9]{9}", text)).replace("[", "").replace("]", "").replace("'", "")
                if not number:
                    flag.append(16)
                    number = ''
                try:
                    issue = dob[1]  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict({"dob": dob[0], "idn": number, "issue_date": issue, 'expiry_date': '', "pincode": '',
                                 "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    expiry = dob[2]  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                except IndexError:
                    flag.append(23)
                    data = dict({"dob": dob[0], "idn": number, "issue_date": issue, 'expiry_date': expiry, "pincode": '',
                                 "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict(
                        {"dob": dob[0], "idn": number, 'issue_date': dob[1], 'expiry_date': dob[2], "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict(
                        {"dob": dob[0], "idn": number[0], 'issue_date': dob[2], 'expiry_date': dob[2], "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 8 or (idname == 8 and country == 'CZECH'):                            # czech identity card
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 80:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", " ").replace(" ",
                                                                                   "").upper()  # space is not replaced (92 partial ratio)
                print(text)
                dob = re.findall(r"[\d]{1,4}[.][\d]{1,4}[.][\d]{1,4}", text)

                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%m.%d.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%m.%d.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)

                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"[0-9]{9}", text)).replace("[", "").replace("]", "").replace("'", "")

                if not number:
                    flag.append(16)
                    number = ''

                expiry = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                print(expiry)
                if not expiry:
                    flag.append(23)
                    expiry = ''
                exp1 = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%Y")
                today = da.today()
                today = today.strftime("%Y")
                if today > exp1:
                    flag.append(26)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})

                issue = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                if not issue:
                    flag.append(24)
                    issue = ''

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})

            elif array == 12 or (idname == 2 and country =='GERMANY'):  # germany identity card
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 60:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", " ").replace(" ",
                                                                                   "").upper()  # space is not replaced (92 partial ratio)
                print(text)
                dob = re.findall(r"[\d]{1,4}[.][\d]{1,4}[.][\d]{1,4}", text)
                print(dob)
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%m.%d.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%m.%d.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([A-z]{1}[0-9]{8})", text)).replace("[", "").replace("]", "").replace("'", "")

                if not number:
                    flag.append(16)
                    number = ''

                expiry = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                if not expiry:
                    flag.append(23)
                    expiry = ''
                exp1 = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%Y")
                today = da.today()
                today = today.strftime("%Y")
                if today > exp1:
                    flag.append(26)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 22 or (idname == 2 and country =='SLOVAKIA'):  # Slovakia identity card
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 80:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", " ").replace(" ",
                                                                                   "").upper()  # space is not replaced (92 partial ratio)
                print(text)
                dob = re.findall(r"[\d]{1,4}[.][\d]{1,4}[.][\d]{1,4}", text)
                print(dob)
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%m.%d.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%m.%d.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)

                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([A-z]{2}[0-9]{6})", text)).replace("[", "").replace("]", "").replace("'", "")

                if not number:
                    flag.append(16)
                    number = ''
                try:
                    expiry = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''

                    exp1 = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                        {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": dob1, "idn": number, "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    issue = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})

            elif array == 28 or (idname == 2 and country == 'UKRAINE'):  # Ukraine Identity card
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 80:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", " ").replace(" ",
                                                                                   "").upper()  # space is not replaced (92 partial ratio)

                dob = re.findall(r"([\d]{1,2}[\d]{1,2}[\d]{1,4})", text)
                print(dob)
                dob1 = datetime.datetime.strptime(dob[0], "%d%m%Y").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d%m%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d%m%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{8}[-][0-9]{5})", text)).replace("[", "").replace("]", "").replace("'",
                                                                                                                   "")
                if not number:
                    flag.append(16)
                    number = ''
                try:
                    expiry = datetime.datetime.strptime(dob[3], "%d%m%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[3], "%d%m%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})

            elif array == 26 or (idname == 12 and country == 'TURKEY'):  # Turkey Identity card
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 59:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", " ").replace(" ",
                                                                                   "").upper()  # space is not replaced (92 partial ratio)

                dob = re.findall(r"([\d]{1,2}[.][\d]{1,2}[.][\d]{1,4})", text)
                print(dob)
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{11})", text)).replace("[", "").replace("]", "").replace("'", "")
                if not number:
                    flag.append(16)
                    number = ''

                expiry = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                if not expiry:
                    flag.append(23)
                    expiry = ''
                exp1 = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%Y")
                today = da.today()
                today = today.strftime("%Y")
                if today > exp1:
                    flag.append(26)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 21 or (idname == 2 and country=='SERBIA'):            # Serbian Identity card
                flag = []
                fullname = str(fullname).upper().replace(" ", "")
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 59:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", " ").replace(" ",
                                                                                   "").upper()  # space is not replaced (92 partial ratio)

                dob = re.findall(r"([\d]{1,2}[.][\d]{1,2}[.][\d]{1,4}[.])", text)
                print(dob)
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y.").strftime("%d/%m/%Y")  # DOB
                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y."):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y.").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                issue = datetime.datetime.strptime(dob[1], "%d.%m.%Y.").strftime("%d/%m/%Y")  # Issue_date
                if not issue:
                    flag.append(24)
                    issue = ''
                number = str(re.findall(r"([0-9]{9})", text)).replace("[", "").replace("]", "").replace("'", "")
                if not number:
                    flag.append(16)
                    number = ''

                expiry = datetime.datetime.strptime(dob[2], "%d.%m.%Y.").strftime("%d/%m/%Y")  # expiry date
                if not expiry:
                    flag.append(23)
                    expiry = ''
                exp1 = datetime.datetime.strptime(dob[1], "%d.%m.%Y.").strftime("%Y")
                today = da.today()
                today = today.strftime("%Y")
                if today > exp1:
                    flag.append(26)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})

                if results23 == [True]:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 5 or (idname == 14 and country == 'BANGLADESH'):                   # Bangladesh Identity card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ",
                                                                                  "").upper()  # space is not replaced (92 partial ratio)

                dob = re.findall(r"[\d]{1,4}[A-z]{1,3}[\d]{1,4}", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d%b%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%Y,%b,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%d/%m/%Y")  # dob
                a16 = fuzz.partial_ratio(fullname, text)

                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"[0-9]{13}", text)).replace("[", "").replace("]", "").replace("'", "")
                if not number:
                    flag.append(16)
                    number = ''

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 18 or (idname == 2 and country =='PHILIPPINES'):                  # PHILIPPINES Identity card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 110:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ",
                                                                                  "").upper()  # space is not replaced (92 partial ratio)

                dob = str(re.findall(r"[\d]{1,4}[/-][\d]{1,2}[/-][\d]{1,2}", text)).replace("[", "").replace("]",
                                                                                                             "").replace(
                    "'", "")
                print(dob)

                try:
                    if dob != datetime.datetime.strptime(dob, "%Y/%m/%d"):
                        wa12 = datetime.datetime.strptime(dob, "%Y/%m/%d").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                dob1 = datetime.datetime.strptime(dob, "%Y/%m/%d").strftime("%d/%m/%Y")  # dob
                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{4}[-][0-9]{7}[-][0-9]{1})", text)).replace("[", "").replace("]",
                                                                                                             "").replace(
                    "'", "")
                if not number:
                    flag.append(16)
                    number = ''

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 23 or (idname == 2 and country=='SOUTHAFRICA'):               # South Africa Identity card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ",
                                                                                  "").upper()  # space is not replaced (92 partial ratio)

                dob = re.findall(r"[\d]{1,4}[A-z]{1,3}[\d]{1,4}", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d%b%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%Y,%b,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%d/%m/%Y")  # dob
                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"[0-9]{13}", text)).replace("[", "").replace("]", "").replace("'", "")
                if not number:
                    flag.append(16)
                    number = ''

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 4 or (idname == 1 and country =='BANGLADESH'):  # Bangladesh Driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((50, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[A-z]{3}[\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d%b%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%Y,%b,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([A-Z]{2}[0-9]{7}[A-Z|0-9]{6})", text)).replace("[", "").replace("]",
                                                                                                           "").replace(
                    "'", "")
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d%b%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d%b%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d%b%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": '', 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                        {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 25 or( idname == 18 and country =='SRILANKA'):  # Sri lanka driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.2, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": '1In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": '2In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": '3In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": '4In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{9})", text)).replace("[", "").replace("]", "").replace("'",
                                                                                                        "")  # 16 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": '', 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 27 or (idname == 1 and country =='UK'):                             # UK driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.2, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": '1In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": '2In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": '3In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": '4In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((50, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([A-Z|9]{5}[0-9]{6}[A-Z|9]{2}[A-Z|0-9]{3})", text)).replace("[", "").replace(
                    "]", "").replace("'", "")  # 16 character
                if not number:
                    number = ''
                    flag.append(16)

                issue = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                if not issue:
                    flag.append(24)
                    issue = ''

                expiry = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                if not expiry:
                    flag.append(23)
                    expiry = ''
                exp1 = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%Y")
                today = da.today()
                today = today.strftime("%Y")
                if today > exp1:
                    flag.append(26)
                    data = dict({"dob": dob1, "idn": '', 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 14 or (idname == 1 and country =='INDONESIA'):  # Indonesian Driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.2, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": '1In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": '2In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": '3In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": '4In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((20, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[,./-][\d]{2}[,./-][\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d-%m-%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d-%m-%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d-%m-%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{12})", text)).replace("[", "").replace("]", "").replace("'",
                                                                                                         "")  # 16 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[2], "%d-%m-%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                try:
                    expiry = datetime.datetime.strptime(dob[1], "%d-%m-%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[1], "%d-%m-%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": '', 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 3 or (idname == 1 and country =='AUSTRIA'):                       # Austrian driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.2, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": '1In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": '2In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": '3In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": '4In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((100, 100), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{8,9})", text)).replace("[", "").replace("]", "").replace("'",
                                                                                                          "")  # 16 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": '', 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 11 or (idname == 1 and country == 'GERMANY'):  # Germany Driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({
                                          "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((20, 20), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{2})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([A-Z|9]{2}[A-Z|0-9]{9})", text)).replace("[", "").replace("]", "").replace(
                    "'", "")  # 16 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d.%m.%y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d.%m.%y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d.%m.%y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": '', 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 24 or (idname == 1 and country == 'SPAIN'):  # Spanish Driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((20, 20), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[-][\d]{2}[-][\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d-%m-%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d-%m-%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d-%m-%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9|A-Z]{8}[-][A-z]{1})", text)).replace("[", "").replace("]", "").replace(
                    "'", "")  # 16 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d-%m-%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d-%m-%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d-%m-%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": '', 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps({"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data}).replace(
                        "svat fa", "").replace("SRR swat", "").replace("afte ava", "")

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 9 or (idname == 24 and country == 'ESTONIA'):  # Estonia Identity card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)

                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{11})", text)).replace("[", "").replace("]", "").replace("'",
                                                                                                         "")  # 16 character
                if not number:
                    number = ''
                    flag.append(16)

                try:
                    expiry = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": '', 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                        return json.dumps({"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 10 or (idname == 2 and country=='FINLAND'):                    # Finland Identity card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 76:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict({"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((11, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{9})", text)).replace("[", "").replace("]", "").replace("'","")  # 9 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict(
                            {"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 15 or (idname == 1 and country == 'ITALY'):  # Italy driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 76:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((28, 28), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[/][\d]{2}[/][\d]{2,4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d/%m/%y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d/%m/%y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d/%m/%y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([A-z]{2}[0-9]{7}[A-z]{1})", text)).replace("[", "").replace("]", "").replace(
                    "'", "")  # 10 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d/%m/%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d/%m/%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d/%m/%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict(
                            {"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps({"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 17 or (idname == 1 and country == 'NORWAY'):  # Norway Driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 76:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((28, 28), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{4}[-][\d]{2}[-][\d]{2})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%Y-%m-%d"):
                        wa12 = datetime.datetime.strptime(dob[0], "%Y-%m-%d").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%Y-%m-%d").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{11})", text)).replace("[", "").replace("]", "").replace("'",
                                                                                                         "")  # 10 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%Y-%m-%d").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': '', "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': '', "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 19 or (idname == 2 and country == 'PORTUGAL'):                       # Portugal identity card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 76:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((12, 12), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[\d]{2}[\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d%m%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d%m%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps(
                                {"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d%m%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 60:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict({"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{9}[A-Z|0-9]{1,2}[0-9]{1})", text)).replace("[", "").replace("]","").replace("'", "")  # 8 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    expiry = str(re.findall(r"([\d]{2}[\d]{2}[2][0][\d]{2})", text)).replace("[", "").replace("]","").replace("'", "")
                    expiry = datetime.datetime.strptime(expiry, "%d%m%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(expiry, "%d/%m/%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': expiry, "gender": ''})
                        return json.dumps({"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 20 or (idname == 1 and country == 'ROMANIA'):  # Romania driving card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    if w > 76:
                        cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi_color = img3[y:y + h, x:x + w]
                        crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((11, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").replace(":", ".").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d.%m.%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 50:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{13})", text)).replace("[", "").replace("]", "").replace("'","")  # 9 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d.%m.%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 30 or (idname == 2 and country =='USA'):  # USA border identity card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((11, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").replace(":", ".").upper()

                dob = re.findall(r"([\d]{2}[A-Z]{3}[\d]{4})", text)
                print(dob)

                try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d%b%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '',
                                         "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False,
                                 "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%d/%m/%Y")  # dob

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 50:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": dob1, "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{1}[-][0-9]{6}[-][0-9]{1})", text)).replace("[", "").replace("]","").replace("'", "")  # 9 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[1], "%d%b%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    expiry = datetime.datetime.strptime(dob[2], "%d%b%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[2], "%d%b%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict(
                            {"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": dob1, "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 6 or (idname == 2 and country == 'BELGIUM'):             # Belgium identity card
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append(20)
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                dob = re.findall(r"([\d]{2}[.][\d]{2}[.][\d]{4})", text)
                print(dob)

                '''try:
                    if dob[0] != datetime.datetime.strptime(dob[0], "%d%b%Y"):
                        wa12 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%Y,%m,%d")
                        f2 = int(wa12[:4])
                        a1 = dt(f2, 1, 1)
                        b1 = dt.today()
                        delta = relativedelta.relativedelta(b1, a1)
                        years = delta.years
                        if years < 18:
                            flag.append(19)
                            data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                            return json.dumps({"data": data, "message": 'User is less than 18 years old', "success": False, "code": flag})
                except ValueError:
                    flag.append(17)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False, "code": flag})
                dob1 = datetime.datetime.strptime(dob[0], "%d%b%Y").strftime("%d/%m/%Y")   #dob'''

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 50:
                    c1 = True
                else:
                    flag.append(18)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps({"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                number = str(re.findall(r"([0-9]{3}[-][0-9]{7}[-][0-9]{2})", text)).replace("[", "").replace("]","").replace("'", "")  # 9 character
                if not number:
                    number = ''
                    flag.append(16)
                try:
                    issue = datetime.datetime.strptime(dob[0], "%d.%m.%Y").strftime("%d/%m/%Y")  # Issue_date
                    if not issue:
                        flag.append(24)
                        issue = ''
                except IndexError:
                    flag.append(24)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})
                try:
                    expiry = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%d/%m/%Y")  # expiry date
                    if not expiry:
                        flag.append(23)
                        expiry = ''
                    exp1 = datetime.datetime.strptime(dob[1], "%d.%m.%Y").strftime("%Y")
                    today = da.today()
                    today = today.strftime("%Y")
                    if today > exp1:
                        flag.append(26)
                        data = dict(
                            {"dob": '', "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                        return json.dumps(
                            {"code": flag, "message": 'Your ID card is expired.', "success": True, "data": data})
                except IndexError:
                    flag.append(23)
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append(21)
                    data = dict({"dob": '', "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append(22)
                    data = dict({"dob": '', "idn": number, 'issue_date': issue, 'expiry_date': expiry, "gender": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

            elif array == 0 or idname == 3 :                #passport
                flag = []
                fullname = str(fullname).replace(" ", "").upper()

                up2 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)
                if len(faces) < 1:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                             "success": False, "code": flag})
                for (x, y, w, h) in faces:
                    cv2.rectangle(up2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = up2[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict({"dob": '', "idn": '', "issue_date": '', "expiry_date": '', "pincode": ''})
                    return json.dumps(
                            {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                             "success": False, "code": flag})
                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)
                if len(bide_encoding) > 0:
                    bide_encoding = bide_encoding[0]
                else:
                    flag.append(30)
                    data = dict({"dob": '', "idn": '', "issue_date": '', "expiry_date": '', "pincode": ''})
                    return json.dumps(
                        {"data": data,"message": 'Please re-upload your profile image',"success": False, "code": flag})

                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                            {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                             "success": False, "code": 201})

                # removing shadow from image
                rgb_planes = cv2.split(up2)
                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                                 dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                img = cv2.imwrite('./images/' + time_str + '_front_card10.jpg', dst)
                try:
                        mrz = read_mrz('./images/' + time_str + '_front_card10.jpg')
                        mrz_data = mrz.to_dict()
                except AttributeError:
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data, "message": 'Please click clear picture of Passport.', "success": False,
                             "code": 201})

                n = mrz_data['names'].replace(" ", "")
                sep = 'KKK'
                n1 = n.split(sep, 1)[0]

                s = mrz_data['surname'].replace(" ", "")

                no = mrz_data['number'].replace("<", " ").replace(" ", "")
                if not no:
                    no = ''
                    flag.append(16)

                se = mrz_data['sex']
                if not se:
                    se = ''
                try:
                    dob = mrz_data['date_of_birth']
                    dob = datetime.datetime.strptime(dob, "%y%m%d").strftime("%d/%m/%Y")
                    wa2 = datetime.datetime.strptime(dob, "%d/%m/%Y").strftime("%Y,%m,%d")
                    f1 = int(wa2[:4])
                    a1 = dt(f1, 1, 1)
                    b1 = dt.today()
                    delta = relativedelta.relativedelta(b1, a1)
                    years = delta.years
                except ValueError:
                    flag.append(17)
                    data = dict({"idn": no, 'gender': se, "pincode": '', "issue_date": '', "expiry_date": ''})
                    return json.dumps(
                            {"data": data, "message": 'D.O.B. extraction failed. Please try again', "success": False,
                             "code": flag})

                exp = mrz_data['expiration_date']
                exp = datetime.datetime.strptime(exp, "%y%m%d").strftime("%d/%m/%Y")
                if not exp:
                    exp = ''
                    flag.append(23)

                if years < 18:
                    flag.append(19)
                    data = dict({"dob": dob, "idn": no, "issue_date": '', "expiry_date": exp, "pincode": ''})
                    return json.dumps({"data": data, "message": 'User is less than 18 years old.', "success": False,
                                           "code": flag})

                a17 = fuzz.WRatio(fullname, (n1 + s))
                if a17 > 60:
                    c2 = True
                else:
                    flag.append(18)
                    data = dict({"dob": dob, "idn": no, "issue_date": '', 'expiry_date': exp, "pincode": ''})
                    return json.dumps(
                            {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                             "code": flag})

                if results23 == [True] and c2 == True:
                    flag.append(21)
                    data = dict(
                            {"dob": dob, "idn": no, "gender": se, "expiry_date": exp, "match": True, "pincode": ''})
                    return json.dumps({"code": flag, "message": 'ID data extract successfully', "success": True,
                                           "data": data}).replace("\\", "").replace("<", "").replace(" <<", "")

                elif results23 == [False] and c2 == True:
                    flag.append(22)
                    data = dict({"dob": dob, "idn": no, "issue_date": '', 'expiry_date': exp, "pincode": ''})
                    return json.dumps(
                            {"data": data,"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                             "success": False, "code": flag})
                else:
                    return json.dumps({"code": flag, "message": 'ID data extraction failed', "success": False,
                                           "data": ''}).replace("\\", "")

            else:                               #else condition for all other front cards
                flag = []
                fullname = str(fullname).replace(" ", "").upper()
                if not Path('./images/' + time_str + '_p_image10.jpg').is_file():
                    return json.dumps({"message": 'Profile Picture not found', "success": False, "code": 201})

                # crop_pic from ID card
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, 1.3, 7)

                if len(faces) < 1:
                    flag.append([16, 17, 20, 23, 24,27])
                    data = dict({"dob": '', "idn": '', "gender": '', "code": flag, "pincode": "", "issue_date": '',
                                 'expiry_date': ''})
                    return json.dumps({"message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                          "data": data, "code": 201})

                for (x, y, w, h) in faces:
                    cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    roi_color = img3[y:y + h, x:x + w]
                    crop_pic = cv2.imwrite('./images/' + time_str + '_croppic10.jpg', roi_color)

                if not Path('./images/' + time_str + '_croppic10.jpg').is_file():
                    flag.append([16, 17, 20, 23, 24,27])
                    data = dict(
                        {"dob": '', 'gender': '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": 201})

                known_image = face_recognition.load_image_file('./images/' + time_str + '_p_image10.jpg')
                unknown_image = face_recognition.load_image_file('./images/' + time_str + '_croppic10.jpg')
                face_locations = face_recognition.face_locations(unknown_image)
                if not face_locations:
                    flag.append(20)
                    data = dict(
                        {"dob": '', "idn": '', "gender": '', "pincode": "", "issue_date": '', 'expiry_date': ''})
                    return json.dumps({"code": 201,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "data": data})

                face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
                bide_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)
                if results23 == [False]:
                    flag.append(22)
                    data = dict({"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": ''})
                    return json.dumps(
                        {"data": data,
                         "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                         "success": False, "code": 201})

                # removing shadow from image
                img3 = cv2.imread('./images/' + time_str + '_front_card10.jpg')
                rgb_planes = cv2.split(img3)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((11, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).replace("\n", "").replace(" ", "").upper()

                a16 = fuzz.partial_ratio(fullname, text)
                print(a16)
                if a16 > 37:
                    c1 = True
                else:
                    flag.append([16, 17, 18, 23, 24,27])
                    data = dict(
                        {"dob": '', "idn": '', "issue_date": '', 'expiry_date': '', "pincode": '', "gender": ''})
                    return json.dumps(
                        {"data": data, "message": 'Fullname not matched. Please try again', "success": False,
                         "code": flag})

                if results23 == [True] and c1 == True:
                    flag.append([16, 17, 21, 23, 24,27])
                    data = dict({"dob": '', "idn": '', 'issue_date': '', 'expiry_date': '', "gender": ''})
                    return json.dumps({"code": flag, "message": 'ID data extraction', "success": True, "data": data})

                elif results23 == [False] and c1 == True:
                    flag.append([16, 17, 22, 23, 24,27])
                    data = dict({"dob": '', "idn": '', 'issue_date': ' ', 'expiry_date': '', "gender": ''})
                    return json.dumps({"data": data,
                                       "message": 'In-correct ID card supplied or not matched with your recently uploaded profile image, Please re-upload.',
                                       "success": False, "code": flag})
                else:
                    return json.dumps(
                        {"code": flag, "message": 'ID data extraction failed', "success": False, "data": ''})

        if idtype == 2:                                                  # back_side
            if 'idback' not in request.files:
                    return json.dumps({"message": 'No image of card', "success": False, "code": 201})
            idback = request.files['idback']
            idback.save('./images/' + time_str + '_back_card10.jpg')

            if 'idname' not in request.form:
                    return json.dumps({"message": 'No idname found', "success": False, "code": 201})
            idname = request.form['idname']
            idname = int(idname)

            '''if 'front_idn' not in request.form:
                return json.dumps({"message": 'No front id number found', "success": False, "code": 201})'''
            front_idn = request.form['front_idn']
            front_idn = str(front_idn)

            path1 = './model_id'
            learn1 = load_learner(path1, 'id_card_back99.pkl')
            learn1 = learn1.load('stage-22_back')
            img = open_image('./images/' + time_str + '_back_card10.jpg')
            pred_class, pred_idx, outputs = learn1.predict(img)
            array1 = pred_idx.tolist()

            if array1 == 0 or idname == 1:                       # driving_ID back
                front_idn = str(front_idn).upper().replace(" ", "")
                flag = []
                tup = cv2.imread('./images/' + time_str + '_back_card10.jpg')
                rgb_planes = cv2.split(tup)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).upper()
                a17 = fuzz.WRatio(front_idn, text)
                if a17 > 40:
                    flag.append(28)
                else:
                    flag.append(29)
                return json.dumps({"address": text, "code": flag, "success": True})

            elif array1 == 1 or idname == 2:            # national_ID back
                front_idn = str(front_idn).upper().replace(" ", "")
                flag = []
                tup = cv2.imread('./images/' + time_str + '_back_card10.jpg')
                rgb_planes = cv2.split(tup)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).upper()
                a17 = fuzz.WRatio(front_idn, text)
                if a17 > 40:
                    flag.append(28)
                else:
                    flag.append(29)
                return json.dumps({"address": text, "code": flag, "success": True})

            elif array1 == 2 or idname == 3:            # passport_ID back
                front_idn = str(front_idn).upper().replace(" ", "")
                flag = []
                tup = cv2.imread('./images/' + time_str + '_back_card10.jpg')
                rgb_planes = cv2.split(tup)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                                     dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).upper()
                a17 = fuzz.WRatio(front_idn, text)
                if a17 > 40:
                    flag.append(28)
                else:
                    flag.append(29)
                return json.dumps({"address": text, "code": flag, "success": True})

            else:
                front_idn = str(front_idn).upper().replace(" ", "")
                flag = []
                tup = cv2.imread('./images/' + time_str + '_back_card10.jpg')
                rgb_planes = cv2.split(tup)

                result_planes = []
                result_norm_planes = []
                for plane in rgb_planes:
                    dilated_img = cv2.dilate(plane, np.ones((10, 10), np.uint8))
                    bg_img = cv2.medianBlur(dilated_img, 21)
                    diff_img = 255 - cv2.absdiff(plane, bg_img)
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=250, norm_type=cv2.NORM_MINMAX,
                                             dtype=cv2.CV_8UC1)
                    result_planes.append(diff_img)
                    result_norm_planes.append(norm_img)

                result = cv2.merge(result_planes)
                result_norm = cv2.merge(result_norm_planes)
                dst = cv2.fastNlMeansDenoisingColored(result_norm, None, 10, 10, 7, 11)  # removing noise from image
                text = pytesseract.image_to_string(dst).upper()
                a17 = fuzz.WRatio(front_idn, text)
                if a17 > 40:
                    flag.append(28)
                else:
                    flag.append(29)

                return json.dumps({"address": text, "code": flag, "success": True})
        else:
            return json.dumps({"message": 'Wrong idtype given. Give only 1 or 2.', "success": False, "code": 201})


@app.route("/face", methods=['POST', 'GET'])
def face():
    if request.method == 'POST':
        time_str = time.strftime("%Y%m%d-%H%M%S")
        if 'p_image' not in request.files:
            return json.dumps({"message": 'No profile image of user', "success": False, "code": 201})
        p_image = request.files['p_image']
        # idfront = Image.open(request.files['idfront'].stream)
        p_image.save('./images/' + time_str + '_p_image20.jpg')

        img = cv2.imread('./images/' + time_str + '_p_image20.jpg')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        os.remove('./images/' + time_str + '_p_image20.jpg')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        b = len(faces)
        print(b)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            a = len(eyes)
            print(a)
            if a < 1:
                return json.dumps(
                            {"message": 'Do not cover your eyes. Please Try Again.', "success": False, "code": 201})
        if b < 1:
            return json.dumps(
                            {"message": 'Please take picture in passport potrait format.', "success": False, "code": 201})
        if b >= 2:
            return json.dumps(
                            {"message": 'Too many faces found.Please Try Again.', "success": False, "code": 201})
        else:
            return json.dumps({"message": 'Profile Picture Uploaded', "success": True, "code": 200})


@app.route("/")
def hello():
    return "FRONT CARD OCR #16-05"


if __name__ == "__main__":
    app.run(debug=True, threaded=True)