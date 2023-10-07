from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
import io
import os
from PIL import Image, ImageOps
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


model = keras.models.load_model('best_model.h5')


def transform_image(pillow_image):
    target_size = (331, 331)
    resize_image = ImageOps.fit(pillow_image, target_size, Image.LANCZOS)

    grayscale_image = ImageOps.grayscale(resize_image)

    img_array = np.array(grayscale_image)

    img_array_eq = cv2.equalizeHist(img_array)
    img_array_eq_rgb = cv2.cvtColor(img_array_eq, cv2.COLOR_GRAY2RGB)

    normalized_image = img_array_eq_rgb / 255.0

    normalized_image = cv2.resize(normalized_image, (331, 331))

    normalized_image = np.expand_dims(normalized_image, axis=0)

    return normalized_image


def predict(x):
    predictions = model(x)
    predictions = tf.nn.softmax(predictions)
    pred0 = predictions[0]
    label0 = np.argmax(pred0)
    return label0


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            tensor = transform_image(pillow_img)
            prediction = predict(tensor)

            label_names = {0: 'Tidak ada DR (Tidak ada Retinopati Diabetes)',
                           1: 'DR Ringan (Mild)',
                           2: 'DR Sedang (Moderate)',
                           3: 'DR Parah (Severe)',
                           4: 'DR Proliferatif (Proliferative)'}

            label_name = label_names.get(prediction, 'Unknown')

            # Konversi nilai 'prediction' menjadi int
            prediction = int(prediction)

            recommendation = {}

            if label_name == 'Tidak ada DR (Tidak ada Retinopati Diabetes)':
                message = ["Lakukan Kontrol Kadar Gula Darah Secara Berkala"]
                general_recommendation = ["Cara mengontrol kadar gula darah\n",
                                          "1. Konsumsi makanan yang tepat: Menurut studi dalam jurnal Education and Health Promotion, melewatkan jadwal makan terlalu lama malah akan menyebabkan gula darah turun dan kemudian melonjak cepat.\n",
                                          "2. Mengontrol porsi makan: Diabetesi dengan berat badan normal juga sebaiknya menjaga porsi makannya sehingga tidak berujung obesitas.\n",
                                          "3. Aktif bergerak dan olahraga teratur: Olahraga dapat membantu sel-sel di otot Anda mengambil lebih banyak glukosa dan mengubahnya menjadi energi, sehingga mampu menurunkan gula darah. Hindari gaya hidup sedentari (bermalas-malasan) dan minim gerakan fisik atau membuang energi, seperti menonton TV, bermain game pada gawai, atau duduk terlalu lama di depan komputer.\n",
                                          "4. Kelola stess dengan baik: Stres berlebihan juga dapat menyebabkan kadar gula darah meningkat akibat pelepasan kortisol alias hormon stres. Nah, agar stres tidak sampai membuat kadar gula darah melonjak, penting untuk Anda memahami cara mengendalikan stres dan mencoba berbagai hal yang dapat memperbaiki suasana hati, merilekskan tubuh, dan menenangkan pikiran.\n",
                                          "5. Istirahat Cukup: Tidur yang cukup dapat menyeimbangkan hormon, menghindari stres, dan membuat Anda mendapatkan cukup energi untuk beraktivitas dan berolahraga pada esok harinya.Dengan demikian, kadar gula darah pun dapat terkendali dengan baik.\n",
                                          "6. Rutin mengecek gula darah: Dengan terus memantau perubahan kadar gula darah, Anda akan lebih mudah untuk menentukan apakah harus melakukan penyesuaian pola makan atau konsumsi obat."
                                          ]
                recommendation = {
                    "message": message,
                    "general_recommendation": general_recommendation
                }
            elif label_name == 'DR Ringan (Mild)':
                message = ["Lakukan Kontrol Kadar Gula Darah Secara Berkala"]
                general_recommendation = ["Cara mengontrol kadar gula darah\n",
                                          "1. Konsumsi makanan yang tepat: Menurut studi dalam jurnal Education and Health Promotion, melewatkan jadwal makan terlalu lama malah akan menyebabkan gula darah turun dan kemudian melonjak cepat.\n",
                                          "2. Mengontrol porsi makan: Diabetesi dengan berat badan normal juga sebaiknya menjaga porsi makannya sehingga tidak berujung obesitas.\n",
                                          "3. Aktif bergerak dan olahraga teratur: Olahraga dapat membantu sel-sel di otot Anda mengambil lebih banyak glukosa dan mengubahnya menjadi energi, sehingga mampu menurunkan gula darah. Hindari gaya hidup sedentari (bermalas-malasan) dan minim gerakan fisik atau membuang energi, seperti menonton TV, bermain game pada gawai, atau duduk terlalu lama di depan komputer.\n",
                                          "4. Kelola stess dengan baik: Stres berlebihan juga dapat menyebabkan kadar gula darah meningkat akibat pelepasan kortisol alias hormon stres. Nah, agar stres tidak sampai membuat kadar gula darah melonjak, penting untuk Anda memahami cara mengendalikan stres dan mencoba berbagai hal yang dapat memperbaiki suasana hati, merilekskan tubuh, dan menenangkan pikiran.\n",
                                          "5. Istirahat Cukup: Tidur yang cukup dapat menyeimbangkan hormon, menghindari stres, dan membuat Anda mendapatkan cukup energi untuk beraktivitas dan berolahraga pada esok harinya.Dengan demikian, kadar gula darah pun dapat terkendali dengan baik.\n",
                                          "6. Rutin mengecek gula darah: Dengan terus memantau perubahan kadar gula darah, Anda akan lebih mudah untuk menentukan apakah harus melakukan penyesuaian pola makan atau konsumsi obat."
                                          ]
                recommendation = {
                    "message": message,
                    "general_recommendation": general_recommendation
                }
            elif label_name == 'DR Sedang (Moderate)':
                message = ["Lakukan konsultasi ke Dokter"]
                general_recommendation = ["Hal-hal yang mungkin dilakukan setelah melakukan konsultasi bersama Dokter\n",
                                          "1. Injeksi mata: Dokter akan menyuntikan steroid pada mata untuk menghentikan peradangan dan mencegah pembentukan pembuluh darah baru. Suntikan anti-VEGF juga mungkin disarankan, yang dapat mengurangi pembengkakan di makula dan meningkatkan penglihatan.\n",
                                          "2. Operasi Leser: Operasi laser yang disebut fotokoagulasi mengurangi pembengkakan di retina dan menghilangkan pembuluh darah abnormal.\n",
                                          "3. Vitrektomi: Jika Anda menderita retinopati diabetik stadium lanjut, Anda mungkin memerlukan vitrektomi. Operasi mata ini mengatasi masalah pada retina dan vitreous, zat seperti jeli di tengah mata. Operasi tersebut dapat menghilangkan darah atau cairan, jaringan parut, dan sebagian gel vitreous sehingga sinar cahaya dapat terfokus dengan baik pada retina.\n"]
                recommendation = {"message": message,
                                  "general_recommendation": general_recommendation}
            elif label_name == 'DR Parah (Severe)':
                message = ["Lakukan konsultasi ke Dokter"]
                general_recommendation = ["Hal-hal yang mungkin dilakukan setelah melakukan konsultasi bersama Dokter\n",
                                          "1. Injeksi mata: Dokter akan menyuntikan steroid pada mata untuk menghentikan peradangan dan mencegah pembentukan pembuluh darah baru. Suntikan anti-VEGF juga mungkin disarankan, yang dapat mengurangi pembengkakan di makula dan meningkatkan penglihatan.\n",
                                          "2. Operasi Leser: Operasi laser yang disebut fotokoagulasi mengurangi pembengkakan di retina dan menghilangkan pembuluh darah abnormal.\n",
                                          "3. Vitrektomi: Jika Anda menderita retinopati diabetik stadium lanjut, Anda mungkin memerlukan vitrektomi. Operasi mata ini mengatasi masalah pada retina dan vitreous, zat seperti jeli di tengah mata. Operasi tersebut dapat menghilangkan darah atau cairan, jaringan parut, dan sebagian gel vitreous sehingga sinar cahaya dapat terfokus dengan baik pada retina.\n"]
                recommendation = {"message": message,
                                  "general_recommendation": general_recommendation}
            elif label_name == 'DR Proliferatif (Proliferative)':
                message = ["Lakukan konsultasi ke Dokter"]
                general_recommendation = ["Hal-hal yang mungkin dilakukan setelah melakukan konsultasi bersama Dokter\n",
                                          "1. Injeksi mata: Dokter akan menyuntikan steroid pada mata untuk menghentikan peradangan dan mencegah pembentukan pembuluh darah baru. Suntikan anti-VEGF juga mungkin disarankan, yang dapat mengurangi pembengkakan di makula dan meningkatkan penglihatan.\n",
                                          "2. Operasi Leser: Operasi laser yang disebut fotokoagulasi mengurangi pembengkakan di retina dan menghilangkan pembuluh darah abnormal.\n",
                                          "3. Vitrektomi: Jika Anda menderita retinopati diabetik stadium lanjut, Anda mungkin memerlukan vitrektomi. Operasi mata ini mengatasi masalah pada retina dan vitreous, zat seperti jeli di tengah mata. Operasi tersebut dapat menghilangkan darah atau cairan, jaringan parut, dan sebagian gel vitreous sehingga sinar cahaya dapat terfokus dengan baik pada retina.\n"]
                recommendation = {"message": message,
                                  "general_recommendation": general_recommendation}

            data = {"prediction": prediction, "label": label_name,
                    "kumilcintabh": recommendation}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)
