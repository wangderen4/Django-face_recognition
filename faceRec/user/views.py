import base64
import datetime
import random
import time

import os
import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from aip import AipFace
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render
from keras.models import load_model

from static import chineseText
from .models import User, UploadImage

# from user.models import User
# from django.http import HttpResponse
# Create your views here.

# 主页面视图函数
def main(request):
    uname = request.session['uname']
    user = User.objects.get(u_name=uname)

    return render(request, 'index.html', {'user': user})


# 注册功能
def register(request):
    if request.method == 'GET':
        return render(request, 'regist.html')
    if request.method == 'POST':
        # 注册
        print('正在运行！！！！！！')
        uname = request.POST.get('username')
        if User.objects.filter(u_name=uname) is not None:
            password = request.POST.get('password')
            # 对密码进行加密
            password = make_password(password)
            # 用户图像数据传过来了~
            uicon = request.POST.get('face')
            type(uicon)
            print('用户注册头像数据》》》》》', uicon)
            print('uicon的数据格式>>>', type(uicon))
            # 删除编码前的标记
            base64str = uicon[22:]
            print(base64str[0:15])
            print('base64str>>>', base64str)
            if len(base64str) % 3 == 1:
                base64str += "=="
            elif len(base64str) % 3 == 2:
                base64str += "="
            # 解码
            img_str = base64.b64decode(base64str)
            # print('imgstr>>>>>', imgstr)
            url = u"C:/Users/wangderen4/Desktop/faceRec/static/media/"
            # 图片命名
            img_name = uname + '_icon_' + str(int(time.time())) + '.jpg'
            print('当前图片名称img_name是》》》》》', img_name)
            url = url + 'user_icon/' + img_name
            print('当前url是》》》》》', url)
            with open(url, 'wb') as f:
                f.write(img_str)
            uicon = 'user_icon/' + img_name

            User.objects.create(u_name=uname, u_password=password, u_icon=uicon)
            return HttpResponseRedirect('/')
        return render(request, 'regist.html', {'name': uname})


# 登录功能
def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
        # pass
    if request.method == 'POST':
        # 如果登录成功，绑定参数到cookie中，set_cookie
        name = request.POST.get('username')
        password = request.POST.get('password')

        # 查询用户是否在数据库中
        if User.objects.filter(u_name=name).exists():
            user = User.objects.get(u_name=name)
            print(user)
            if check_password(password, user.u_password):
                request.session['uname'] = user.u_name
                # ticket = 'agdoajbfjad'
                ticket = ''
                for i in range(15):
                    s = 'abcdefghijklmnopqrstuvwxyz'
                    # 获取随机的字符串
                    ticket += random.choice(s)
                now_time = int(time.time())
                ticket = 'TK' + ticket + str(now_time)
                # 绑定令牌到cookie里面
                response = HttpResponseRedirect('/main/')
                # max_age 存活时间(秒)
                response.set_cookie('ticket', ticket, max_age=10000)
                # 存在服务端
                user.u_ticket = ticket
                user.save()  # 保存
                return response
            else:
                return render(request, 'login.html', {'password': '用户密码错误'})
        else:
            # return HttpResponse('用户不存在')
            return render(request, 'login.html', {'name': '用户不存在'})


# 人脸登录
def face_login(request):
    if request.method == "GET":
        return render(request, 'face_login.html')
    if request.method == 'POST':
        # 用户图像数据传过来了~
        face_img = request.POST.get('face')
        # 删除base64的编码信息，提取base64编码数据
        base64str = face_img[22:]
        print('base64str>>>', base64str)
        # 对base64数据进行补全，防止数据不能够被编码成图片
        if len(base64str) % 3 == 1:
            base64str += "=="
        elif len(base64str) % 3 == 2:
            base64str += "="
        # 对base64数据进行解码
        img_str = base64.b64decode(base64str)
        # 准备把数据写进图片，这个是保存图片的基本路径
        static_url = u"C:/Users/wangderen4/Desktop/faceRec/static/media/"
        # 图片命名
        img_name = 'face_' + str(int(time.time())) + '.jpg'
        print('当前img_name是》》》》》', img_name)
        # 图片保存最终的位置及名称
        unknown_url = static_url + 'login_img/' + img_name
        print('当前unknown_url是》》》》》', unknown_url)
        # 把base64数据写入图片
        with open(unknown_url, 'wb') as f:
            f.write(img_str)
        # 让程序睡一会，来确保能加载传过来的图片~
        time.sleep(0.5)
        # 创建空字典用于存放{user：face_encode}
        face_dict = {}
        print('face_dict的字典》》》', face_dict)
        # 遍历user_icon下面所有用户头像，用来比对
        for known_user_path in os.listdir(
                'C:/Users/wangderen4/Desktop/faceRec/static/media/user_icon'):
            # 加载图片
            print('known_user_path>>>', known_user_path)
            load_img = face_recognition.load_image_file(static_url +'user_icon/'+ known_user_path)
            # 获取人脸数据,对人脸进行编码
            face_encode = face_recognition.face_encodings(load_img)
            # 截取图片名（这里应该把images文件中的图片名命名为为人物名）
            user_name = known_user_path.split('_')[0]
            face_dict[user_name] = face_encode
        print('face_dict的类型》》》', type(face_dict),'face_dict》》》', face_dict)
        try:
            load_unknown_img = face_recognition.load_image_file(unknown_url)
            unknown_img_encoding = face_recognition.face_encodings(load_unknown_img)[0]
            for i in face_dict:
                results = face_recognition.compare_faces(face_dict[i], unknown_img_encoding, tolerance=0.4)
                print('类型》》》',type(results[0]),'值》》》',results[0])
                # if 'True' in str(results[0]):
                # print('len(results)>>>', (results[0]))
                print('现在的str(results[0])是》》》', str(results[0]))
                if str(results[0]) == 'True':
                    user = User.objects.get(u_name=i)
                    print('当前用户为》》》',user)
                    request.session['uname'] = user.u_name
                    # ticket = 'agdoajbfjad'
                    ticket = ''
                    for i in range(15):
                        s = 'abcdefghijklmnopqrstuvwxyz'
                        # 获取随机的字符串
                        ticket += random.choice(s)
                    now_time = int(time.time())
                    ticket = 'TK' + ticket + str(now_time)
                    # 绑定令牌到cookie里面
                    response = HttpResponseRedirect('/main/')
                    # max_age 存活时间(秒)
                    response.set_cookie('ticket', ticket, max_age=10000)
                    # 存在服务端
                    user.u_ticket = ticket
                    user.save()  # 保存
                    return response
                else:
                    continue
            else:
                title = '用户不存在，请进行注册'
                msg = '用户不存在，请进行注册'
                return render(request, 'redirect.html', {'msg': msg, 'title': title})
        except Exception as e:
            print(e)
            title = '用户不存在，请进行注册'
            msg = '用户不存在，请进行注册'
            return render(request, 'redirect.html', {'msg': msg, 'title': title})


# def index(request):
#     if request.method == 'GET':
#         # 获取所有学生信息
#         ticket = request.COOKIES.get('ticket')
#         if not ticket:
#             return HttpResponseRedirect('/uauth/login/')
#         if User.objects.filter(u_ticket=ticket).exists():
#             stuinfos = StudentInfo.objects.all()
#             return render(request, 'index.html', {'stuinfos': stuinfos})
#         else:
#             return HttpResponseRedirect('/uauth/login/')

# 注销
def logout(request):
    if request.method == 'GET':
        # 删除用户session
        # del request.session['uname']
        response = HttpResponseRedirect('/')
        # response.delete_cookie('ticket')
        # 返回响应体

        return response


# 用户基本信息完善
def user_base_info(request):
    uname = request.session['uname']
    if request.method == 'GET':
        return render(request, 'user_base_info.html', {'uname': uname})
    if request.method == 'POST':
        user = User.objects.get(u_name=uname)
        new_user_icon: object = request.FILES.get('img')
        print('图片名》》》', user.u_icon.name.split('.'))
        sign = request.POST.get('sign')
        email = request.POST.get('email')
        # 构造文件名以及文件路径
        new_user_icon.name = uname + '_icon_' + str(int(time.time())) + '.' + new_user_icon.name.split('.')[
            -1]
        print('tuaixng>>>>>', new_user_icon.name)
        if new_user_icon.name.split('.')[-1] not in ['jpeg', 'jpg', 'png']:
            return HttpResponse('输入文件有误')
        try:
            user.u_icon = new_user_icon
            print('图象>>>>>', user.u_icon)
            user.u_email = email
            user.u_sign = sign
            user.save()
            url = u"C:/Users/wangderen4/Desktop/faceRec/static/media/"
            url = url + 'user_icon/' + new_user_icon.name
            print('图象url>>>>>', url)
            with open(url, 'wb') as f:
                # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，
                # 加载到内存中，并将这一部分内容写入到目录下，
                # 写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。
                # 就是为了节省内存空间。
                for data in new_user_icon.chunks():
                    f.write(data)
        except Exception as e:
            print(e)
        return render(request, 'user_base_info.html', {'uname': uname})


# 性别识别功能
def sex_reg(request):
    uname = request.session['uname']
    if request.method == 'POST':
        user = User.objects.get(u_name=uname)
        # 前端获取的图片
        upload_img = request.FILES.get('img')
        # 构造文件名和文件路径
        upload_img.name = uname + '_photo_' + str(int(time.time())) + '.' + upload_img.name.split('.')[-1]
        if upload_img.name.split('.')[-1] not in ['jpeg', 'jpg', 'png']:
            return HttpResponse('<h1 style="color:red;">输入文件有误</h1>')
        # 基础路径
        base_url = u"C:/Users/wangderen4/Desktop/faceRec/static/media/"
        # 对上传的图片进行命名
        img_name = uname + '_photo_' + str(int(time.time())) + '.jpg'
        print('当前name是》》》》》', img_name)
        upload_url = base_url + 'user_photo/' + img_name
        print('当前url是》》》》》', upload_url)
        with open(upload_url, 'wb') as f:
            # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，
            # 加载到内存中，并将这一部分内容写入到目录下，
            # 写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。
            # 就是为了节省内存空间。
            for data in upload_img.chunks():
                f.write(data)
        sex_img = cv2.imread(upload_url)
        print('img是>>>>>', sex_img)
        face_classifier = cv2.CascadeClassifier(
            'C:/Users/wangderen4/Desktop/faceRec/static/haarcascade_frontalface_default.xml'
        )
        # gray：转换的灰图
        gray = cv2.cvtColor(sex_img, cv2.COLOR_BGR2GRAY)
        # scaleFactor：图像缩放比例，可理解为相机的X倍镜
        # minNeighbors：对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏
        # minSize：特征检测点的最小尺寸
        faces = face_classifier.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(140, 140))
        gender_classifier = load_model(
            "C:/Users/wangderen4/Desktop/faceRec/static/classifier/gender_models/simple_CNN.81-0.96.hdf5")
        print('执行到这里啦~~~~~')
        gender_labels = {0: '女', 1: '男'}
        # print('gender_labels>>>>>>>>>>>', gender_labels)
        color = (255, 255, 255)
        for (x, y, w, h) in faces:
            face = sex_img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, 0)
            face = face / 255.0
            gender_label_arg = np.argmax(gender_classifier.predict(face))
            gender = gender_labels[gender_label_arg]
            cv2.rectangle(sex_img, (x, y), (x + h, y + w), color, 2)
            sex_img = chineseText.cv2ImgAddText(sex_img, gender, x + h, y, color, 30)

        sex_reg_done = '/static/media/' + 'img_done/' + uname + str(int(time.time())) + '.jpg'
        print('sex_reg_done的图片路径>>>>>>', sex_reg_done)
        sex_reg_done_url = 'C:/Users/wangderen4/Desktop/faceRec/' + sex_reg_done
        # cv2.imshow('img',sex_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(sex_reg_done_url, sex_img)
        print('图片写入成功》》》位置是', sex_reg_done_url)
        cv2.imwrite(sex_reg_done_url, sex_img)
        return render(request, 'index.html', {'user': user, 'img': sex_reg_done})


# 人脸定位功能
def face_detect(request):
    # 对图片预处理区
    uname = request.session['uname']
    if request.method == 'POST':
        user = User.objects.get(u_name=uname)
        upload_img = request.FILES.get('img')
        # 构造文件名和文件路径
        upload_img.name = uname + '_photo_' + str(int(time.time())) + '.' + upload_img.name.split('.')[-1]
        if upload_img.name.split('.')[-1] not in ['jpeg', 'jpg', 'png']:
            return HttpResponse('<h1 style="color:red;">输入文件有误</h1>')
        try:
            url = u"C:/Users/wangderen4/Desktop/faceRec/static/media/"
            img_url = url + 'user_photo/' + upload_img.name
            with open(img_url, 'wb') as f:
                # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，
                # 加载到内存中，并将这一部分内容写入到目录下，写完以后，内存清空；
                # 下一次再从本地磁盘读取一部分数据放入内存。就是为了节省内存空间。
                for data in upload_img.chunks():
                    f.write(data)
        except Exception as e:
            print(e)

        # 人脸识别开始
        # 上传的图片保存的位置和图片名
        filepath = img_url
        # OpenCV人脸识别分类器
        classifier = cv2.CascadeClassifier(
            "C:/Users/wangderen4/Desktop/faceRec/static/haarcascade_frontalface_default.xml"
        )
        # 程序开始时间
        startTime = datetime.datetime.now()
        # 读取图片
        img = cv2.imread(filepath)
        # 转换灰色
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 定义绘制颜色
        color = (0, 255, 0)
        # 调用识别人脸
        faceRects = classifier.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        # 大于0则检测到人脸
        if len(faceRects):
            # 单独框出每一张人脸
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 框出人脸
                cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
                # 左眼
                cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                           color)
                # 右眼
                cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                           color)
                # 嘴巴
                cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                              (x + 5 * w // 8, y + 7 * h // 8), color)

    # 程序结束时间
    endTime = datetime.datetime.now()
    totalTime = endTime - startTime
    print(totalTime)
    img_detect_done = '/static/media/' + 'img_done/' + uname + str(int(time.time())) + '.jpg'
    # 处理后保存图片的地址
    img_detect_done_url = 'C:/Users/wangderen4/Desktop/faceRec' + img_detect_done
    print('处理后保存图片的地址是》》》', img_detect_done_url)

    cv2.imwrite(img_detect_done_url, img)
    return render(request, 'index.html', {'user': user, 'img': img_detect_done})


# 表情识别功能
def phiz_reg(request):
    uname = request.session['uname']
    if request.method == 'POST':
        user = User.objects.get(u_name=uname)
        upload_img = request.FILES.get('img')
        # 构造文件名和文件路径
        upload_img.name = uname + '_photo_' + str(int(time.time())) + '.' + upload_img.name.split('.')[-1]
        if upload_img.name.split('.')[-1] not in ['jpeg', 'jpg', 'png']:
            return HttpResponse('输入文件有误')
        try:
            url = u"C:/Users/wangderen4/Desktop/faceRec/static/media/"
            img_url = url + 'user_photo/' + upload_img.name
            print('img_url是》》》》》》', img_url)
            with open(img_url, 'wb') as f:
                # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，
                # 加载到内存中，并将这一部分内容写入到目录下，
                # 写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。
                # 就是为了节省内存空间。
                for data in upload_img.chunks():
                    f.write(data)
        except Exception as e:
            print(e)
        # 上传的图片保存的位置和图片名
        filepath = img_url
        startTime = datetime.datetime.now()
        emotion_classifier = load_model(
            'C:/Users/wangderen4/Desktop/faceRec/static/classifier/emotion_models/simple_CNN.530-0.65.hdf5')
        endTime = datetime.datetime.now()
        print('表情训练模型加载时间》》》》', endTime - startTime)

        emotion_labels = {
            0: '生气',
            1: '厌恶',
            2: '恐惧',
            3: '开心',
            4: '难过',
            5: '惊喜',
            6: '平静'
        }

        img = cv2.imread(filepath)
        face_classifier = cv2.CascadeClassifier(
            "C:/Users/wangderen4/Desktop/faceRec/static/haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))
        color = (255, 0, 0)

        for (x, y, w, h) in faces:
            gray_face = gray[(y):(y + h), (x):(x + w)]
            gray_face = cv2.resize(gray_face, (48, 48))
            gray_face = gray_face / 255.0
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
            emotion = emotion_labels[emotion_label_arg]
            cv2.rectangle(img, (x + 10, y + 10), (x + h - 10, y + w - 10),
                          (255, 255, 255), 2)
            img = chineseText.cv2ImgAddText(img, emotion, x + h * 0.3, y, color, 20)

        phiz_reg_done = '/static/media/' + 'img_done/' + uname + str(int(time.time())) + '.jpg'
        print('phiz_reg_done图片路径>>>>>>', phiz_reg_done)
        # 处理后保存图片的地址
        phiz_reg_done_url = 'C:/Users/wangderen4/Desktop/faceRec' + phiz_reg_done
        cv2.imwrite(phiz_reg_done_url, img)
        return render(request, 'index.html', {'user': user, 'img': phiz_reg_done})


# 面部化妆
def face_draw(request):
    uname = request.session['uname']
    if request.method == 'POST':
        user = User.objects.get(u_name=uname)
        img = request.FILES.get('img')
        # 构造文件名和文件路径
        img.name = uname + '_photo_' + str(int(time.time())) + '.' + img.name.split('.')[-1]
        if img.name.split('.')[-1] not in ['jpeg', 'jpg', 'png']:
            msg = '输入文件有误'
            return render(request, 'err.html', {'msg': msg})
        try:
            url = u"C:/Users/wangderen4/Desktop/faceRec/static/media/"
            img_url = url + 'user_photo/' + img.name

            with open(img_url, 'wb') as f:
                # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，
                # 加载到内存中，并将这一部分内容写入到目录下，
                # 写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。
                # 就是为了节省内存空间。
                for data in img.chunks():
                    f.write(data)
        except Exception as e:
            print(e)
            # 上传的图片保存的位置和图片名
        filepath = img_url
        # 将图片文件文件加载到numpy数组中
        image = face_recognition.load_image_file(filepath)

        # Find all facial features in all the faces in the image
        # 查找图像中所有面部的所有面部特征
        face_landmarks_list = face_recognition.face_landmarks(image)
        for face_landmarks in face_landmarks_list:
            pil_image = Image.fromarray(image)
            d = ImageDraw.Draw(pil_image, 'RGBA')

            # Make the eyebrows into a nightmare
            # 让眉毛成为一场噩梦
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

            # Gloss the lips
            # 唇彩上妆
            d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
            d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
            d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=6)
            d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=6)

            # Sparkle the eyes
            # 眼睛上妆
            d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

            # Apply some eyeliner
            # 眼线上妆
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

            print('这个图像》》》》', pil_image)

            face_draw_done = '/static/media/' + 'img_done/' + uname + str(int(time.time())) + '.jpg'
            face_draw_done_url = url + 'img_done/' + uname + str(int(time.time())) + '.jpg'  # 处理后保存图片的地址
            pil_image.save(face_draw_done_url)
        return render(request, 'index.html', {'user': user, 'img': face_draw_done})
