# -*- coding: utf-8 -*-
# @Time    : 2019/5/3 9:14
# @Author  : wangderen4
# @File    : urls.py
# @Project: face_rec


from django.conf.urls import url
from . import views

urlpatterns = [
    #注册路由
    url(r'^register/', views.register),
    #登录
    url(r'^$', views.login),
    #人脸登录
    url(r'^face_login/', views.face_login),
    #注销退出
    url(r'^logout/', views.logout),
    #主功能界面
    url(r'main/', views.main),
    #用户基本信息
    url(r'baseinfo/', views.user_base_info),
    #性别识别
    url(r'sex_reg/', views.sex_reg),
    #人脸检测
    url(r'face_detect', views.face_detect),
    #面部识别
    url(r'phiz_reg', views.phiz_reg),
    #面部化妆
    url(r'face_draw/', views.face_draw),
]
