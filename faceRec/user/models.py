from django.db import models

# Create your models here.


class User(models.Model):
    u_id = models.AutoField(primary_key=True)
    u_name = models.CharField(max_length=18, unique=True)#用户名
    u_icon = models.ImageField(upload_to='user_icon/')#用户头像
    u_password = models.CharField(max_length=254)#用户密码
    u_sign = models.TextField()#用户签名
    u_email = models.EmailField()#用户邮箱
    u_ticket = models.CharField(max_length=30, null=True)#登录ticket

    def __str__(self):
        return '用户名：%s' % self.u_name


class UploadImage(models.Model):
    image_h = models.PositiveIntegerField(default=200)
    image_w = models.PositiveIntegerField(default=200)
    img_path = models.ImageField(width_field=image_w, height_field=image_h,upload_to='user_photo/')#用户上传图片
    user = models.ForeignKey('User', on_delete=models.CASCADE)


class DoneImage(models.Model):
    image_h = models.PositiveIntegerField(default=200)
    image_w = models.PositiveIntegerField(default=200)
    done_time = models.TimeField(auto_now_add=True)
    image_done = models.ImageField(width_field=image_w, height_field=image_h,upload_to='img_done/')
    upload_image = models.ForeignKey('UploadImage', on_delete=models.CASCADE)






