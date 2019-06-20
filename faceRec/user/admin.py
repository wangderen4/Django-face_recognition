from django.contrib import admin
from .models import User
# Register your models here.


class UserAdmin(admin.ModelAdmin):
    list_display = ['pk', 'u_name', 'u_icon', 'u_password', 'u_sign', 'u_email']
    list_filter = ['u_name']
    list_per_page = 5