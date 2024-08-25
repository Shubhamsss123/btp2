from django.shortcuts import render,HttpResponse,redirect
import os
import shutil
# Create your views here.
from .models import ImgModel,UserModel
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
def img_view(request):
    if request.method == "POST":
        # Handle the image upload
        folder_path = os.path.join(settings.MEDIA_ROOT, "dams")
        if 'image' in request.FILES:

            
        
        # Check if the folder already exists
            if os.path.exists(folder_path):
                # Delete the folder and all its contents
                shutil.rmtree(folder_path)
            # img = ImgModel(image=request.FILES['image'])
            uploaded_file = request.FILES['image']
            file_name = "dam.jpg"  # Rename the file to dam.jpg

            # Construct the full path where the file will be saved
            save_path = os.path.join("dams", file_name)

            # Save the file to the specified path with the new name
            saved_file = default_storage.save(save_path, ContentFile(uploaded_file.read()))

            # Save the file information to the database
            img = ImgModel(image=saved_file)
            img.save()
        
    user = UserModel.objects.first()  # Fetch the first user instance (or None if empty)

    if user:
        context = {
            "name": user.name,
            "position": user.position,
        }
    else:
        context = {
            "name": "N/A",
            "position": "N/A",
        }

    return render(request, 'home.html', context)
def user_view(request):

    if request.method == "POST":
        name = request.POST.get('name')
        position = request.POST.get('position')
        
        # Check if there's already one row in the database
        if UserModel.objects.exists():
            # If there's an existing row, delete it
            UserModel.objects.all().delete()
        
        # Create a new row with the submitted data
        user = UserModel.objects.create(name=name, position=position)
        user.save()
        # Redirect to home with the data
    user = UserModel.objects.first()  # Fetch the first user instance (or None if empty)

    if user:
        context = {
            "name": user.name,
            "position": user.position,
        }
    else:
        context = {
            "name": "N/A",
            "position": "N/A",
        }

    return render(request, 'home.html', context)
    