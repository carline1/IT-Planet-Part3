from django.http import JsonResponse
from rest_framework.decorators import api_view
import search_engine

@api_view(["GET"])
def welcome(request):
    content = {"message": "Welcome to the BookStore!"}
    return JsonResponse(content)

@api_view(["GET"])
def find_similar(request):
    content = {"images":find_similar_by_path(request.get("imageName"))}
    return JsonResponse(content)

def find_similar_by_path(image_path):
    image_path = r"D:\Projects\Python_Projects\IT-Planet-Part3\image" + image_path + ".jpg"
    img = Image.open(image_path)
    img_class, cutted_img_arr = find_class_and_cut_img(image_path, detection_graph)
    cutted_img = Image.fromarray(np.uint8(cutted_img_arr)).convert('RGB')
    return find_similar_img(cutted_img, img_class, cut_img_paths, features, fe, 9)