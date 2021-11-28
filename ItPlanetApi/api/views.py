from django.http import JsonResponse
from rest_framework.decorators import api_view
from PIL import Image
import numpy as np
import cv2
import search_engine as se

global fe

@api_view(["GET"])
def welcome(request):
    content = {"message": "Welcome to the BookStore!"}
    return JsonResponse(content)

@api_view(["GET"])
def find_similar(request):
    content = {"images":find_similar_by_path(request.GET["image"])}
    return JsonResponse(content)

def find_similar_by_path(image_path):
	global fe
	# FULL_PATH = r"D:/Рабочий стол/Всякое/GitHub/IT-Planet-Part3/img/img/"
	FULL_PATH = "../img/img/"
	image_path = FULL_PATH + image_path 
	cut_img_path = "../cut_img_path"
	img = Image.open(image_path)
	image = cv2.imread(image_path)
	img_class, cutted_img_arr = se.find_class_and_cut_img(image_path, se.detection_graph)
	cutted_img = Image.fromarray(np.uint8(cutted_img_arr)).convert('RGB')
	fe = se.FeatureExtractor()
	return se.find_similar_img(cutted_img, img_class, se.cut_img_paths, se.features, fe, 9)