from django.http import HttpResponse

import cv2
import numpy as np

def environment(request):
	return HttpResponse("Endpoint for environment.")
