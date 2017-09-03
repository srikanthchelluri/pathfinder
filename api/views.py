from django.http import HttpResponse

import numpy as np
import cv2

def environment(request):
	return HttpResponse("Endpoint for environment.")
