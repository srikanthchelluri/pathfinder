from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from . import process

IMGUR_BASE = "http://i.imgur.com/"

def environment(request):
	# Check if request is POST
	if request.method == "GET":
		return render(request, 'form.html')

	# Collect hash of the image
	image_hash = request.POST.get('hash')

	# Process the file (run CV algorithm) and collect matrix
	matrix = process.determine_obstacle_matrix(IMGUR_BASE + image_hash + ".png")
	# Find path through matrix
	path = process.find_path(matrix)
	# Return representation of the path
	vector = process.create_vector(path)

	return JsonResponse(vector, safe=False)