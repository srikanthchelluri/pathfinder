from django.http import HttpResponse
from . import process

def environment(request):
	# Check if request is POST

	# Collect and uniquely name the file

	# Process the file (run CV algorithm) and collect matrix
	matrix = process.determine_obstacle_matrix("https://i.imgur.com/swpvW95.png") # TODO: must change
	# Find path through matrix
	path = process.find_path(matrix)
	# Return representation of the path
	vector = process.create_vector(path)

	return HttpResponse(vector)