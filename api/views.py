from django.http import HttpResponse
from . import process

def environment(request):
	# Check if request is POST

	# Collect and uniquely name the file

	# Process the file (run CV algorithm) and collect matrix

	# Find path through matrix

	# Return representation of the path

	return HttpResponse("Endpoint for environment.")