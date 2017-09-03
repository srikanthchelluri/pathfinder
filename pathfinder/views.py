from django.http import HttpResponse

def index(request):
	return HttpResponse("This is not a client-side application. Please use the endpoints.")
