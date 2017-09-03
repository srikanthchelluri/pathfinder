from django.http import HttpResponse

def environment(request):
	return HttpResponse("Endpoint for environment.")
