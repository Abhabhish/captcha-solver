from .files.solve_captcha import solve_captcha
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import base64



def download_base64_image(url, path):
    base64_image = url.split(",")[1]
    image_data = base64.b64decode(base64_image)
    with open(path, "wb") as image_file:
        image_file.write(image_data)


@csrf_exempt
def get_prediction(request):
    img_url = request.POST['img_url']
    download_base64_image(img_url,'captcha.jpg')
    prediction = solve_captcha('captcha.jpg')
    print(prediction)
    return JsonResponse({'prediction':prediction})

    
