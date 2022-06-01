from sanic import Sanic, response
from warmup import load_model
from run import run_model

# do the warmup step globally, to have a reuseable model instance
model = load_model()

app = Sanic("my_app")


@app.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    return response.json({"state": "healthy"})

@app.route('/', methods=["POST"]) # Do not edit - POST requests to "/" are a required interface
def inference(request):
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json

    # Parse out your arguments
    s3_url = model_inputs.get("s3_url")

    if s3_url == None:
        return response.json({'message': "No URI provided"})
    
    output = run_model(model, s3_url)

    return response.json(output) # Do not edit - returning a dictionary as JSON is a required interface


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000", workers=1)
