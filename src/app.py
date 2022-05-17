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
    signal = model_inputs.get('signal', None)
    sr = model_inputs.get('sr', None)
    offsets = model_inputs.get('offsets', None)
    if signal == None:
        return response.json({'message': "No signal provided"})
    
    output = run_model(model, signal, sr, offsets)

    return response.json(output) # Do not edit - returning a dictionary as JSON is a required interface


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000", workers=1)
