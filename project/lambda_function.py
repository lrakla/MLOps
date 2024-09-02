import json
from hydra import compose, initialize

from prediction_service.inference import Inference
def lambda_handler(event, context):
	with initialize(config_path="configs"):
		cfg = compose(config_name="config")

		inferencing_instance = Inference(cfg)

	inference_sample = None

	if "body" in event:
		try:
			body = json.loads(event["body"])  
			inference_sample = body.get("sentence", None)
		except (json.JSONDecodeError, TypeError):
			return {
				"statusCode": 400,
				"body": json.dumps({"error": "Invalid JSON in request body"})
				}
		
	if inference_sample is None:
		inference_sample = event.get("sentence", None)

	if inference_sample is None:
		return {
			"statusCode": 400,
			"body": json.dumps({"error": "'sentence' key not found in event"})
		}
	input = {"sentence": inference_sample}
	response = inferencing_instance.predict(input)
	return {
			"statusCode": 200,
			"headers": {},
			"body": json.dumps(response.cpu().numpy().tolist())
		}

def lambda_handler_local(event, context):
	with initialize(config_path="configs"):
		cfg = compose(config_name="config")

		inferencing_instance = Inference(cfg)
        
	if "resource" in event.keys():
		body = event["body"]
		body = json.loads(body)
		inference_sample = body["sentence"]
	else:
		inference_sample = event["sentence"]

	input = {"sentence": inference_sample}
	response = inferencing_instance.predict(input)
	return {
		"statusCode": 200,
		"headers": {},
		"body": json.dumps(response.cpu().numpy().tolist())
	}

if __name__ == "__main__":
	event = {
		"sentence": "This is a test sentence."
	}
	resp = lambda_handler_local(event, None)
	print(resp)