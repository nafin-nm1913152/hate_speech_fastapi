from fastapi import FastAPI, Request
import uvicorn
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


app = FastAPI()

@app.get("/")
def read_root():
    return {"text": "This hate speech model api point"}

def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertForSequenceClassification.from_pretrained("Nafin/hate_speech")
    return tokenizer,model

d = {
    
  0: 'Hate Speech', 
  1: 'Offensive Language', 
  2: 'Neither'
}

tokenizer,model = get_model()

@app.post("/predict")
async def read_root(request: Request):
    data = await request.json()
    print(data)
    if 'text' in data:
        user_input = data['text']
        print("User input:", user_input)
        print("type:", type(user_input))

        tokenized_input = tokenizer([user_input], return_tensors="tf", padding=True, truncation=True)
        print("Tokenized input:", tokenized_input)

        logits = model(tokenized_input)["logits"]
        print("Logits:", logits)

        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        print("Probabilities:", probabilities)

        predicted_class = tf.argmax(probabilities).numpy()
        print("Predicted class:", predicted_class)

        class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        print("Class labels:", class_labels)

        predicted_label = class_labels[predicted_class]
        print("Predicted label:", predicted_label)

        response = {"Received Text": user_input, "Prediction": predicted_label}
    else:
        response = {"Recieved Text": "No Text Found"}
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
