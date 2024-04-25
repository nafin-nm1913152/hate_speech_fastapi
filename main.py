from fastapi import FastAPI, Request
import uvicorn
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


app = FastAPI()

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


# ---------------------------------------------------------------------------------------


# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch

# app = FastAPI()

# class InputText(BaseModel):
#     text: str

# # Load your model
# model = BertForSequenceClassification.from_pretrained("Nafin/hate_speech_bert_extension")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# @app.post("/predict/")
# def predict(input_text: InputText):
#     # Tokenize input text
    
#     encoded_input = tokenizer(input_text.text, return_tensors="pt", padding=True, truncation=True)
    
#     # Run model prediction
#     with torch.no_grad():
#         output = model(**encoded_input)
#         logits = output.logits
    
#     # Get predicted class
#     predicted_class = torch.argmax(logits, dim=1).item()
    
#     # Define class labels
#     class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    
#     # Return predicted class
#     return {"prediction": class_labels[predicted_class]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# ---------------------------------------------------------

# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import BertTokenizer, TFBertForSequenceClassification
# import tensorflow as tf

# app = FastAPI()

# class InputText(BaseModel):
#     text: str

# # Load your model
# model = TFBertForSequenceClassification.from_pretrained("Nafin/hate_speech_bert_extension")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# @app.post("/predict/")
# def predict(input_text: InputText):
#     # Tokenize input text
#     inputs = tokenizer(input_text.text, return_tensors="tf", padding=True, truncation=True)

#     # Run model prediction
#     output = model(inputs)

#     # Get predicted class
#     predicted_class = tf.argmax(output.logits, axis=1).numpy()[0]

#     # Define class labels
#     class_labels = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}

#     # Return predicted class
#     return {"prediction": class_labels[predicted_class]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
