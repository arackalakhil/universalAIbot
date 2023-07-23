import random
import json
import torch
from model import NeuralNet
from nltk_utilis import bag_of_words,tokanize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('train.json','r') as f:
    intents=json.load(f)
FILE="data.pth"
data=torch.load(FILE)
input_size=data["input_size"]
hidden_size=data["hidden_size"]
output_size=data["output_size"]
all_words=data["all_words"]
tags=data["tags"]
model_state=data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()


bot_name="AI"
print("first question type 'quit' to exit")
while True:
    sentence= input("you: ")
    if sentence =="quit":
        break
    sentence=tokanize(sentence)
    x=bag_of_words(sentence,all_words)
    x=x.reshape(1,x.shape[0])
    x=torch.from_numpy(x)  
    output=model(x)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]
    if prob.item()>0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}:{random.choice(intent['responses'])}")
    else :
        print(f"{bot_name}:not under stand")
