from transformers import pipeline, Conversation
from flask import Flask, render_template, request
from torch import device, cuda


def chatbot_response(msg):
    conv.add_user_input(msg)
    response = pipe(conv)
    return response


dev = device("cuda:0") if cuda.is_available() else device("cpu")
pipe = pipeline(model="facebook/blenderbot-400M-distill",
                device=dev)
conv = Conversation()

app = Flask(__name__)
app.static_folder = "static"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText).generated_responses[-1]


if __name__ == "__main__":
    app.run()
