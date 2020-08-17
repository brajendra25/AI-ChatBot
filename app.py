from flask import Flask, render_template,request
import json 
import urllib.request
import BusinessLayer.chatbotBL as CBL


app = Flask(__name__)

colors = [
    "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
    "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
    "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

'''Chat Bot '''
@app.route('/')
def index():
    return render_template('index.html', title="Chat-Bot ")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(CBL.chat_bow(userText.lower()))
'''End'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
