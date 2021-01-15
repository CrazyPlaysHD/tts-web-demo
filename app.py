from flask import Flask,render_template, Response, request
import sys
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from tornado.ioloop import IOLoop
from text2speech import T2S
import os
from time import time

model = 'tacotron2'
vocoder = 'waveglow'
t2s = T2S(model, vocoder)
sample_text = {
    'vi' : 'nhập vào một văn bản',
}


# Initialize Flask.
app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def texttospeech():
    if request.method == 'POST':
        result = request.form

        dict_input = {}

        dict_input['vocoder'] = result['input_vocoder']
        dict_input['model'] = result['input_model']
        dict_input['text'] = result['input_text']
        dict_input['d'] = result['d']
        dict_input['p'] = result['p']
        dict_input['e'] = result['e']
        dict_input['sig'] = result['sig']
        dict_input['strength'] = result['strength']
        print('d', dict_input['d'])
        print('p', dict_input['p'])
        print('e', dict_input['e'])
        print('s', dict_input['sig'])
        print('s', dict_input['strength'])

        t2s.model = dict_input['model']
        t2s.vocoder = dict_input['vocoder']

        audio, t0 = t2s.tts(dict_input)

        return render_template('simple.html', voice=audio, sample_text=sample_text.get('vi'), model=t2s.model, vocoder=t2s.vocoder, time=t0)

#Route to render GUI
@app.route('/')
def show_entries():
    return render_template('simple.html', sample_text=sample_text.get('vi'), voice=None, model=t2s.model, vocoder=t2s.vocoder)

#Route to stream music
@app.route('/<voice>', methods=['GET'])
def streammp3(voice):
    
    def generate():    
        with open(os.path.join('./', 'wavs', voice), "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
            
    return Response(generate(), mimetype="audio/mp3")


#launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 8888
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()
