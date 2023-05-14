import time
from tornado.ioloop import PeriodicCallback
import tornado.websocket
import tornado.web
import webbrowser
import _thread
import threading
import requests
import argparse
import base64
import hashlib
import os
import time
import json
import inference

running = False
content = None

emo = None

musicPath = "test_data.txt"

pre_out = None


def processThread():
    global emo
    global pre_out
    global content
    print("processThread start")
    while True:
        if emo != None:
            localEmo = emo
            emo = None
            res, pre_out = inference.inference(
                inference.median, [localEmo for _ in range(16)], musicPath, pre_out)
            content = inference.buildPlaySeq(res)
        time.sleep(0.1)


class WebSocket(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request, **kwargs):
        global running
        if running:
            self.close()
        running = True
        self.looper = None
        print("running", running)
        super().__init__(application, request, **kwargs)

    def on_message(self, message):
        global emo
        global pre_out
        if self.looper == None:
            pre_out = None
            self.looper = PeriodicCallback(self.loop, 10)
            self.looper.start()

        try:
            dataPair = message.split(":")
            if dataPair == "emo" and len(dataPair) >= 2:
                estr = dataPair[1].split(",")
                if len(estr) >= 2:
                    emo = (estr[0], estr[1])
        except Exception as err:
            print(err)

    def check_origin(self, origin):
        return True

    def on_close(self):
        global running
        running = False
        print("running", running)

    def loop(self):
        global content
        try:
            if content != None:
                self.write_message(content)
                content = None
        except tornado.websocket.WebSocketClosedError:
            self.looper.stop()


_thread.start_new_thread(processThread, ())
handlers = [(r"/websocket", WebSocket)]
application = tornado.web.Application(handlers)
application.listen(8208)
tornado.ioloop.IOLoop.instance().start()
