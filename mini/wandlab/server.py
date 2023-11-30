# -*- encoding: utf-8 -*-
#-------------------------------------------------#
# Date created          : 2020. 8. 18.
# Date last modified    : 2020. 8. 19.
# Author                : chamadams@gmail.com
# Site                  : http://wandlab.com
# License               : GNU General Public License(GPL) 2.0
# Version               : 0.1.0
# Python Version        : 3.6+
#-------------------------------------------------#

from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool

from wandlab import Streamer



app = FastAPI()
streamer = Streamer()

@app.get('/stream')
async def stream(src: int = 0):
    return StreamingResponse(stream_gen(src), media_type="multipart/x-mixed-replace; boundary=frame")

async def stream_gen(src):
    try:
        await run_in_threadpool(streamer.run, src)

        while True:
            frame = streamer.bytescode()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except GeneratorExit:
        # print('[wandlab]', 'disconnected stream')
        await run_in_threadpool(streamer.stop)