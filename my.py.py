import cv2
import asyncio
import websockets
import json

async def send_video_frames(websocket, path):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to JSON and send it to the client
        data = json.dumps({
            'data': frame.tobytes(),
            'width': frame.shape[1],
            'height': frame.shape[0]
        })
        await websocket.send(data)

start_server = websockets.serve(send_video_frames, 'localhost', 7288)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
