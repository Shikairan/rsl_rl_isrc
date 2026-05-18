import json
import zmq
ctx = zmq.Context()
req = ctx.socket(zmq.REQ)
req.connect("tcp://localhost:15556")  # CTRL_REP 端口
# 只监控 env 3
req.send_json({"state": [0, 0, 3, 14]})
print(req.recv_json())  # 期望 {"ok": true}
