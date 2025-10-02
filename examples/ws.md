WebSocket RPC examples

- Connect with websocat (or wscat):

  websocat ws://<ip>/rpc

- After connecting, you should receive an initial NotifyFullStatus message.

- Send a request (JSON-RPC 2.0):

  {"id":1,"method":"EM.GetStatus","params":{"id":0}}

- Expect a response with the result object.

