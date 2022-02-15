# Precise Lite Runner

use precise-lite models in your projects

```python
from precise_lite_runner import PreciseLiteListener, ReadWriteStream

stream = None # if stream is None it will open a pyaudio stream automatically
# stream = ReadWriteStream()  

chunk_size = 2048
precise_model = "/path/here.tflite"

def on_activation():
    has_found = True # do something

runner = PreciseLiteListener(model=precise_model, stream=stream, chunk_size=chunk_size,
                             trigger_level=3, sensitivity=0.5, on_activation=on_activation)

            
runner.start()

while True:
  # stream.write(chunk) # feed audio from somewhere if using ReadWriteStream
  sleep(1)
  
runner.stop()
```
