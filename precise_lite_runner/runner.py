import atexit
import time
from subprocess import PIPE, Popen
from threading import Thread, Event

import numpy as np
import pyaudio
import tflit
from pyaudio import PyAudio, paInt16

from precise_lite_runner.params import inject_params, pr
from precise_lite_runner.util import buffer_to_audio, ThresholdDecoder
from precise_lite_runner.vectorization import vectorize_raw, add_deltas


class TFLiteRunner:
    def __init__(self, model_name: str):
        #  Setup tflite environment
        self.model = tflit.Model(model_name)
        self.interpreter = self.model.interpreter
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def summary(self):
        self.model.summary()

    def predict(self, inputs: np.ndarray):
        # Format output to match Keras's model.predict output
        count = 0
        output_data = np.ndarray((inputs.shape[0], 1), dtype=np.float32)

        # Support for multiple inputs
        for input in inputs:
            # Format as float32. Add a wrapper dimension.
            current = np.array([input]).astype(np.float32)

            # Load data, run inference and extract output from tensor
            self.interpreter.set_tensor(self.input_details[0]['index'],
                                        current)
            self.interpreter.invoke()
            output_data[count] = self.interpreter.get_tensor(
                self.output_details[0]['index'])
            count += 1

        return output_data

    def run(self, inp: np.ndarray) -> float:
        return self.predict(inp[np.newaxis])[0][0]


class Listener:
    """Listener that preprocesses audio into MFCC vectors
     and executes neural networks"""

    def __init__(self, model_name: str, chunk_size: int = -1):
        self.window_audio = np.array([])
        self.pr = inject_params(model_name)
        self.mfccs = np.zeros((self.pr.n_features, self.pr.n_mfcc))
        self.chunk_size = chunk_size
        self.runner = TFLiteRunner(model_name)
        self.threshold_decoder = ThresholdDecoder(self.pr.threshold_config,
                                                  pr.threshold_center)

    def clear(self):
        self.window_audio = np.array([])
        self.mfccs = np.zeros((self.pr.n_features, self.pr.n_mfcc))

    def update_vectors(self, stream):
        if isinstance(stream, np.ndarray):
            buffer_audio = stream
        else:
            if isinstance(stream, (bytes, bytearray)):
                chunk = stream
            else:
                chunk = stream.read(self.chunk_size)
            if len(chunk) == 0:
                raise EOFError
            buffer_audio = buffer_to_audio(chunk)

        self.window_audio = np.concatenate((self.window_audio, buffer_audio))

        if len(self.window_audio) >= self.pr.window_samples:
            new_features = vectorize_raw(self.window_audio)
            self.window_audio = self.window_audio[
                                len(new_features) * self.pr.hop_samples:]
            if len(new_features) > len(self.mfccs):
                new_features = new_features[-len(self.mfccs):]
            self.mfccs = np.concatenate(
                (self.mfccs[len(new_features):], new_features))

        return self.mfccs

    def update(self, stream):
        mfccs = self.update_vectors(stream)
        if self.pr.use_delta:
            mfccs = add_deltas(mfccs)
        raw_output = self.runner.run(mfccs)
        return self.threshold_decoder.decode(raw_output)


class Engine:
    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size

    def start(self):
        pass

    def stop(self):
        pass

    def get_prediction(self, chunk):
        raise NotImplementedError


class PreciseEngine(Engine):
    """
    Wraps a binary precise executable

    Args:
        exe_file (Union[str, list]): Either filename or list of arguments
                                     (ie. ['python', 'precise/scripts/engine.py'])
        model_file (str): Location to .pb model file to use (with .pb.params)
        chunk_size (int): Number of *bytes* per prediction. Higher numbers
                          decrease CPU usage but increase latency
    """

    def __init__(self, exe_file, model_file, chunk_size=2048):
        Engine.__init__(self, chunk_size)
        self.exe_args = exe_file if isinstance(exe_file, list) else [exe_file]
        self.exe_args += [model_file, str(self.chunk_size)]
        self.proc = None

    def start(self):
        self.proc = Popen(self.exe_args, stdin=PIPE, stdout=PIPE)

    def stop(self):
        if self.proc:
            self.proc.kill()
            self.proc = None

    def get_prediction(self, chunk):
        if len(chunk) != self.chunk_size:
            raise ValueError('Invalid chunk size: ' + str(len(chunk)))
        self.proc.stdin.write(chunk)
        self.proc.stdin.flush()
        return float(self.proc.stdout.readline())


class ListenerEngine(Engine):
    def __init__(self, listener, chunk_size=2048):
        Engine.__init__(self, chunk_size)
        self.get_prediction = listener.update


class ReadWriteStream:
    """
    Class used to support writing binary audio data at any pace,
    optionally chopping when the buffer gets too large
    """

    def __init__(self, s=b'', chop_samples=-1):
        self.buffer = s
        self.write_event = Event()
        self.chop_samples = chop_samples

    def __len__(self):
        return len(self.buffer)

    def read(self, n=-1, timeout=None):
        if n == -1:
            n = len(self.buffer)
        if 0 < self.chop_samples < len(self.buffer):
            samples_left = len(self.buffer) % self.chop_samples
            self.buffer = self.buffer[-samples_left:]
        return_time = 1e10 if timeout is None else (
                timeout + time.time()
        )
        while len(self.buffer) < n:
            self.write_event.clear()
            if not self.write_event.wait(return_time - time.time()):
                return b''
        chunk = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return chunk

    def write(self, s):
        self.buffer += s
        self.write_event.set()

    def flush(self):
        """Makes compatible with sys.stdout"""
        pass


class TriggerDetector:
    """
    Reads predictions and detects activations
    This prevents multiple close activations from occurring when
    the predictions look like ...!!!..!!...
    """

    def __init__(self, chunk_size, sensitivity=0.5, trigger_level=3):
        self.chunk_size = chunk_size
        self.sensitivity = sensitivity
        self.trigger_level = trigger_level
        self.activation = 0

    def update(self, prob):
        # type: (float) -> bool
        """Returns whether the new prediction caused an activation"""
        chunk_activated = prob > 1.0 - self.sensitivity

        if chunk_activated or self.activation < 0:
            self.activation += 1
            has_activated = self.activation > self.trigger_level
            if has_activated or chunk_activated and self.activation < 0:
                self.activation = -(8 * 2048) // self.chunk_size

            if has_activated:
                return True
        elif self.activation > 0:
            self.activation -= 1
        return False


class PreciseRunner:
    """
    Wrapper to use Precise. Example:
    >>> def on_act():
    ...     print('Activation!')
    ...
    >>> p = PreciseRunner(PreciseEngine('./precise-engine'), on_activation=on_act)
    >>> p.start()
    >>> from time import sleep; sleep(10)
    >>> p.stop()

    Args:
        engine (Engine): Object containing info on the binary engine
        trigger_level (int): Number of chunk activations needed to trigger on_activation
                       Higher values add latency but reduce false positives
        sensitivity (float): From 0.0 to 1.0, how sensitive the network should be
        stream (BinaryIO): Binary audio stream to read 16000 Hz 1 channel int16
                           audio from. If not given, the microphone is used
        on_prediction (Callable): callback for every new prediction
        on_activation (Callable): callback for when the wake word is heard
    """

    def __init__(self, engine, trigger_level=3, sensitivity=0.5, stream=None,
                 on_prediction=lambda x: None, on_activation=lambda: None):
        self.engine = engine
        self.trigger_level = trigger_level
        self.stream = stream
        self.on_prediction = on_prediction
        self.on_activation = on_activation
        self.chunk_size = engine.chunk_size

        self.pa = None
        self.thread = None
        self.running = False
        self.is_paused = False
        self.detector = TriggerDetector(self.chunk_size, sensitivity,
                                        trigger_level)
        atexit.register(self.stop)

    def _wrap_stream_read(self, stream):
        """
        pyaudio.Stream.read takes samples as n, not bytes
        so read(n) should be read(n // sample_depth)
        """

        if getattr(stream.read, '__func__', None) is pyaudio.Stream.read:
            stream.read = lambda x: pyaudio.Stream.read(stream, x // 2, False)

    def start(self):
        """Start listening from stream"""
        if self.stream is None:
            self.pa = PyAudio()
            self.stream = self.pa.open(
                16000, 1, paInt16, True, frames_per_buffer=self.chunk_size
            )

        self._wrap_stream_read(self.stream)

        self.engine.start()
        self.running = True
        self.is_paused = False
        self.thread = Thread(target=self._handle_predictions, daemon=True)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop listening and close stream"""
        if self.thread:
            self.running = False
            if isinstance(self.stream, ReadWriteStream):
                self.stream.write(b'\0' * self.chunk_size)
            self.thread.join()
            self.thread = None

        self.engine.stop()

        if self.pa:
            self.pa.terminate()
            self.stream.stop_stream()
            self.stream = self.pa = None

    def pause(self):
        self.is_paused = True

    def play(self):
        self.is_paused = False

    def _handle_predictions(self):
        """Continuously check Precise process output"""
        while self.running:
            # t0 = time.time()
            chunk = self.stream.read(self.chunk_size)

            if self.is_paused:
                continue

            prob = self.engine.get_prediction(chunk)
            self.on_prediction(prob)
            if self.detector.update(prob):
                self.on_activation()
            # t1 = time.time()
            # print("Prediction time: %.4f" % ((t1-t0)))
