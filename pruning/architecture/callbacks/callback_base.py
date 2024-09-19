from typing import Iterable


class Callback:
    def on_fit_start(self):
        pass

    def on_fit_end(self):
        pass

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    def on_training_step_start(self):
        pass

    def on_training_step_end(self, outputs):
        pass

    def on_validation_step_start(self):
        pass

    def on_validation_step_end(self, outputs):
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: Iterable[Callback]):
        self.callbacks = callbacks

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def remove(self, callback: Callback):
        self.callbacks.remove(callback)

    def on_fit_start(self):
        for callback in self.callbacks:
            callback.on_fit_start()

    def on_fit_end(self):
        for callback in self.callbacks:
            callback.on_fit_end()

    def on_validation_start(self):
        for callback in self.callbacks:
            callback.on_validation_start()

    def on_validation_end(self):
        for callback in self.callbacks:
            callback.on_validation_end()

    def on_training_step_start(self):
        for callback in self.callbacks:
            callback.on_training_step_start()

    def on_training_step_end(self, outputs):
        for callback in self.callbacks:
            callback.on_training_step_end(outputs)

    def on_validation_step_start(self):
        for callback in self.callbacks:
            callback.on_validation_step_start()

    def on_validation_step_end(self, outputs):
        for callback in self.callbacks:
            callback.on_validation_step_end(outputs)
