import tensorflow as tf
from tensorflow import keras
K = keras.backend

class OneCycleScheduler(keras.callbacks.Callback):
  def __init__(self, *, total_iterations, 
               min_lr, max_lr, 
               min_momentum, max_momentum,
               annealing_phase, annealing_rate):

    self.total_iterations = total_iterations
    self.min_lr = min_lr
    self.max_lr = max_lr
    self.min_momentum = min_momentum
    self.max_momentum = max_momentum
    self.annealing_phase = annealing_phase
    self.annealing_rate = annealing_rate
    self.min_annealing_lr = min_lr * annealing_rate
    self.lr_peak = int(total_iterations * (1 - annealing_phase) / 2)
    self.iterations = 0
    self.history = {}
  
  def on_train_begin(self, logs=None):
    logs = logs or {}
    K.set_value(self.model.optimizer.lr, self.min_lr)
    K.set_value(self.model.optimizer.momentum, self.max_momentum)
  
  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    self.iterations += 1

    self.history.setdefault("lr", []).append(K.get_value(self.model.optimizer.lr))
    self.history.setdefault("momentum", []).append(K.get_value(self.model.optimizer.momentum))
    self.history.setdefault("iterations", []).append(K.get_value(self.iterations))

    for k, v in logs.items():
      self.history.setdefault(k, []).append(v)

    K.set_value(self.model.optimizer.lr, self._calculate_lr())
    K.set_value(self.model.optimizer.momentum, self._calculate_lr())

  def _calculate_lr(self):
    if self.iterations < self.lr_peak:
      m = self.max_lr - self.min_lr
      x = self.iterations / self.lr_peak
      return self.min_lr + m * x
    elif self.iterations < 2 * self.lr_peak:
      m = self.max_lr - self.min_lr
      x = self.iterations / self.lr_peak - 1
      return self.max_lr - m * x
    else:
      m = self.min_lr - self.min_annealing_lr
      x = (self.iterations - 2 * self.lr_peak) / (self.total_iterations - 2 * self.lr_peak)
      return self.min_lr - m * x

  def _calculate_momentum(self):
    m = self.max_momentum - self.min_momentum
    if self.iterations < self.lr_peak:
      x = self.iterations / self.lr_peak
      return self.max_momentum - m * x
    elif self.iterations < 2 * self.lr_peak:
      x = self.iterations / self.lr_peak - 1
      return self.min_momentum + m * x
    else:
      return self.max_momentum
