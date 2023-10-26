from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
import tensorflow as tf

class CustomAdam(Optimizer):
    def __init__(self, learning_rate=0.001, name='CustomAdam', **kwargs):
        super(CustomAdam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')

        m_scaled_g_values = grad * lr_t

        var_update = var.assign_sub(m_scaled_g_values, use_locking=self._use_locking)
        m_t = m.assign(m * 0.9 + m_scaled_g_values, use_locking=self._use_locking)

        updates = [var_update, m_t]
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Your implementation for sparse gradient updates
        raise NotImplementedError("Sparse gradient updates are not supported.")

# Usage in app.py
# import tensorflow as tf
# from custom_optimizer import CustomAdam
# from tensorflow.keras.models import load_model
#
# lstm_model = load_model('lstm_model.h5', custom_objects={'CustomAdam': CustomAdam})
