from tensorflow.keras import initializers, layers
import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np
import sympy
import cirq

tf.keras.saving.get_custom_objects().clear()

@tf.keras.saving.register_keras_serializable()
class Circuit(layers.Layer):
    """Implements a parameterised quantum circuit with input re-uploading as a layer."""
    def __init__(self, num_layers, name="PQC", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_layers = num_layers

    def build(self, input_shape):
        self.num_qubits = input_shape[1]
        self.qubits = cirq.GridQubit.rect(1, self.num_qubits)
        
        # Symbols for PQC weights
        params = sympy.symbols(f'param(0:{3 * self.num_layers * self.num_qubits})') # three rotations per layer per qubit
        inputs = sympy.symbols(f'input(0:{self.num_layers})_(0:{self.num_qubits})')
        params = np.asarray(params).reshape((self.num_layers, self.num_qubits, 3))
        inputs = np.asarray(inputs).reshape((self.num_layers, self.num_qubits))

        # Circuit Definition
        self.circuit = cirq.Circuit()
        for l in range(self.num_layers):
            self.circuit += self.encode(self.qubits, inputs[l])
            self.circuit += self.entangle(self.qubits)
            self.circuit += self.rotate(self.qubits, params[l])

        self.rotation_weights = self.add_weight(shape=(3 * self.num_qubits * self.num_layers,),
                                     initializer=initializers.RandomNormal(mean=0, stddev=np.pi/2),
                                     trainable=True, name="Rotation_Weights")
        
        self.input_weights = self.add_weight(shape=(self.num_qubits * self.num_layers,),
                                     initializer='ones', # expect inputs in range [-1,1]
                                     trainable=True, name="Input_Weights")
        
        # Symbol order
        symbols = list(map(str, np.append(params.flatten(), inputs.flatten())))
        self.symbol_order = tf.constant([symbols.index(a) for a in sorted(symbols)])
        self.pqc = tfq.layers.ControlledPQC(self.circuit, [cirq.Z(q) for q in self.qubits])
        
    @staticmethod
    def encode(qubits, inputs):
        """Returns a layer encoding the state"""
        return cirq.Circuit(cirq.rx(inputs[i])(q) for i, q in enumerate(qubits))

    @staticmethod
    def rotate(qubits, params):
        """Returns a layer rotating each qubit."""
        return cirq.Circuit([cirq.rx(params[i, 0])(q), cirq.ry(params[i, 1])(q), cirq.rz(params[i, 2])(q)] for i, q in enumerate(qubits))

    @staticmethod
    def entangle(qubits):
        """Returns a layer entangling the qubits with CZ gates."""
        return [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, np.roll(qubits, -1, axis=0))]

    @tf.function
    def call(self, inputs):
        batch_size = tf.gather(tf.shape(inputs), 0)
        tiled_params = tf.tile([self.rotation_weights], multiples=[batch_size, 1])
        repeated_inputs = tf.repeat(inputs, repeats=self.num_layers, axis=1)
        weighted_inputs = tf.multiply(self.input_weights, repeated_inputs)
        tiled_weighted_inputs = tf.tile(weighted_inputs, multiples=[1, batch_size])

        parameters = tf.concat([tiled_params, tiled_weighted_inputs], axis=1)
        parameters_ordered = tf.gather(
                        parameters,
                        self.symbol_order, axis=1)
        
        circuits = tf.repeat(tfq.convert_to_tensor([cirq.Circuit()]),
                        repeats=batch_size)
        
        return self.pqc([circuits, parameters_ordered])

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
        })
        return config

@tf.keras.saving.register_keras_serializable()
class Scale(layers.Layer):
    def __init__(self, name="Scaling", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[1],), initializer='ones',
                                 trainable=True)
    tf.function
    def call(self, inputs):
        return tf.multiply(inputs, self.w)
        
    def get_config(self):
        return super().get_config()