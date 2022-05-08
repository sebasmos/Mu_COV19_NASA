import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class MyModel(tf.keras.Model):

    def __init__(self, x, n_outputs=4):
        
        super(MyModel, self).__init__()

        self.input_shape = x[0].shape
        self.layer_1 = Dense(128, input_shape=self.input_shape, activation='relu')
        self.layer_2 = Dense(64, activation='relu')
        # Sigmoid Out
        self.out = Dense(n_classes, activation='softmax')      

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.out(x)
        
        return x

