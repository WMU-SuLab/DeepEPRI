from tensorflow.keras import initializers
#from keras.engine.topology import Layer, InputSpec
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import LSTM
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Activation
from tensorflow.keras import metrics
#from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import random
import os

MAX_LEN_en = 100
MAX_LEN_pr = 100
NB_WORDS = 16385
EMBEDDING_DIM = 100
                          
class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self._trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) +
                      K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def set_seed(seed):
    tf.random.set_seed(seed)
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def binary_focal_loss(alpha, gamma):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed

alpha = 0.5
gamma = 3
LOSS = binary_focal_loss(alpha, gamma)

def DeepEPRI():
    enhancers = Input(shape=(MAX_LEN_en,))
    promoters = Input(shape=(MAX_LEN_pr,))


    emb_en = Embedding(NB_WORDS, EMBEDDING_DIM, trainable=True)(enhancers) 
    emb_pr = Embedding(NB_WORDS, EMBEDDING_DIM, trainable=True)(promoters) 
 
    enhancer_conv_layer = Conv1D(filters=64,
                                 kernel_size=60,
                                 padding="same",
                                 input_shape=(MAX_LEN_en, EMBEDDING_DIM) 
                                )
    enhancer_max_pool_layer = MaxPooling1D(pool_size=20, strides=20) 

    # Build enhancer branch
    enhancer_branch = Sequential()
    enhancer_branch.add(enhancer_conv_layer)
    enhancer_branch.add(Activation("relu"))
    enhancer_branch.add(enhancer_max_pool_layer)
    enhancer_branch.add(BatchNormalization())
    enhancer_branch.add(Dropout(0.5))
    enhancer_out = enhancer_branch(emb_en)
    promoter_conv_layer = Conv1D(filters=64,
                                 kernel_size=60,
                                 padding="same",
                                 input_shape=(MAX_LEN_pr, EMBEDDING_DIM)

                                )
    promoter_max_pool_layer = MaxPooling1D(pool_size=20, strides=20)

    promoter_branch = Sequential() #tf.keras.Sequential()
    promoter_branch.add(promoter_conv_layer)
    promoter_branch.add(Activation("relu"))
    promoter_branch.add(promoter_max_pool_layer)
    promoter_branch.add(BatchNormalization())
    promoter_branch.add(Dropout(0.5))
    promoter_out = promoter_branch(emb_pr)

    l_lstm_1 = Bidirectional(LSTM(50, return_sequences=True))(enhancer_out)
    l_lstm_2 = Bidirectional(LSTM(50, return_sequences=True))(promoter_out)
    l_att_1 = AttLayer(50)(l_lstm_1)
    l_att_2 = AttLayer(50)(l_lstm_2)
    subtract_layer = Subtract()([l_att_1, l_att_2])
    multiply_layer = Multiply()([l_att_1, l_att_2])

    merge_layer = Concatenate(axis=1)([l_att_1, l_att_2])
    bn = BatchNormalization()(merge_layer)
    dt = Dropout(0.5)(bn)

    dt = Dense(units=64, kernel_initializer="glorot_uniform")(dt)
    dt = BatchNormalization()(dt)
    dt = Activation("relu")(dt)
    dt = Dropout(0.5)(dt)
    preds = Dense(1, activation='sigmoid')(dt)
    model = Model(inputs=[enhancers, promoters], outputs=preds)
    adam = keras.optimizers.Adam(learning_rate=6.25e-5) #The initial learning rate is 1e-3
    model.compile(loss='binary_crossentropy',
                  #loss=LOSS,
                  optimizer=adam, metrics=['accuracy',metrics.Precision(), metrics.Recall()])
    return model


