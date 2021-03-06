import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


from text_generate import TextGenerator
import copy
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K


class Config:
    embedding_size = 300
    batch_size = 32
    drop_rate = 0.1


train_generator = TextGenerator(Config.batch_size, 10000)
train_generator.init("/home/qw/xiaohongshu_category_arrange_train", True)
val_generator = copy.deepcopy(train_generator)
val_generator.init("/home/qw/xiaohongshu_category_arrange_val", False)


class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(
            name="WQ",
            shape=(input_shape[0][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.WK = self.add_weight(
            name="WK",
            shape=(input_shape[1][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.WV = self.add_weight(
            name="WV",
            shape=(input_shape[2][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode="mul"):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == "mul":
                return inputs * mask
            if mode == "add":
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        Q_seq, K_seq, V_seq = x
        Q_len, V_len = None, None
        print("build attention")

        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))

        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))

        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, "add")
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)

        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, "mul")

        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def transfromer(n_symbols, config):
    S_inputs = Input(shape=(None,), dtype="int32")
    embeddings = Embedding(
        input_dim=n_symbols,
        output_dim=config.embedding_size,
        input_length=config.batch_size,
    )(S_inputs)
    # embeddings = Position_Embedding()(embeddings)
    O_seq = Attention(8, 16)([embeddings, embeddings, embeddings,])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(Config.drop_rate)(O_seq)
    outputs = Dense(2, activation="softmax")(O_seq)
    model = Model(inputs=S_inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


model = transfromer(train_generator.label_size, Config)
model.summary()
