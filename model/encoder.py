import tensorflow as tf

from .components.positional import add_timing_signal_nd

class Encoder(object):
    """Class with a __call__ method that applies convolutions to an image"""

    def __init__(self, config):
        self._config = config


    def __call__(self, training, img, dropout):
        """Applies convolutions to the image

        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels), of type
                tf.uint8
            tf.uint8 因为 2^8 = 256，所以元素值区间 [0, 255]

        Returns:
            the encoded images, shape = (?, h', w', c')

        """
        img = tf.cast(img, tf.float32) / 255.

        with tf.variable_scope("convolutional_encoder"):
            # conv + max pool -> /2
            out = tf.layers.conv2d(img, 64, 3, 1, "SAME",
                                   activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            # conv + max pool -> /2
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                                   activation=tf.nn.relu)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            # regular conv -> id
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                                   activation=tf.nn.relu)

            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                                   activation=tf.nn.relu)

            if self._config.encoder_cnn == "vanilla":
                out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

            out = tf.layers.conv2d(out, 512, 3, 1, "SAME",
                                   activation=tf.nn.relu)

            if self._config.encoder_cnn == "vanilla":
                out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

            if self._config.encoder_cnn == "cnn":
                # conv with stride /2 (replaces the 2 max pool)
                out = tf.layers.conv2d(out, 512, (2, 4), 2, "SAME")

            # conv
            out = tf.layers.conv2d(out, 512, 3, 1, "VALID",
                                   activation=tf.nn.relu)

            if self._config.positional_embeddings:
                # from tensor2tensor lib - positional embeddings
                # 嵌入位置信息（positional）
                # 后面将会有一个 flatten 的过程，会丢失掉位置信息，所以现在必须把位置信息嵌入
                # 嵌入的方法有很多，比如加，乘，缩放等等，这里用 tensor2tensor 的实现
                out = add_timing_signal_nd(out)

        return out
