import tensorflow as tf

class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, freq_mask_param=10, time_mask_param=10, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(self, mel_spectrogram):
        # Frequency Masking
        freq_mask = tf.random.uniform(shape=[], minval=0, maxval=self.freq_mask_param, dtype=tf.int32)
        freq_start = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(mel_spectrogram)[0] - freq_mask, dtype=tf.int32)
        freq_end = freq_start + freq_mask
        mel_spectrogram = tf.tensor_scatter_nd_update(
            mel_spectrogram,
            [[i, j] for i in range(freq_start, freq_end) for j in range(tf.shape(mel_spectrogram)[1])],
            tf.zeros((freq_mask * tf.shape(mel_spectrogram)[1]), dtype=tf.float32)
        )

        # Time Masking
        time_mask = tf.random.uniform(shape=[], minval=0, maxval=self.time_mask_param, dtype=tf.int32)
        time_start = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(mel_spectrogram)[1] - time_mask, dtype=tf.int32)
        time_end = time_start + time_mask
        mel_spectrogram = tf.tensor_scatter_nd_update(
            mel_spectrogram,
            [[i, j] for i in range(tf.shape(mel_spectrogram)[0]) for j in range(time_start, time_end)],
            tf.zeros((tf.shape(mel_spectrogram)[0] * time_mask), dtype=tf.float32)
        )
        return mel_spectrogram

    def get_config(self):
        config = super().get_config()
        config.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
        })
        return config


