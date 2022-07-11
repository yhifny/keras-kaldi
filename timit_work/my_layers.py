import tensorflow as tf

class ContextExpansion(tf.keras.layers.Layer):
    def __init__(self, left=0,right=0, **kwargs):
        super().__init__(**kwargs)
        self.left = left
        self.right = right

    def expand(self, feature):
        if self.left == self.right == 0:
            return feature
        # expand
        feature = [feature]
        for i in range(self.left):
            feature.append(tf.concat((tf.reshape(feature[-1][0], [1, feature[-1][0].shape[-1]]), feature[-1][:-1]), axis=0))
        feature.reverse()
        for i in range(self.right):
            
            feature.append(tf.concat([feature[-1][1:], tf.reshape(feature[-1][-1], [1, feature[-1][-1].shape[-1]])], axis=0))
        return tf.concat(feature,  axis=1)
     
    def call(self, feature, **kwargs):
        print(tf.shape(feature)[0])
        out= tf.map_fn(self.expand, feature)
        return tf.stack(out,  axis=0)
    
    def compute_output_shape(self, input_shape):
        output_dim = input_shape[2]*(self.left +self.right +1)
        return (input_shape[0], input_shape[1], output_dim)

    def get_config(self):
        config = {
            'left': self.left,
            'right': self.right,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
