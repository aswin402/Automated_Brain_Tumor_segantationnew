import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow available but GPU support disabled for CPU-only mode")
except (ImportError, RuntimeError) as e:
    TENSORFLOW_AVAILABLE = False
    logger.warning(f"TensorFlow not available or error: {e} - U-Net segmentation features will not be available")


if TENSORFLOW_AVAILABLE:
    class DiceLoss(keras.losses.Loss):
        """Dice Loss for segmentation."""
        def __init__(self, smooth=1e-6, **kwargs):
            super(DiceLoss, self).__init__(**kwargs)
            self.smooth = smooth
        
        def call(self, y_true, y_pred):
            y_pred = tf.cast(y_pred, y_true.dtype)
            
            intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
            union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1.0 - tf.reduce_mean(dice)


    def combined_loss(y_true, y_pred, dice_weight=0.5, bce_weight=0.5):
        """Combined Dice Loss + Binary Cross Entropy."""
        dice = DiceLoss()(y_true, y_pred)
        bce = binary_crossentropy(y_true, y_pred)
        bce = tf.reduce_mean(bce)
        
        return dice_weight * dice + bce_weight * bce


    class UNet:
        def __init__(self, input_shape=(256, 256, 1), num_classes=1, filters_start=32, 
                     dropout_rate=0.3, use_batch_norm=True):
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.filters_start = filters_start
            self.dropout_rate = dropout_rate
            self.use_batch_norm = use_batch_norm
            
        def conv_block(self, inputs, filters, kernel_size=3, activation='relu', padding='same'):
            """Convolutional block with batch normalization and residual connection."""
            x = layers.Conv2D(filters, kernel_size, padding=padding, activation=None, 
                             kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)
            
            x = layers.Conv2D(filters, kernel_size, padding=padding, activation=None,
                             kernel_regularizer=keras.regularizers.l2(1e-4))(x)
            if self.use_batch_norm:
                x = layers.BatchNormalization()(x)
            
            if inputs.shape[-1] == filters:
                x = layers.Add()([x, inputs])
            
            x = layers.Activation(activation)(x)
            return x
        
        def encoder_block(self, inputs, filters):
            """Encoder block with downsampling."""
            conv = self.conv_block(inputs, filters)
            pool = layers.MaxPooling2D((2, 2))(conv)
            return conv, pool
        
        def decoder_block(self, inputs, skip_connection, filters):
            """Decoder block with upsampling."""
            x = layers.UpSampling2D((2, 2))(inputs)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            
            x = layers.Concatenate()([x, skip_connection])
            conv = self.conv_block(x, filters)
            
            return conv
        
        def build(self):
            """Build enhanced U-Net architecture with 5 levels."""
            inputs = layers.Input(shape=self.input_shape)
            
            # Encoder
            logger.info("Building U-Net encoder...")
            conv1, pool1 = self.encoder_block(inputs, self.filters_start)
            conv2, pool2 = self.encoder_block(pool1, self.filters_start * 2)
            conv3, pool3 = self.encoder_block(pool2, self.filters_start * 4)
            conv4, pool4 = self.encoder_block(pool3, self.filters_start * 8)
            conv5, pool5 = self.encoder_block(pool4, self.filters_start * 16)
            
            # Bottleneck
            logger.info("Building U-Net bottleneck...")
            conv_bn = self.conv_block(pool5, self.filters_start * 32)
            conv_bn = layers.Dropout(self.dropout_rate * 0.8)(conv_bn)
            
            # Decoder
            logger.info("Building U-Net decoder...")
            dec5 = self.decoder_block(conv_bn, conv5, self.filters_start * 16)
            dec5 = layers.Dropout(self.dropout_rate * 0.7)(dec5)
            
            dec4 = self.decoder_block(dec5, conv4, self.filters_start * 8)
            dec4 = layers.Dropout(self.dropout_rate * 0.6)(dec4)
            
            dec3 = self.decoder_block(dec4, conv3, self.filters_start * 4)
            dec3 = layers.Dropout(self.dropout_rate * 0.5)(dec3)
            
            dec2 = self.decoder_block(dec3, conv2, self.filters_start * 2)
            dec1 = self.decoder_block(dec2, conv1, self.filters_start)
            
            # Output layer
            if self.num_classes == 1:
                outputs = layers.Conv2D(1, 1, activation='sigmoid', padding='same')(dec1)
            else:
                outputs = layers.Conv2D(self.num_classes, 1, activation='softmax', padding='same')(dec1)
            
            model = Model(inputs, outputs, name='UNet')
            logger.info("U-Net model built successfully")
            
            return model


    def build_unet_model(input_shape=(256, 256, 1), num_classes=1, filters_start=32, 
                         dropout_rate=0.3, use_batch_norm=True):
        """Build and return U-Net model."""
        unet = UNet(input_shape=input_shape, num_classes=num_classes, 
                    filters_start=filters_start, dropout_rate=dropout_rate,
                    use_batch_norm=use_batch_norm)
        return unet.build()


    def compile_unet_model(model, learning_rate=0.001):
        """Compile U-Net model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss=DiceLoss(),
            metrics=[
                keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5),
                keras.metrics.Recall(),
                keras.metrics.Precision(),
            ]
        )
        
        return model


    class SegmentationMetrics:
        @staticmethod
        def dice_score(y_true, y_pred, smooth=1e-6):
            """Calculate Dice Score."""
            y_pred = (y_pred > 0.5).astype(float)
            
            intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
            union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
            
            dice = (2.0 * intersection + smooth) / (union + smooth)
            return tf.reduce_mean(dice).numpy()
        
        @staticmethod
        def iou_score(y_true, y_pred, smooth=1e-6):
            """Calculate IoU (Intersection over Union) Score."""
            y_pred = (y_pred > 0.5).astype(float)
            
            intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
            union = tf.reduce_sum(tf.maximum(y_true, y_pred), axis=[1, 2, 3])
            
            iou = (intersection + smooth) / (union + smooth)
            return tf.reduce_mean(iou).numpy()
else:
    logger.warning("TensorFlow components are disabled. Cannot use segmentation features.")


if __name__ == "__main__":
    if TENSORFLOW_AVAILABLE:
        model = build_unet_model()
        model.summary()
        
        compile_unet_model(model)
        print("Model compiled successfully")
    else:
        print("TensorFlow not installed - cannot build U-Net model")
