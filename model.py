# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Activation, Input, Lambda, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization, Dropout, Reshape, Flatten
from tensorflow.keras.layers import Add, Concatenate, Average

#%%
def slice(x, index):  
    # Slice function to extract a specific channel (index) from the input
    return x[:,:,index]

def RespNet(maxlen=300, 
            drop_rate=0.1, 
            width_multiple=2, 
            classes=2, 
            classifier_activation=None, 
            model_name='RespNet', 
            classify=False,
            active_learner=False,
            include_top=False,
            label_count = [['DelayedCycling', 2], ['PrematureCycling', 2], ['DoubleTrig', 2], ['InefTrig', 2]],
            **kwargs):
    
    # Input layer with shape of (maxlen, 2)
    inputs = Input(shape=(maxlen, 2)) 
    
    # Slice and reshape the first channel of the input (index=0)
    x1 = Lambda(slice, output_shape=(maxlen, 1), arguments={'index':0})(inputs)
    x1 = Reshape((maxlen, 1))(x1)
    # Apply Conv1D with BatchNormalization, Activation, and MaxPooling for feature extraction
    x1 = Conv1D(filters=64*width_multiple, kernel_size=11, strides=1, padding='same', use_bias=True, name='conv1d_11')(x1)
    x1 = BatchNormalization(epsilon=1.001e-5, name='bn_11')(x1)
    x1 = Activation('relu')(x1) 
    x1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(x1)
    x1 = Dropout(drop_rate)(x1)
    
    # Continue applying Conv1D layers for further feature extraction
    x1 = Conv1D(filters=32*width_multiple, kernel_size=5, strides=1, padding='same', use_bias=True, name='conv1d_12')(x1)
    x1 = BatchNormalization(epsilon=1.001e-5, name='bn_12')(x1)
    x1 = Activation('relu')(x1) 
    x1 = Dropout(drop_rate)(x1)

    x1 = Conv1D(filters=32*width_multiple, kernel_size=5, strides=1, padding='same', use_bias=True, name='conv1d_13')(x1)
    x1 = BatchNormalization(epsilon=1.001e-5, name='bn_13')(x1)
    x1 = Activation('relu')(x1) 
    x1 = Dropout(drop_rate)(x1)

    x1 = Conv1D(filters=32*width_multiple, kernel_size=3, strides=1, padding='same', use_bias=True, name='conv1d_14')(x1)
    x1 = BatchNormalization(epsilon=1.001e-5, name='bn_14')(x1)
    x1 = Activation('relu')(x1) 
    x1 = Dropout(drop_rate)(x1)

    x1 = Conv1D(filters=16*width_multiple, kernel_size=3, strides=1, padding='same', use_bias=True, name='conv1d_15')(x1)
    x1 = BatchNormalization(epsilon=1.001e-5, name='bn_15')(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(pool_size=2, strides=2, padding='same')(x1)
    x1 = Dropout(drop_rate)(x1)
    
    # Slice and reshape the second channel of the input (index=1)
    x2 = Lambda(slice, output_shape=(maxlen, 1), arguments={'index':1})(inputs)
    x2 = Reshape((maxlen, 1))(x2)  
    # Apply similar Conv1D, BatchNormalization, and Dropout layers as for x1
    x2 = Conv1D(filters=64*width_multiple, kernel_size=11, strides=1, padding='same', use_bias=True, name='conv1d_21')(x2)
    x2 = BatchNormalization(epsilon=1.001e-5, name='bn_21')(x2)
    x2 = Activation('relu')(x2) 
    x2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(x2)
    x2 = Dropout(drop_rate)(x2)

    x2 = Conv1D(filters=32*width_multiple, kernel_size=5, strides=1, padding='same', use_bias=True, name='conv1d_22')(x2)
    x2 = BatchNormalization(epsilon=1.001e-5, name='bn_22')(x2)
    x2 = Activation('relu')(x2) 
    x2 = Dropout(drop_rate)(x2)

    x2 = Conv1D(filters=32*width_multiple, kernel_size=5, strides=1, padding='same', use_bias=True, name='conv1d_23')(x2)
    x2 = BatchNormalization(epsilon=1.001e-5, name='bn_23')(x2)
    x2 = Activation('relu')(x2) 
    x2 = Dropout(drop_rate)(x2)

    x2 = Conv1D(filters=32*width_multiple, kernel_size=3, strides=1, padding='same', use_bias=True, name='conv1d_24')(x2)
    x2 = BatchNormalization(epsilon=1.001e-5, name='bn_24')(x2)
    x2 = Activation('relu')(x2) 
    x2 = Dropout(drop_rate)(x2)

    x2 = Conv1D(filters=16*width_multiple, kernel_size=3, strides=1, padding='same', use_bias=True, name='conv1d_25')(x2)
    x2 = BatchNormalization(epsilon=1.001e-5, name='bn_25')(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(pool_size=2, strides=2, padding='same')(x2)
    x2 = Dropout(drop_rate)(x2)
    
    # Add the two feature maps (x1 and x2)
    x = Add(name='add')([x1, x2])
       
    if active_learner:
        # If active learner, flatten and apply Embedding and Dense layers for classification tasks
        x = Flatten(name='flatten')(x)
        x = Dense(128, activation='relu', name='embedding')(x)
        x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count]
    elif include_top:
        # If include_top, apply GlobalAveragePooling and final Dense layer for classification
        x = GlobalAveragePooling1D(name='avg_pool')(x)
        x = Dense(classes, activation=classifier_activation, name='predictions')(x)
    else:
        # If contrastive pretraining, apply GlobalAveragePooling
        x = GlobalAveragePooling1D(name='avg_pool')(x)

    if classify:
        # Apply Dense layers for classification if classify is True
        x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count]
        
    # Define the model    
    model = Model(inputs, x, name=model_name)
    
    return model

#%%

# Non-linear MLP as projection head
def get_projection(project_dim=128):
    # Define a simple projection head with two dense layers
    inputs = Input(shape=(project_dim,))
    
    # First dense layer with ReLU activation
    x = Dense(project_dim, activation="relu", name='projection_1')(inputs)
    # Second dense layer to map to the projection space
    p = Dense(project_dim, name='projection_2')(x)
    
    # Define and return the projection head model
    h = Model(inputs, p, name="projection_head")
    return h

# Contrastive learning Model
class ContrastiveModel(tf.keras.Model):
    def __init__(self, temperature, encoder, projection_head):
        # Initialize contrastive model with temperature, encoder, and projection head
        super().__init__()

        self.temperature = temperature
        self.encoder = encoder
        self.projection_head = projection_head
        
        # Print summary of encoder and projection head
        self.encoder.summary()
        self.projection_head.summary()


    def compile(self, contrastive_optimizer, **kwargs):
        # Compile the contrastive model with optimizer and loss function
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer

        # Initialize loss and accuracy metrics
        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="c_acc")


    @property
    def metrics(self):
        # Return the list of metrics to track
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature)

        # Create contrastive labels (batch indices)
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        
        # Update accuracy metrics
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(contrastive_labels, tf.transpose(similarities))

        # Symmetrized version of the cross-entropy loss
        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(contrastive_labels, similarities, from_logits=True)
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(contrastive_labels, tf.transpose(similarities), from_logits=True)
        
        # Return average loss
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        # Unpack data into two augmented views of the same data
        (ds_one, ds_two) = data
        
        with tf.GradientTape() as tape:
            # Pass both augmented views through the encoder to get feature representations
            features_1 = self.encoder(ds_one, training=True)
            features_2 = self.encoder(ds_two, training=True)
            
            # Pass the feature representations through the projection head
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            
            # Compute contrastive loss
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        
        # Compute gradients and apply updates
        gradients = tape.gradient(contrastive_loss,
                                  self.encoder.trainable_weights + self.projection_head.trainable_weights,
                                  )
        
        self.contrastive_optimizer.apply_gradients(
            zip(gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
                )
            )
        
        # Update loss tracker
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack test data and labels
        test_data, labels = data

        # Pass the test data through the encoder and compute class logits
        features = self.encoder(test_data, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        
        # Update loss and accuracy metrics for the probe
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)

        # Return only the probe metrics at test time
        return {m.name: m.result() for m in self.metrics[2:]}
    
    def call(self, data, training=True):
        # your custom code when you call the model or just pass, you don't need this method for training
        pass
    
#%%