import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
from sklearn.metrics import confusion_matrix, roc_curve### metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (Dense, Input, Permute, RandomContrast,
                                     RandomFlip, RandomRotation,
                                     Rescaling, Resizing) 
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from transformers import ViTFeatureExtractor, TFViTModel, ViTConfig
from tensorflow.keras.layers import Input, Layer

CONFIGURATION = {
   "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 20,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 128,
    "NUM_CLASSES": 3,
    "PATCH_SIZE": 16,
    "PROJ_DIM": 768,
    "CLASS_NAMES": ["angry", "happy", "sad"],
}

train_directory = "./data/Emotions Dataset/Emotions Dataset/train"
val_directory = "./data/Emotions Dataset/Emotions Dataset/test"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    labels='inferred',
    label_mode='categorical',
    class_names=CONFIGURATION["CLASS_NAMES"],
    color_mode='rgb',
    batch_size=CONFIGURATION["BATCH_SIZE"],
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_directory,
    labels='inferred',
    label_mode='categorical',
    class_names=CONFIGURATION["CLASS_NAMES"],
    color_mode='rgb',
    batch_size=1,
    image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
    shuffle=True,
    seed=99,
)

#Visualize data
plt.figure(figsize = (12,12))

for images, labels in train_dataset.take(1):
  for i in range(16):
    ax = plt.subplot(4,4, i+1)
    plt.imshow(images[i]/255.)
    plt.title(CONFIGURATION["CLASS_NAMES"][tf.argmax(labels[i], axis = 0).numpy()])
    plt.axis("off")


## Data Augmentation
augment_layers = tf.keras.Sequential([
  RandomRotation(factor = (-0.025, 0.025)),
  RandomFlip(mode='horizontal',),
  RandomContrast(factor=0.1),
])
def augment_layer(image, label):
  return augment_layers(image, training = True), label

###Dataset Preparation
training_dataset = (
    train_dataset
    .map(augment_layer, num_parallel_calls = tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
validation_dataset = (
    val_dataset
    .prefetch(tf.data.AUTOTUNE)
)
resize_rescale_layers = tf.keras.Sequential([
       Resizing(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
       Rescaling(1./255),
])

resize_rescale_hf = tf.keras.Sequential([
       Resizing(224, 224),
       Rescaling(1./255),
       Permute((3,1,2))
])

# ## Modeling
# class ViTModelWrapper(Layer):
#   def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
#       super(ViTModelWrapper, self).__init__()
#       self.vit_model = TFViTModel.from_pretrained(model_name)

#   def call(self, inputs):
#       x = tf.image.resize(inputs, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
#       x = tf.transpose(x, perm=[0, 3, 1, 2])  # (batch, 3, 224, 224)
#       x = resize_rescale_hf(inputs)
#       x = self.vit_model.vit(x, training=False)[0][:, 0, :]
#       return x

# inputs = Input(shape=(256, 256, 3))
# x = ViTModelWrapper()(inputs)
# outputs = Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')(x)

# hf_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# # hf_model.summary()

# class ViTWithAttentionLayer(Layer):
#     def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
#         super(ViTWithAttentionLayer, self).__init__()
#         config = ViTConfig.from_pretrained(model_name, output_attentions=True)
#         self.vit_model = TFViTModel.from_pretrained(model_name, config=config)

#     def call(self, inputs):
#         x = tf.image.resize(inputs, (256, 256))
#         x = resize_rescale_hf(inputs)
#         outputs = self.vit_model.vit(x)
#         attentions = outputs['attentions']
#         return attentions

# inputs = Input(shape=(256, 256, 3))
# x = ViTWithAttentionLayer()(inputs)

# model = tf.keras.Model(inputs=inputs, outputs=x)

 ## Training
n_sample_0 = 1525 # angry
n_sample_1 = 3019 # happy
n_sample_2 = 2255 # sad
class_weights = {0:6799/n_sample_0, 1: 6799/n_sample_1, 2: 6799/n_sample_2}

loss_function = CategoricalCrossentropy()
metrics = [CategoricalAccuracy(name = "accuracy"), TopKCategoricalAccuracy(k=2, name = "top_k_accuracy")]



resize_rescale_hf = tf.keras.Sequential([
       Resizing(224, 224),
       Rescaling(1./255),
       Permute((3,1,2))
])


class ViTModelWrapper(Layer):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super(ViTModelWrapper, self).__init__()
        self.vit_model = TFViTModel.from_pretrained(model_name)

    def call(self, inputs):
        x = tf.image.resize(inputs, (CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]))
        x = resize_rescale_hf(inputs)
        x = self.vit_model.vit(x, training=False)[0][:, 0, :]
        return x

# Define input layer
inputs = Input(shape=(256, 256, 3))
# Use the ViT wrapper layer
x = ViTModelWrapper()(inputs)
# Define output layer
outputs = Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')(x)

# Create the Keras model
hf_model = tf.keras.Model(inputs=inputs, outputs=outputs)
# hf_model.summary()

hf_model.compile(
  optimizer = Adam(learning_rate = CONFIGURATION["LEARNING_RATE"]),
  loss = loss_function,
  metrics = metrics,
)

# Huấn luyện mô hình ViT
hf_history = hf_model.fit(
  training_dataset,
  validation_data = validation_dataset,
  epochs = CONFIGURATION["N_EPOCHS"],
  verbose = 1,
  class_weight = class_weights,
)

# model loss
plt.plot(hf_history.history['loss'])
plt.plot(hf_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

#model accuracy
plt.plot(hf_history.history['accuracy'])
plt.plot(hf_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()

hf_model.save("./model/hf_model.h5")
