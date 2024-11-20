import tensorflow as tf
import matplotlib.pyplot as plt### plotting bar chart
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns
import random
import matplotlib.cm as cm
import numpy as np
from tensorflow.keras.models import load_model

CONFIGURATION = {
    "BATCH_SIZE": 32,
    "IM_SIZE": 256,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 30,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 6,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 128,
    "NUM_CLASSES": 3,#7
    "PATCH_SIZE": 16,
    "PROJ_DIM": 768,
    "CLASS_NAMES": ["angry", "happy", "sad"],
    # "CLASS_NAMES": ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', "surprised"]
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
validation_dataset = (
    val_dataset
    .prefetch(tf.data.AUTOTUNE)
)
trained_model = load_model('./model/emotionModel.h5')

all_images = []
all_labels = []
for images, labels in validation_dataset:
    all_images.extend(images)
    all_labels.extend(labels)

# Randomly sample 16 images and their corresponding labels
random_indices = random.sample(range(len(all_images)), 16)
random_images = [all_images[i] for i in random_indices]
random_labels = [all_labels[i] for i in random_indices]

plt.figure(figsize=(12, 12))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(random_images[i] / 255.0)

    true_label = CONFIGURATION["CLASS_NAMES"][tf.argmax(random_labels[i], axis=-1).numpy()]
    
    predictions = trained_model(tf.expand_dims(random_images[i], axis=0)) 
    probabilities = tf.nn.softmax(predictions[0]).numpy() 
    pred_label = CONFIGURATION["CLASS_NAMES"][np.argmax(probabilities)]

    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis("off")
plt.show()

#confusion matrix
predicted = []
labels = []

for im, label in validation_dataset:
  predicted.append(trained_model(im))
  labels.append(label.numpy())

print(np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()]))
print(np.concatenate([np.argmax(predicted[:-1], axis = -1).flatten(), np.argmax(predicted[-1], axis = -1).flatten()]))

pred = np.concatenate([np.argmax(predicted[:-1], axis = -1).flatten(), np.argmax(predicted[-1], axis = -1).flatten()])
lab = np.concatenate([np.argmax(labels[:-1], axis = -1).flatten(), np.argmax(labels[-1], axis = -1).flatten()])

cm = confusion_matrix(lab, pred)
print(cm)
plt.figure(figsize=(8,8))

sns.heatmap(cm, annot=True, fmt="d", xticklabels=CONFIGURATION['CLASS_NAMES'], yticklabels=CONFIGURATION['CLASS_NAMES'])
plt.title('Confusion matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()