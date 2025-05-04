import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model

#params
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
Model_Path = "./DogDetector/dog_detector_model.keras"

# load dataset
train_dir = "./DogDetector/dataset/train"
val_dir = "./DogDetector/dataset/val"


train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

#make or load model
if os.path.exists(Model_Path):
    print(f"model exists at: {Model_Path}")
    model = load_model(Model_Path)

else:
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    #sets model and sets the optimizer type
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    # training the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    model.save("./DogDetector/dog_detector_model.keras")

    # Step 6: Plot Results
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Dog Detection Accuracy')
    plt.show()


#checks excaluation and prediction
loss, accuracy = model.evaluate(val_gen)
print(f"Val accuracy is : {accuracy} - Val loss is : {loss}")



    #if model already exists and you want to use it to predict something, uncomment code below
# img_path = "./DogDetector/labrador.jpg"
# img = image.load_img(img_path, target_size=(256,256))
# img_array = image.img_to_array(img)
# img_array = img_array/255.0
# img_array = np.expand_dims(img_array, axis=0)
# prediction = model.predict(img_array)
# print("▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲")
# if prediction[0][0] > .5:
#     print("image prediction: Dog!")
# else:
#     print("image prediction: No Dog!")
# print("▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼")