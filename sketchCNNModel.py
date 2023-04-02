class_image_map = {
    "Alfandega R-21": "66 (1).png",
    "Altura maxima permitida R-15": "66 (2).png",
    "Ciclista transite a direita R-35b": "66 (3).png",
    "Ciclista transite a esquerda R-35a": "66 (4).png",
    "Ciclistas a esquerda pedestres a direita R-36a": "66 (5).png",
    "Circulacao exclusiva de bicicletas R-34": "66 (6).png",
    "Circulacao exclusiva de caminhoes R-39": "66 (7).png",
    "Circulacao exclusiva de onibus R-32": "66 (8).png",
    "Comprimento maximo permitido R-18": "66 (9).png",
    "Conserve se a direita R-23": "66 (10).png",
    "De preferencia R-2": "66 (11).png",
    "Duplo sentido de circulacao R-28": "66 (12).png",
    "Passagem obrigatoria R-24b": "66 (13).png",
    "Pedestre ande pela direita R-31": "66 (14).png",
    "Pedestre ande pela esquerda R-30": "66 (15).png",
    "Pedestres a esquerda ciclistas a direita R-36b": "66 (16).png",
    "Peso bruto total maximo permitido R-14": "66 (17).png",
    "Peso maximo permitido por eixo R-17": "66 (18).png",
    "Proibido acionar buzina ou sinal sonoro R-20": "66 (19).png",
    "Proibido estacionar R-6a": "66 (20).png",
    "Proibido mudar de faixa ou pista de transito da direita para esquerda R-8b": "66 (21).png",
    "Proibido mudar de faixa ou pista de transito da esquerda para direita R-8a": "66 (22).png",
    "Proibido parar e estacionar R-6c": "66 (23).png",
    "Proibido retornar a direita R-5b": "66 (24).png",
    "Proibido retornar a esquerda R-5a": "66 (25).png",
    "Proibido transito de bicicletas R-12": "66 (26).png",
    "Proibido transito de caminhoes R-9": "66 (27).png",
    "Proibido transito de motocicletas motonetas e ciclomotores R-37": "66 (28).png",
    "Proibido transito de onibus R-38": "66 (29).png",
    "Proibido transito de pedestres R-29": "66 (30).png",
    "Proibido transito de tratores e maquinas de obras R-13": "66 (31).png",
    "Proibido transito de veiculos auto motores R-10": "66 (32).png",
    "Proibido transito de veiculos auto motores R-10": "66 (32).png",
    "Proibido transito de veiculos de tracao animal R-11": "66 (33).png",
    "Proibido transito de bicicletas R-12": "66 (34).png",
    "Proibido transito de tratores e maquinas de obras R-13": "66 (35).png",
    "Peso bruto total maximo permitido R-14": "66 (36).png",
    "Peso maximo permitido por eixo R-17": "66 (37).png",
    "Velocidade maxima permitida R-19": "66 (38).png",
    "Proibido acionar buzina ou sinal sonoro R-20": "66 (39).png",
    "Uso obrigatorio de corrente R-22": "66 (40).png",
    "Conserve se a direita R-23": "66 (41).png",
    "Passagem obrigatoria R-24b": "66 (42).png",
    "Vire a esquerda R-25a": "66 (43).png",
    "Vire a direita R-25b": "66 (44).png",
    "Siga em frente ou a esquerda R-25c": "66 (45).png",
    "Siga em frente ou a direita R-25d": "66 (46).png",
    "Siga em frente R-26": "66 (47).png",
    "Onibus caminhoes e veiculos de grande porte mantenham se a direita R-27": "66 (48).png",
    "Duplo sentido de circulacao R-28": "66 (49).png",
    "Pedestre ande pela esquerda R-30": "66 (50).png",
    "Pedestre ande pela direita R-31": "66 (51).png",
    "Sentido circular na rota R-33": "66 (52).png",
    "Circulacao exclusiva de bicicletas R-34": "66 (53).png",
    "Ciclista transite a esquerda R-35a": "66 (54).png",
    "Ciclistas a esquerda pedestres a direita R-36a": "66 (55).png",
    "Proibido transito de motocicletas motonetas e ciclomotores R-37": "66 (56).png",
    "Proibido transito de onibus R-38": "66 (57).png",
    "Circulacao exclusiva de caminhoes R-39": "66 (58).png",
    "Transito proibido a carros de mao R-40": "66 (59).png",
    "Regulamentacao Inform 1": "66 (60).png",
    "Regulamentacao Inform 2": "66 (61).png",
    "Regulamentacao Inform 3": "66 (62).png",
    "Regulamentacao Inform 4": "66 (63).png",
    "Regulamentacao Inform 5": "66 (64).png",
    "Regulamentacao Inform 6": "66 (65).png",
    "Regulamentacao Inform 7": "66 (66).png"
}


#now we will use the following neural network model to do the train and test
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

# Define the class to integer mapping


# Load the images and corresponding labels
images = []
labels = []
for key in class_image_map:
    for i in range(1, 67):
        image_path = os.path.join("/content/sample_data/testImages", f"{i}.png")

        if class_image_map[key] == f"{i}.png":
            img = keras.preprocessing.image.load_img(
                image_path, target_size=(224, 224)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(key)

# Convert the images and labels to numpy arrays
images = tf.convert_to_tensor(images)
labels = tf.convert_to_tensor(labels)

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Define the CNN model
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
