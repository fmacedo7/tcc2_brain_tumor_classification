'''
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import keras
print("Keras version:", keras.__version__)

This is necessary in colab notebook to install the specific versions bellow

!pip uninstall -y tensorflow keras
!pip install tensorflow==2.11.0 keras==2.11.0
'''
# import system libs
import os
import time
import shutil
import pathlib
import itertools

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from keras.models import Sequential
from keras.optimizers import Adam, Adamax
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.normalization.batch_normalization import (
    BatchNormalization,
)
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.regularizers import l1, l2
# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')

# def limitgpu(maxmem):
# 	gpus = tf.config.list_physical_devices('GPU')
# 	if gpus:
# 		# Restrict TensorFlow to only allocate a fraction of GPU memory
# 		try:
# 			for gpu in gpus:
# 				tf.config.experimental.set_virtual_device_configuration(gpu,
# 						[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=maxmem)])
# 		except RuntimeError as e:
# 			# Virtual devices must be set before GPUs have been initialized
# 			print(e)


# # 3.5 gigaaaaaaaaaaaaaasssss
# limitgpu(3072+512) 

#---------Create a dataframe from dataset----------#

# Diretório onde estão as subpastas
base_dir = '/home/ubuntu/Documents/tcc2_brain_tumor_classification/dataset'

# Lista das classes (doenças)
classes = ['1', '2', '3']

# Define as proporções de divisão (por exemplo, 80% treino, 10% teste, 10% validação)
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# Listas para armazenar os caminhos dos arquivos e suas etiquetas
train_filepaths = []
train_labels = []

test_filepaths = []
test_labels = []

val_filepaths = []
val_labels = []

# Loop através das subpastas e coleta os caminhos de arquivos e etiquetas
for disease in classes:
    source_dir = os.path.join(base_dir, disease)
    images = os.listdir(source_dir)

    # Divide as imagens em treino, teste e validação de forma estratificada
    train, test_val = train_test_split(images, shuffle=True, random_state=123, test_size=(1 - train_ratio), stratify=[disease] * len(images))
    test, val = train_test_split(test_val, shuffle=True, random_state=123, test_size=val_ratio / (test_ratio + val_ratio), stratify=[disease] * len(test_val))

    # Adiciona os caminhos de arquivos e etiquetas às listas correspondentes
    for image in train:
        train_filepaths.append(os.path.join(source_dir, image))
        train_labels.append(disease)
    for image in test:
        test_filepaths.append(os.path.join(source_dir, image))
        test_labels.append(disease)
    for image in val:
        val_filepaths.append(os.path.join(source_dir, image))
        val_labels.append(disease)

print("Criação das listas e divisão concluídas.")
print(len(train_filepaths))  # 80% do total de imagens
print(len(val_filepaths))    # 10% do total de imagens
print(len(test_filepaths))   # 10% do total de imagens

# Cria DataFrames para cada conjunto
train_data = {'filepaths': train_filepaths, 'labels': train_labels}
train_images = pd.DataFrame(train_data)

val_data = {'filepaths': val_filepaths, 'labels': val_labels}
val_images = pd.DataFrame(val_data)

test_data = {'filepaths': test_filepaths, 'labels': test_labels}
test_images = pd.DataFrame(test_data)

#Generate Images bellow

def create_data_generators(train_images, val_images, test_images, image_size=(112, 112), batch_size=32):
    input_shape = image_size + (3,)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_images,
        x_col='filepaths',
        y_col='labels',
        target_size=image_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_images,
        x_col='filepaths',
        y_col='labels',
        target_size=image_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False,
    )

    test_batch_size = min(80, len(test_images))

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_images,
        x_col='filepaths',
        y_col='labels',
        target_size=image_size,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_generator, val_generator, test_generator

# Chamar a função com a ordem correta dos parâmetros
train_generator, val_generator, test_generator = create_data_generators(train_images, val_images, test_images)

#------------- Show some images from data generator -------------#

def show_images_with_labels(generator, num_images=25):
    x, y = next(generator)
    class_labels = list(generator.class_indices.keys())

    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(x))):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x[i])
        predicted_label = class_labels[np.argmax(y[i])]
        plt.title(f"Label: {predicted_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_images_with_labels(train_generator)
'''
Label 1: Meningioma;
Label 2: Glioma;
Label 3: Pituitary Tumor
'''

#----------- Callbacks ------------#

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, patience, stop_patience, accuracy_threshold, learning_rate_factor, batches, epochs, ask_epoch):
        super(CustomCallback, self).__init__()
        self.model = model
        self.patience = patience
        self.stop_patience = stop_patience
        self.accuracy_threshold = accuracy_threshold
        self.learning_rate_factor = learning_rate_factor
        self.batches = batches
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.best_weights = None
        self.best_loss = float('inf')
        self.wait = 0

    def on_train_begin(self, logs=None):
        print("Training has started.")
        print(f"Number of epochs: {self.epochs}")
        print(f"Batch size: {self.batches}")
        print(f"Learning rate: {tf.keras.backend.get_value(self.model.optimizer.lr)}")
        if self.model.optimizer:
            print(f"Learnig rate: {self.model.optimizer.learning_rate.numpy()}")
        else:
            print("Warning: Optimizer not yet compiled.")

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        current_accuracy = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')

        print(f"\nEpoch {epoch + 1}/{self.epochs}")
        print(f"Training Loss: {current_loss:.4f} | Training Accuracy: {current_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")

        improvement = (self.best_loss - val_loss) / self.best_loss * 100
        print(f"Loss Improvement: {improvement:.2f}%")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_learning_rate = learning_rate * self.learning_rate_factor
                tf.keras.backend.set_value(self.model.optimizer.lr, new_learning_rate)
                print(f"Learning Rate reduced to {new_learning_rate}")
                self.wait = 0

        if epoch + 1 == self.ask_epoch:
            user_input = input("Do you want to continue training? (yes/no): ")
            if user_input.lower() != 'yes':
                self.model.set_weights(self.best_weights)
                self.model.stop_training = False
                print("Training interrupted by user input.")

        if self.wait >= self.stop_patience:
            self.model.set_weights(self.best_weights)
            self.model.stop_training = True
            print(f"Training stopped due to no improvement for {self.stop_patience} epochs.")

        epoch_duration = time.time() - self.start_time
        print(f"Epoch Duration: {epoch_duration:.2f} seconds")

    def on_train_end(self, logs=None):
        total_duration = time.time() - self.start_time
        print(f"Total Training Duration: {total_duration:.2f} seconds")
        print("Training has ended.")

#--------- Training History ---------#

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Verificando se val_loss e val_accuracy não estão vazios antes de usar index()
    if val_loss and val_accuracy:
        best_epoch = val_loss.index(min(val_loss)) + 1
        best_accuracy_epoch = val_accuracy.index(max(val_accuracy)) + 1

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(loss) + 1), loss, label='Treinamento')
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validação')
        plt.scatter(best_epoch, min(val_loss), color='red', label=f'Melhor Época ({best_epoch})', zorder=5)
        plt.annotate(f"{min(val_loss):.4f}", (best_epoch, min(val_loss)), textcoords="offset points", xytext=(0,10), ha='center', color='red')
        plt.xlabel('Época')
        plt.ylabel('Perda')
        plt.legend()
        plt.title('Histórico de Perda')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(accuracy) + 1), accuracy, label='Treinamento')
        plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validação')
        plt.scatter(best_accuracy_epoch, max(val_accuracy), color='green', label=f'Melhor Época ({best_accuracy_epoch})', zorder=5)
        plt.annotate(f"{max(val_accuracy):.4f}", (best_accuracy_epoch, max(val_accuracy)), textcoords="offset points", xytext=(0,10), ha='center', color='green')
        plt.xlabel('Época')
        plt.ylabel('Precisão')
        plt.legend()
        plt.title('Histórico de Precisão')

        plt.tight_layout()
        plt.show()
    else:
        print("Erro: Os dados de validação não estão presentes no histórico.")

#Data reading
try:
    batch_size = 40
    train_generator, test_generator, val_generator = create_data_generators(train_images, test_images, val_images)
    print("Data generators criados com sucesso.")
except NameError as ne:
    print(f"Erro de nome: {ne}. Verifique se todas as variáveis estão definidas corretamente.")
except SyntaxError as se:
    print(f"Erro de sintaxe: {se}. Verifique a sintaxe do seu código.")
except FileNotFoundError as fnfe:
    print(f"Erro de arquivo não encontrado: {fnfe}. Verifique os caminhos dos arquivos.")
except Exception as e:
    print(f"Erro inesperado: {e}. Detalhes do erro:", type(e).__name__, e)

#--------- Creating Xception Pre-trained Model ------------#

img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(train_generator.class_indices)

base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_shape=img_shape,
    pooling='max'
)

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=l2(0.016), activity_regularizer=l1(0.006),
          bias_regularizer=l1(0.006), activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(class_count, activation='softmax')
])

# Otimizador com taxa de aprendizado mais agressiva
opt = keras.optimizers.Adam(learning_rate=1e-4)

# model.compile(Adamax(learning_rate=0.004), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])#

print(base_model.output_shape)

model.summary()

#Callbacks Parameters

def on_train_begin(self, logs=None):
    tf.keras.backend.set_value(self.model.optimizer.lr, 1e-4)
    print("Training has started.")
    print(f"Initial Learning rate: {tf.keras.backend.get_value(self.model.optimizer.lr)}")

batch_size = 40
epochs = 40
patience = 1
stop_patience = 3
threshold = 0.9
factor = 0.5
ask_epoch = 5
batches = int(np.ceil(len(train_generator.labels) / batch_size))

# Custom Callback
custom_callback = CustomCallback(
    model=model,
    patience=patience,
    stop_patience=stop_patience,
    learning_rate_factor=factor,
    batches=batches,
    epochs=epochs,
    ask_epoch=ask_epoch,
    accuracy_threshold=threshold
)

# ReduceLROnPlateau Callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [custom_callback, reduce_lr]

#Training Models
history = model.fit(
    x=train_generator,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=None,
    shuffle=False
)

#Evaluete Model

def calculate_custom_batch_size(test_data_length, max_batch_size=80):
    """
    Calcula o tamanho de lote personalizado com base no comprimento dos dados de teste.

    Args:
    - test_data_length (int): Comprimento dos dados de teste.
    - max_batch_size (int, opcional): Tamanho máximo do lote. Padrão é 80.

    Returns:
    - custom_batch_size (int): Tamanho do lote personalizado calculado.
    """
    for divisor in range(1, test_data_length + 1):
        if test_data_length % divisor == 0 and test_data_length // divisor <= max_batch_size:
            return test_data_length // divisor

    return 1  # Valor de fallback, caso nenhum divisor adequado seja encontrado

def evaluate_model_with_generator(model, data_generator, name):
    """
    Avalia o modelo usando um gerador de dados e imprime a perda e a precisão.

    Args:
    - model (keras.Model): O modelo Keras a ser avaliado.
    - data_generator (keras.preprocessing.image.ImageDataGenerator): O gerador de dados.
    - name (str): O nome do conjunto de dados sendo avaliado (para exibição).
    """
    loss, accuracy = model.evaluate(data_generator, verbose=0)
    print(f"{name} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

# Calcular o tamanho de lote personalizado com base no número de imagens de teste
custom_batch_size = calculate_custom_batch_size(len(test_images))

# Avaliar o modelo nos conjuntos de treinamento, validação e teste
evaluate_model_with_generator(model, train_generator, "Treinamento")
evaluate_model_with_generator(model, val_generator, "Validação")
evaluate_model_with_generator(model, test_generator, "Teste (Custom Batch Size)")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Assuming you have your test data and predicted labels
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, classes=['1', '2', '3'])
plt.show()