# import system libs
import os
import time
import shutil
import pathlib
import itertools

import cv2
import matplotlib
import numpy as np
from lime import lime_image
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
from keras.models import Sequential, Model
from keras.optimizers import Adam, Adamax
from keras.losses import categorical_crossentropy
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.normalization.batch_normalization import (
    BatchNormalization,
)
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1, l2
from keras.utils import img_to_array, load_img
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')

# Diretório onde estão as subpastas
base_dir = '/home/ubuntu/Documents/tcc2_brain_tumor_classification/dataset'

# # Diretório da classe 1
# class1_dir = 'dataset/1'

# # Diretório de saída para imagens aumentadas da Classe 1
# output_dir = 'dataset/1_augmented'
# os.makedirs(output_dir, exist_ok=True)

# # Configurando o ImageDataGenerator
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Aumentar imagens da Classe 1
# for image_path in os.listdir(class1_dir):
#     img = load_img(os.path.join(class1_dir, image_path))  # Carregar imagem
#     img_array = img_to_array(img)  # Converter para array NumPy
#     img_array = img_array.reshape((1,) + img_array.shape)  # Redimensionar

#     # Gerar até 3 imagens aumentadas para cada imagem original
#     generated_count = 0
#     for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
#         generated_count += 1
#         if generated_count >= 3:
#             break


# Lista das classes (doenças)
classes = ['1_augmented', '2', '3']

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

# Diretório da classe 1
class1_dir = 'dataset/1'

# Diretório de saída para imagens aumentadas da Classe 1
output_dir = 'dataset/1_augmented'
os.makedirs(output_dir, exist_ok=True)

# Configurando o ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Aumentar imagens da Classe 1
for image_path in os.listdir(class1_dir):
    img = load_img(os.path.join(class1_dir, image_path))  # Carregar imagem
    img_array = img_to_array(img)  # Converter para array NumPy
    img_array = img_array.reshape((1,) + img_array.shape)  # Redimensionar

    # Gerar até 3 imagens aumentadas para cada imagem original
    generated_count = 0
    for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
        generated_count += 1
        if generated_count >= 3:
            break


# Pre-processing images
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

def show_images_with_labels(generator, num_images=25):
    x, y = next(generator)
    class_labels = list(generator.class_indices.keys())

    plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(x))):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x[i])
        predicted_label = class_labels[np.argmax(y[i])]
        plt.title(f"Class: {predicted_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("output_image.png")
    plt.close()

show_images_with_labels(train_generator)

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

def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

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

#Creating Xception Pre-trained Model
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

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])#

print(base_model.output_shape)

model.summary()

#Callbacks Parameters
def on_train_begin(self, logs=None):
    tf.keras.backend.set_value(self.model.optimizer.lr, 1e-4)
    print("Training has started.")
    print(f"Initial Learning rate: {tf.keras.backend.get_value(self.model.optimizer.lr)}")

batch_size = 48 #Optimizing batch for GPU
epochs = 40
patience = 1
stop_patience = 3
threshold = 0.9
factor = 0.5
ask_epoch = 5
batches = int(np.ceil(len(train_generator.labels) / batch_size))

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

def find_penultimate_layer(model):
    """
    Retorna o nome da última camada convolucional no modelo.
    """
    for layer in reversed(model.layers):
        if 'conv' in layer.name or 'Conv2D' in str(layer):
            print(f"Penultimate Conv Layer: {layer.name}")
            return layer.name
    raise ValueError("Nenhuma camada convolucional encontrada no modelo base.")

def create_submodel_for_gradcam(full_model, target_layer_name):
    """
    Cria um submodelo funcional conectando as entradas da Xception à camada convolucional alvo.
    """
    # Acessa o base_model (Xception)
    base_model = full_model.layers[0]  

    # Obtém a saída da camada convolucional de interesse
    penultimate_layer_output = base_model.get_layer(target_layer_name).output  

    # Retorna um submodelo funcional
    return Model(inputs=base_model.input, outputs=penultimate_layer_output)


def apply_gradcam_batch(full_model, image_paths, class_index, penultimate_layer_name, batch_size=5, save_path=None):
    """
    Aplica Grad-CAM em lotes de imagens e salva ou exibe os resultados.

    Args:
        full_model: Modelo completo treinado.
        image_paths: Lista com os caminhos das imagens.
        class_index: Índice da classe de interesse.
        penultimate_layer_name: Nome da camada convolucional.
        batch_size: Número de imagens por lote.
        save_path: Caminho para salvar a figura (opcional).
    """
    # Cria o submodelo funcional
    submodel = create_submodel_for_gradcam(full_model, penultimate_layer_name)

    # Configura Grad-CAM no submodelo
    gradcam = Gradcam(submodel, clone=False)
    score = CategoricalScore([class_index])

    total_images = len(image_paths)
    total_batches = (total_images + batch_size - 1) // batch_size

    fig, axes = plt.subplots(total_images, 2, figsize=(10, 5 * total_images))
    axes = np.array(axes).reshape(-1, 2)

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_images)
        batch_paths = image_paths[batch_start:batch_end]

        batch_images = []
        for img_path in batch_paths:
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            batch_images.append(img_array)

        batch_images = np.array(batch_images)

        # Calcula o Grad-CAM para o lote atual
        heatmaps = gradcam(score, batch_images, penultimate_layer=penultimate_layer_name)

        for i, (img_path, heatmap) in enumerate(zip(batch_paths, heatmaps)):
            img_idx = batch_start + i
            img = load_img(img_path, target_size=(224, 224))

            axes[img_idx, 0].imshow(img)
            axes[img_idx, 0].set_title("Input Image")
            axes[img_idx, 0].axis('off')

            axes[img_idx, 1].imshow(img)
            axes[img_idx, 1].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[img_idx, 1].set_title("Grad-CAM Heatmap")
            axes[img_idx, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Grad-CAMs salvos em: {save_path}")
    else:
        plt.show()
        plt.close()

image_paths = [
    'dataset/1_augmented/aug_0_3.jpeg',
    'dataset/1_augmented/aug_0_122.jpeg',
    'dataset/1/2343.png',
    'dataset/2/619.png',
    'dataset/3/993.png'
]

apply_gradcam_batch(
    full_model=model,
    image_paths=image_paths,
    class_index=1,  # Classe de interesse
    penultimate_layer_name='block14_sepconv2_act',
    batch_size=5,  # Lotes de 3 imagens
    save_path="gradcam_batch_output.png"
)

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

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Gera e salva a matriz de confusão como uma imagem separada.

    Args:
        cm: Matriz de confusão já calculada.
        class_names: Lista com os nomes das classes.
        save_path: Caminho para salvar a matriz de confusão.
    """

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")

# Predição dos dados de teste
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Exibição da matriz de confusão
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, class_names=['1', '2', '3'], save_path="output_confusion_matrix.png")

def calculate_metrics(y_true, y_pred, class_names):
    """
    Calcula e exibe métricas de precisão, recall e F1-score.

    Args:
        y_true: Rótulos reais.
        y_pred: Rótulos preditos pelo modelo.
        class_names: Lista com os nomes das classes.
    """

    # Relatório detalhado por classe
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Cálculo das métricas globais
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("\nGlobal Metrics:")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1-score (Weighted): {f1:.4f}")

# Cálculo de métricas
calculate_metrics(test_generator.classes, y_pred, class_names=['1', '2', '3'])