import numpy as np
import matplotlib.pyplot as plt

# Retourne la synthèse d'un modèle sous la forme d'un dictionnaire
def add_model_in_synthese(loss_type, loss_score, mean_iou_score, training_time, predict_time):
    return {
            "loss type": loss_type,
            "loss score": loss_score,
            "mean_iou": mean_iou_score,
            "Training time": training_time,
            "Predict time": predict_time
        }

# Affiche graphiquement l'évolution de la fonction loss et de la métrique lors de l'entraînement d'un modèle
def draw_history(history):
    plt.subplots(1, 2, figsize=(15,4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['mean_iou'])
    plt.plot(history['val_mean_iou'])
    plt.title('model mean_iou')
    plt.ylabel('mean_iou')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.show()