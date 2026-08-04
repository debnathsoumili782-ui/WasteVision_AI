import tensorflow as tf

model = tf.keras.models.load_model("model/biowaste_best_model.h5")

model.save("model/biowaste_best_model.keras")