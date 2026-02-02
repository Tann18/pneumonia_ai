import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# ===============================
# REBUILD MODEL ARCHITECTURE
# ===============================
base = tf.keras.applications.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    include_top=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dense(256, activation="relu")(x)
output = tf.keras.layers.Dense(3, activation="softmax")(x)

model = tf.keras.Model(inputs=base.input, outputs=output)

# ===============================
# LOAD WEIGHTS ONLY
# ===============================
model.load_weights("best_pneumonia_model.h5")

# ===============================
# SAVE CLEAN MODEL
# ===============================
model.save("best_pneumonia_model.keras")

print("✅ SUCCESS: Clean model rebuilt and saved")