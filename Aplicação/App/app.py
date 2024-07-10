import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tkinter as tk
from tkinter import filedialog, Label, ttk, Canvas, Frame
from PIL import Image, ImageTk
import numpy as np
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import importlib

# Function to display versions of imported libraries
def show_versions():
    libraries = ['tensorflow', 'PIL', 'numpy', 'matplotlib', 'networkx']
    versions = {}
    for lib in libraries:
        try:
            module = importlib.import_module(lib)
            if lib == 'PIL':
                version = module.__version__
            else:
                version = module.__version__
            versions[lib] = version
        except ImportError:
            versions[lib] = 'Not Installed'
    for lib, version in versions.items():
        print(f"{lib}: {version}")

# Show library versions at startup
show_versions()

# Classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Global variables for the model
model = None
image_size = (32, 32)
preprocess_input = None
base_model = None
final_model = None

# Dictionary to hold the accuracy percentages for the models
model_accuracies = {
    'TransferLearning_Sem_DataAugmentation.h5': 85.2,
    'TransferLearning_Com_DataAugmentation.h5': 87.6,
    'From_Scratch_Sem_DataAugmentation.h5': 84.3,
    'From_Scratch_Com_DataAugmentation.h5': 85.2
}

def default_preprocess_input(img_array):
    return img_array

def load_base_model(model_name):
    global preprocess_input, image_size
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False)
        preprocess_input = vgg_preprocess
        image_size = (224, 224)
    elif model_name == 'VGG16Agu':
        base_model = None
        preprocess_input = default_preprocess_input
        image_size = (150, 150)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
        preprocess_input = resnet_preprocess
        image_size = (224, 224)
    else:
        base_model = None
        preprocess_input = default_preprocess_input
        image_size = (32, 32)
    
    if base_model:
        model = Model(inputs=base_model.input, outputs=base_model.output)
        return model, preprocess_input, image_size
    else:
        return None, preprocess_input, image_size

def model_predict(img_path, model, preprocess_input, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    if model:
        features = model.predict(img_array)
        if features.shape[1:3] != (4, 4):
            features = tf.image.resize(features, (4, 4))
    else:
        features = img_array
    
    preds = final_model.predict(features)
    return preds

def lazy_load_visualization_modules():
    global nx, plt
    import networkx as nx
    import matplotlib.pyplot as plt

def get_layer_size(layer):
    if isinstance(layer, tf.keras.layers.InputLayer):
        return layer.input_shape[0][1]
    elif hasattr(layer, 'units'):
        return layer.units
    elif hasattr(layer, 'output_shape'):
        if isinstance(layer.output_shape, list):
            return layer.output_shape[0][1]
        else:
            return layer.output_shape[1]
    else:
        return 0

def visualize_network(preds, input_image):
    lazy_load_visualization_modules()
    np.random.seed(42)  # Set seed for reproducibility

    fig, ax = plt.subplots(figsize=(10, 8))

    G = nx.DiGraph()
    layers = [layer for layer in final_model.layers if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.Dense))]
    layer_sizes = [get_layer_size(layer) for layer in layers]
    node_counts = [min(size, 32) for size in layer_sizes]

    nodes = []
    for i, count in enumerate(node_counts):
        nodes.append([f"Layer {i} Neuron {j}" for j in range(count)])

    for layer in nodes:
        G.add_nodes_from(layer)

    for i in range(len(nodes) - 1):
        for u in nodes[i]:
            for v in nodes[i + 1]:
                if np.random.rand() > 0.7:
                    G.add_edge(u, v, weight=np.random.randn())

    pos = {}
    layer_sep = 3.0
    node_sep = 20.0

    y_offsets = []
    max_nodes = max(len(layer) for layer in nodes)
    for layer in nodes:
        y_offsets.append((max_nodes - len(layer)) / 2.0 * node_sep)

    for i, layer in enumerate(nodes):
        for j, node in enumerate(layer):
            pos[node] = (i * layer_sep, j * node_sep + y_offsets[i])

    for j, node in enumerate(nodes[0]):
        pos[node] = (-2.0, j * node_sep)

    last_layer = nodes[-1]
    class_y_offset = (max_nodes - len(last_layer)) / 2.0 * node_sep
    for j, node in enumerate(last_layer):
        pos[node] = (layer_sep * len(nodes), j * node_sep + class_y_offset)

    edges = G.edges(data=True)
    edge_colors = ['red' for (u, v, d) in edges]
    edge_weights = [abs(d['weight']) for (u, v, d) in edges]

    nx.draw(G, pos, with_labels=False, node_size=200, ax=ax, edge_color=edge_colors, width=edge_weights)

    input_labels = {node: f"Input {j}" for j, node in enumerate(nodes[0])}
    class_labels = {node: classes[j] for j, node in enumerate(nodes[-1])}

    nx.draw_networkx_labels(G, pos, labels=input_labels, ax=ax, verticalalignment='center', horizontalalignment='right', font_size=10)
    nx.draw_networkx_labels(G, pos, labels=class_labels, ax=ax, verticalalignment='center', horizontalalignment='left', font_size=10)

    plt.title('Network Visualization')
    plt.tight_layout()

    network_img_path = 'static/network.png'
    plt.savefig(network_img_path)
    plt.close()

    return network_img_path

def visualize_predictions(preds):
    lazy_load_visualization_modules()
    fig, ax = plt.subplots(figsize=(6, 8))

    y_pos = np.arange(len(classes))
    performance = preds[0]

    ax.barh(y_pos, performance, align='center', color='blue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')

    plt.tight_layout()

    predictions_img_path = 'static/predictions.png'
    plt.savefig(predictions_img_path)
    plt.close()

    return predictions_img_path

def upload_image():
    def load_and_visualize():
        file_path = filedialog.askopenfilename()
        if file_path:
            if 'frame_img_progress' in globals() and frame_img_progress.winfo_exists():
                frame_img_progress.destroy()
            
            frame_img_progress = tk.Frame(window)
            frame_img_progress.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky='w')

            progress.grid(row=3, column=0, padx=10, pady=10, sticky='w')
            progress.start()

            img = Image.open(file_path)
            img = img.resize(image_size, Image.LANCZOS)
            img = ImageTk.PhotoImage(img)

            panel.config(image=img)
            panel.image = img

            preds = model_predict(file_path, base_model, preprocess_input, image_size)
            prediction = np.argmax(preds, axis=1)[0]
            result_label.config(text=f"Prediction: {classes[prediction]}")

            network_img_path = visualize_network(preds, img)
            img_network = Image.open(network_img_path)
            img_network = ImageTk.PhotoImage(img_network)
            network_panel.config(image=img_network)
            network_panel.image = img_network

            predictions_img_path = visualize_predictions(preds)
            img_predictions = Image.open(predictions_img_path)
            img_predictions = ImageTk.PhotoImage(img_predictions)
            predictions_panel.config(image=img_predictions)
            predictions_panel.image = img_predictions

            progress.stop()
            progress.grid_remove()

    threading.Thread(target=load_and_visualize).start()

def select_model():
    model_choice = model_var.get()
    augmentation = augmentation_var.get()
    
    global base_model, preprocess_input, image_size, final_model
    
    if model_choice == 'Transfer Learning' and augmentation == 'Sem Data Augmentation':
        model_path = 'TransferLearning_Sem_DataAugmentation.h5'
        base_model, preprocess_input, image_size = load_base_model('VGG16')
    elif model_choice == 'Transfer Learning' and augmentation == 'Com Data Augmentation':
        model_path = 'TransferLearning_Com_DataAugmentation.h5'
        base_model, preprocess_input, image_size = load_base_model('VGG16Agu')
    elif model_choice == 'From Scratch' and augmentation == 'Sem Data Augmentation':
        model_path = 'From_Scratch_Sem_DataAugmentation.h5'
        base_model, preprocess_input, image_size = load_base_model('FromScratch')
    elif model_choice == 'From Scratch' and augmentation == 'Com Data Augmentation':
        model_path = 'From_Scratch_Com_DataAugmentation.h5'
        base_model, preprocess_input, image_size = load_base_model('FromScratch')
    
    threading.Thread(target=lambda: load_final_model(model_path)).start()

    accuracy = model_accuracies.get(model_path, "Unknown")
    accuracy_label.config(text=f"Model Accuracy: {accuracy}%")

    upload_button.grid(row=2, column=0, padx=10, pady=10)

def load_final_model(model_path):
    global final_model
    final_model = load_model(model_path)

# GUI Configuration
window = tk.Tk()
window.title("Image Classifier")
window.geometry("1200x800")
window.state('zoomed')  # Start maximized

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12, 'bold'), foreground='blue')
style.configure('TLabel', font=('Helvetica', 12), padding=10)
style.configure('TProgressbar', thickness=30, background='light blue', troughcolor='gray')

model_var = tk.StringVar()
model_var.set("Transfer Learning")

augmentation_var = tk.StringVar()
augmentation_var.set("Sem Data Augmentation")

model_label = ttk.Label(window, text="Choose the model:", font=('Helvetica', 12))
model_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

model_menu = ttk.OptionMenu(window, model_var, "Transfer Learning", "Transfer Learning", "From Scratch")
model_menu.grid(row=0, column=1, padx=10, pady=10, sticky='w')

augmentation_label = ttk.Label(window, text="Choose Data Augmentation:", font=('Helvetica', 12))
augmentation_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

augmentation_menu = ttk.OptionMenu(window, augmentation_var, "Sem Data Augmentation", "Sem Data Augmentation", "Com Data Augmentation")
augmentation_menu.grid(row=1, column=1, padx=10, pady=10, sticky='w')

select_button = ttk.Button(window, text="Select Model", command=select_model, style='TButton')
select_button.grid(row=2, column=1, padx=10, pady=10, sticky='w')

upload_button = ttk.Button(window, text="Upload Image", command=upload_image, style='TButton')
upload_button.grid(row=2, column=0, padx=10, pady=10, sticky='w')

result_label = tk.Label(window, text="Prediction: ", font=('Helvetica', 12))
result_label.grid(row=3, column=1, padx=10, pady=10, sticky='w')

accuracy_label = tk.Label(window, text="Model Accuracy: ", font=('Helvetica', 12))
accuracy_label.grid(row=4, column=1, padx=10, pady=10, sticky='w')

progress = ttk.Progressbar(window, orient=tk.HORIZONTAL, length=300, mode='indeterminate', style='TProgressbar')

canvas = Canvas(window)
scroll_x = ttk.Scrollbar(window, orient="horizontal", command=canvas.xview)
scroll_y = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
canvas.configure(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)

frame = Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

canvas.grid(row=5, column=0, columnspan=4, sticky='nsew')
scroll_x.grid(row=6, column=0, columnspan=4, sticky='ew')
scroll_y.grid(row=5, column=4, sticky='ns')

window.grid_rowconfigure(5, weight=1)
window.grid_columnconfigure(1, weight=1)

def configure_canvas(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", configure_canvas)

panel = Label(frame)
panel.grid(row=0, column=0, padx=5, pady=5)

network_panel = Label(frame)
network_panel.grid(row=5, column=0, columnspan=1)

predictions_panel = Label(frame)
predictions_panel.grid(row=5, column=1, columnspan=1)

window.mainloop()
