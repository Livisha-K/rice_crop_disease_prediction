<!DOCTYPE html>
<html>

<body>
<h1>AgroLeafGuard: Rice Crop Disease Prediction System</h1>

<h2>Rice Leaf Diseases Classification with Convolutional Neural Network (CNN)</h2>

<h2>Overview</h2>

<p>This project utilizes a Convolutional Neural Network (CNN) to classify rice leaf diseases based on images. The dataset consists of images of rice leaves affected by bacterial leaf blight, brown spot, and leaf smut. The CNN model is trained on this dataset to make predictions.</p>

<h2>Requirements</h2>

<ul>
    <li>Python 3.x</li>
    <li>TensorFlow</li>
    <li>scikit-learn</li>
    <li>OpenCV</li>
    <li>NumPy</li>
    <li>Matplotlib</li>
</ul>

<h2>Setup</h2>

</li>
    <li>Clone the repository:

   </li> git clone https://github.com/Livisha-K/rice_crop_disease_prediction.git

<li>Install the required dependencies:</li>

pip install tensorflow scikit-learn opencv-python numpy matplotlib

<li>Download the dataset and place it in the specified directory <code>C:\Deep Learning\CNN\rice_leaf_diseases</code>. Ensure the dataset structure follows the format:</li>
</li> 
<code>C:\Deep Learning\CNN\rice_leaf_diseases
├── Bacterial leaf blight
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Brown spot
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Leaf smut
    ├── image1.jpg
    ├── image2.jpg
    └── ...
</code>
</ol>

<h2>Usage</h2>

<ol>
    <li>Run the script <code>rice_leaf_diseases.ipynb</code> to train the CNN model:</li>

<p><code>rice_leaf_diseases.ipynb</code>This will train the model using the dataset and display training/validation metrics.</p>

<li>After training, you can use the trained model to make predictions on new images:</li>

<p><code>rice_leaf_diseases.ipynb</code>This will load a sample image (<code>t1.JPG</code>) from the specified path and display the predicted disease.</p>
</ol>

<h2>Model Architecture</h2>

<p>The CNN model architecture consists of convolutional layers with max-pooling, followed by densely connected layers. The final layer uses the softmax activation function for multi-class classification.</p>


<code>Model Architecture:
    ___________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 25)        250
    ___________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 25)        0
    ___________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 15)        3390
    ___________________________________________________________
    flatten (Flatten)            (None, 1815)              0
    ___________________________________________________________
    dense (Dense)                (None, 30)                54480
    ___________________________________________________________
    dense_1 (Dense)              (None, 3)                 93
    =================================================================
    Total params: 58,213
    Trainable params: 58,213
    Non-trainable params: 0
</code>

<h2>Results</h2>

<p>The model achieves an accuracy of approximately 88.16% on the validation set after 15 epochs.</p>

<h2>Author</h2>

<p>Livisha K</p>

</body>

</html>
