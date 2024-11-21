import React, { useState } from 'react';
import * as tf from '@tensorflow/tfjs';

const Upload = () => {
  const [model, setModel] = useState(null); // State for storing the model
  const [message, setMessage] = useState('');
  const [selectedImage, setSelectedImage] = useState(null); // State for image
  const [prediction, setPrediction] = useState(''); // State for prediction

  // Load TensorFlow.js model
  const loadModel = async () => {
    try {
      const loadedModel = await tf.loadLayersModel('/models/model.json'); // Path to your TensorFlow.js model
      setModel(loadedModel);
      setMessage('Model loaded successfully!');
      console.log('Model loaded:', loadedModel);
    } catch (error) {
      setMessage('Error loading model. Check console for details.');
      console.error('Error loading model:', error);
    }
  };

  // Handle image upload
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => setSelectedImage(reader.result); // Save image as base64
      reader.readAsDataURL(file);
    }
  };

  const predictImage = async () => {
    if (!model || !selectedImage) {
      alert('Please upload an image and ensure the model is loaded.');
      return;
    }
  
    const imgElement = document.getElementById('uploaded-image');
  
    try {
      // Preprocess the image
      const tensor = tf.browser
        .fromPixels(imgElement)
        .resizeNearestNeighbor([224, 224]) // Resize to 224x224
        .expandDims() // Add batch dimension
        .toFloat()
        .div(255.0); // Normalize pixel values
  
      // Perform the prediction
      const predictions = model.predict(tensor).arraySync(); // Get prediction probabilities
      console.log('Predictions:', predictions);
  
      // Extract the first element (batch size of 1)
      const predictionArray = predictions[0]; // Flatten the batch
      console.log('Flattened Predictions:', predictionArray);
  
      // Extract class and confidence
      const predictedClassIndex = predictionArray.indexOf(Math.max(...predictionArray)); // Get class index
      const confidence = Math.max(...predictionArray); // Get highest probability
  
      // Define class labels (use your Python-trained labels)
      const classLabels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];
  
      // Get the label for the predicted class
      const predictedClassLabel = classLabels[predictedClassIndex] || 'Unknown';
  
      // Update the state with the prediction
      setPrediction(`Prediction: ${predictedClassLabel} (Confidence: ${(confidence * 100).toFixed(2)}%)`);
    } catch (error) {
      console.error('Error during prediction:', error);
      setPrediction('Prediction error. Check console for details.');
    }
  };
  

  return (
    <div>
    <h2>AI Skin Infection Detection</h2>
    <button onClick={loadModel}>Load Model</button>
    <p>{message}</p>
    <input type="file" accept="image/*" onChange={handleImageUpload} />
    {selectedImage && (
      <div>
        <img
          id="uploaded-image"
          src={selectedImage}
          alt="Uploaded Preview"
          style={{ maxWidth: '100%', height: 'auto', margin: '20px 0' }}
        />
      </div>
    )}
    <button onClick={predictImage} style={{ marginTop: '10px' }}>Predict</button>
    {prediction && (
      <div style={{ marginTop: '20px', padding: '10px', backgroundColor: '#e6f7ff', border: '1px solid #91d5ff' }}>
        <h3 style={{ color: '#096dd9' }}>{prediction}</h3>
      </div>
    )}
  </div>
  
  );
};

export default Upload;
