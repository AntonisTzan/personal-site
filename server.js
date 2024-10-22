// server.js
const express = require('express');
const bodyParser = require('body-parser');
const tf = require('@tensorflow/tfjs-node'); // For TensorFlow
const app = express();
const port = 3000;

// Middleware to parse JSON bodies
app.use(bodyParser.json());
app.use(express.static('public')); // Serve static files from public directory

// Load your TFLite model
let model;
(async () => {
    model = await tf.loadGraphModel('file://model.tflite');
})();

// Route for classification
app.post('/classify', async (req, res) => {
    const inputText = req.body.input;

    // Tokenization (handled in the front-end)
    // Pass the tokenized input to the model for classification
    // Dummy output for demonstration purposes
    const predictions = await model.predict(/* your input here */);
    
    // Logic for redirect based on predictions
    const probDiff = predictions[0][0] - predictions[0][1];

    if (probDiff > 0.1) {
        res.json({ redirect: 'https://your-site.com/page1' });
    } else if (probDiff < -0.1) {
        res.json({ redirect: 'https://your-site.com/page2' });
    } else {
        res.json({ redirect: 'https://your-site.com/page3' });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});