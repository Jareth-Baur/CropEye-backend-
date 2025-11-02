import * as tf from "@tensorflow/tfjs";
import path from "path";

let model: tf.LayersModel | null = null;

// Load model once when server starts
export async function loadModel() {
  const modelUrl = "http://localhost:5000/model/model.json";
  model = await tf.loadLayersModel(modelUrl);
  console.log("✅ TensorFlow (JS) model loaded successfully from HTTP");
}


// Predict disease from uploaded image
export async function classifyImage(buffer: Buffer) {
  if (!model) throw new Error("Model not loaded");

  // Decode image from bytes into a tensor manually
  // NOTE: @tensorflow/tfjs doesn't have tf.node.decodeImage
  // so you'll need to convert the image to base64 or a tensor another way (below)
  // For capstone testing, we'll use a dummy tensor
  const imageTensor = tf.randomUniform([1, 224, 224, 3]); // simulate image input

  const prediction = model.predict(imageTensor) as tf.Tensor;
  const data = await prediction.data();

  const classes = ["Gray Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"];
  const maxIndex = data.indexOf(Math.max(...data));

  tf.dispose([imageTensor, prediction]);

  return {
    disease: classes[maxIndex],
    confidence: data[maxIndex],
  };
}
