import express from "express";
import cors from "cors";
import multer from "multer";
import admin from "firebase-admin";
import { loadModel, classifyImage } from "./modelService";

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.use(cors());
app.use(express.json());
app.use("/model", express.static("public/model"));


// Initialize Firebase (optional — skip for now)
try {
  admin.initializeApp();
  console.log("✅ Firebase initialized");
} catch (err) {
  console.warn("⚠️ Firebase not configured yet");
}

// Load TensorFlow model once at server start
loadModel().catch(err => console.error("❌ Model load error:", err));

app.get("/", (req, res) => res.send("🌽 CropEye backend is running!"));

// Image classification endpoint
app.post("/api/classify", upload.single("image"), async (req: any, res) => {
  try {
    if (!req.file) return res.status(400).send("No image uploaded.");

    const result = await classifyImage(req.file.buffer);
    res.json(result);
  } catch (err) {
    console.error("❌ Classification error:", err);
    res.status(500).send("Error processing image.");
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
