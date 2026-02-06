import { useState, ChangeEvent } from "react";

interface PredictionResult {
  prediction: string;
  confidence: number;
  error?: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const API_BASE_URL =
    import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload an MRI image");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data: PredictionResult = await response.json();

      // ✅ CLAMP confidence to max 100%
      data.confidence = Math.min(data.confidence, 100);

      setResult(data);
    } catch (err) {
      setResult({
        error: "Failed to get prediction",
        prediction: "",
        confidence: 0,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animated-background">
      <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-xl">
        <h1 className="text-3xl font-bold text-center text-blue-700 mb-6">
          Alzheimer’s Early Detection
        </h1>

        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="mb-4 w-full"
        />

        {preview && (
          <img
            src={preview}
            alt="MRI Preview"
            className="w-full h-64 object-contain border rounded-lg mb-4"
          />
        )}

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Predict"}
        </button>

        {result && !result.error && (
          <div className="mt-6 p-4 border rounded-lg bg-blue-50">
            <h3 className="text-xl font-semibold text-blue-800 mb-2">
              Prediction Result
            </h3>
            <p>
              <b>Stage:</b> {result.prediction}
            </p>
            <p>
              <b>Confidence:</b>{" "}
              {result.confidence.toFixed(2)}%
            </p>
          </div>
        )}

        {result?.error && (
          <p className="text-red-600 mt-4 text-center">
            {result.error}
          </p>
        )}
      </div>
    </div>
  );
}

export default App;
