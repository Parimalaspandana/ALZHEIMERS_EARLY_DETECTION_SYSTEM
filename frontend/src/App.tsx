import { useState, ChangeEvent } from "react";

interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities?: Record<string, number>;
  error?: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

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
      const response = await fetch("http://127.0.0.1:8001/predict", {
        method: "POST",
        body: formData,
      });

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (error) {
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  return ( 
      <div style={{ padding: 40 }}>
      <h1>Alzheimer's Early Detection</h1>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      <br />
      <br />

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {result && !result.error && (
        <div style={{ marginTop: 20 }}>
          <h3>Result</h3>
          <p>
            <b>Prediction:</b> {result.prediction}
          </p>
          <p>
            <b>Confidence:</b> {result.confidence}%
          </p>
        </div>
      )}

      {result?.error && (
        <p style={{ color: "red", marginTop: 20 }}>
          Error: {result.error}
        </p>
      )}
    </div>
  );
}

export default App;
