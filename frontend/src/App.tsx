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
      )}import { useState, ChangeEvent } from "react";

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

  // This will use your Render URL in production and localhost during development
  const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";

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
      // Updated to use the dynamic URL
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Server responded with an error");
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error details:", error);
      alert("Prediction failed. Make sure the backend is awake and CORS is enabled.");
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
    <div style={{ padding: 40, maxWidth: "600px", margin: "0 auto", fontFamily: "sans-serif" }}>
      <h1>Alzheimer's Early Detection</h1>

      <input
        type="file"
        accept="image/*"
        onChange={handleFileChange}
      />

      <br />
      <br />

      <button 
        onClick={handleSubmit} 
        disabled={loading}
        style={{ padding: "10px 20px", cursor: loading ? "not-allowed" : "pointer" }}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {result && !result.error && (
        <div style={{ marginTop: 20, padding: 20, border: "1px solid #ddd", borderRadius: 8 }}>
          <h3>Result</h3>
          <p>
            <b>Prediction:</b> {result.prediction}
          </p>
          <p>
            <b>Confidence:</b> {(result.confidence * 100).toFixed(2)}%
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

      {result?.error && (
        <p style={{ color: "red", marginTop: 20 }}>
          Error: {result.error}
        </p>
      )}
    </div>
  );
}

export default App;
