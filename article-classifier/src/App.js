import { useState } from "react";

export default function ArticleClassifier() {
  const [title, setTitle] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleClassify = async () => {
    if (!title.trim()) {
      setError("Vui lòng nhập tiêu đề bài báo!");
      return;
    }
    setLoading(true);
    setError(null);
    setPredictions([]);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });

      if (!response.ok) {
        throw new Error("Lỗi khi gọi API");
      }

      const data = await response.json();

      // Kiểm tra API có trả về predictions không
      if (!data.predictions || !Array.isArray(data.predictions)) {
        throw new Error("Dữ liệu trả về không hợp lệ!");
      }

      setPredictions(data.predictions);
    } catch (err) {
      setError(err.message || "Không thể phân loại bài báo. Vui lòng thử lại!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white p-6 rounded-xl shadow-md w-96">
        <h1 className="text-xl font-bold mb-4">Phân loại bài báo</h1>
        <input
          type="text"
          placeholder="Nhập tiêu đề bài báo..."
          className="w-full p-2 border rounded-md mb-4"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
        <button
          className="w-full bg-blue-500 text-white p-2 rounded-md"
          onClick={handleClassify}
          disabled={loading}
        >
          {loading ? "Đang phân loại..." : "Phân loại"}
        </button>

        {error && <p className="text-red-500 mt-4">{error}</p>}

        {predictions.length > 0 && (
          <div className="mt-4">
            <h2 className="font-semibold">Kết quả phân loại:</h2>
            <ul className="mt-2 text-lg">
              {predictions.map((pred, index) => (
                <li key={index}>
                  <strong>Nhóm {pred.class}:</strong> {pred.confidence.toFixed(2)}%
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
