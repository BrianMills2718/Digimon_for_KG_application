// GraphRAG Control Panel UI v1
// (Immersive artifact: graphrag_ui_v1)

const { useState } = React;

function GraphRAGControlPanel() {
  const [mode, setMode] = useState('build');
  const [method, setMethod] = useState('LGraphRAG');
  const [dataset, setDataset] = useState('MySampleTexts');
  const [question, setQuestion] = useState('');
  const [logs, setLogs] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const methodOptions = [
    { label: 'LGraphRAG', value: 'LGraphRAG' },
    { label: 'GGraphRAG', value: 'GGraphRAG' },
    { label: 'RAPTOR', value: 'RAPTOR' },
  ];
  const datasetOptions = [
    { label: 'MySampleTexts', value: 'MySampleTexts' },
    // Add more datasets as needed
  ];

  // Simulate logs and output for demonstration
  function simulateOperation(op) {
    setLoading(true);
    setLogs('');
    setResult('');
    setTimeout(() => {
      let fakeLog = `--- ${op.toUpperCase()} MODE ---\n`;
      fakeLog += `Method: ${method}\nDataset: ${dataset}\n`;
      if (op === 'query') {
        fakeLog += `Question: ${question}\n`;
      }
      fakeLog += `\n[INFO] Simulating ${op}...\n`;
      if (op === 'build') {
        fakeLog += '[INFO] Graph and index artifacts created.\n';
        setResult('Artifacts saved to: ./results/' + dataset + '/<method>/');
      } else if (op === 'query') {
        fakeLog += '[INFO] Query processed.\n';
        setResult('Generated answer: "Simulated answer for: ' + question + '"');
      } else if (op === 'evaluate') {
        fakeLog += '[INFO] Evaluation complete.\n';
        setResult('Metrics saved to: ./results/' + dataset + '/<method>/Evaluation_Outputs/' + method + '/' + dataset + '_evaluation_metrics.json');
      }
      setLogs(fakeLog);
      setLoading(false);
    }, 1200);
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center py-8 px-2">
      <div className="w-full max-w-xl bg-white shadow-xl rounded-lg p-8">
        <h1 className="text-3xl font-bold mb-2 text-slate-800 flex items-center gap-2">
          {/* Icon placeholder */}
          <span className="inline-block w-7 h-7 bg-slate-200 rounded-full mr-2"></span>
          GraphRAG Control Panel
        </h1>
        <p className="text-slate-500 mb-6">Interactively build, query, and evaluate your GraphRAG pipelines.</p>
        <div className="flex flex-col gap-4 mb-6">
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-slate-700 mb-1">RAG Method</label>
              <select
                className="w-full border border-slate-300 rounded px-3 py-2 bg-slate-50 focus:ring-2 focus:ring-slate-300"
                value={method}
                onChange={e => setMethod(e.target.value)}
              >
                {methodOptions.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset</label>
              <select
                className="w-full border border-slate-300 rounded px-3 py-2 bg-slate-50 focus:ring-2 focus:ring-slate-300"
                value={dataset}
                onChange={e => setDataset(e.target.value)}
              >
                {datasetOptions.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-slate-700 mb-1">Mode</label>
              <div className="flex gap-2">
                <button
                  className={`flex-1 px-4 py-2 rounded ${mode === 'build' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700'} font-semibold transition`}
                  onClick={() => setMode('build')}
                >Build</button>
                <button
                  className={`flex-1 px-4 py-2 rounded ${mode === 'query' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700'} font-semibold transition`}
                  onClick={() => setMode('query')}
                >Query</button>
                <button
                  className={`flex-1 px-4 py-2 rounded ${mode === 'evaluate' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-700'} font-semibold transition`}
                  onClick={() => setMode('evaluate')}
                >Evaluate</button>
              </div>
            </div>
          </div>
          {mode === 'query' && (
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Question</label>
              <input
                type="text"
                className="w-full border border-slate-300 rounded px-3 py-2 bg-slate-50 focus:ring-2 focus:ring-slate-300"
                value={question}
                onChange={e => setQuestion(e.target.value)}
                placeholder="Type your question here..."
              />
            </div>
          )}
        </div>
        <button
          className="w-full py-3 rounded bg-blue-700 text-white font-bold text-lg shadow hover:bg-blue-800 transition flex items-center justify-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
          onClick={() => simulateOperation(mode)}
          disabled={loading || (mode === 'query' && !question)}
        >
          {loading ? (
            <span className="animate-spin inline-block w-5 h-5 border-2 border-white border-t-blue-700 rounded-full"></span>
          ) : (
            <span>{mode.charAt(0).toUpperCase() + mode.slice(1)}</span>
          )}
        </button>
        <div className="mt-6">
          <label className="block text-sm font-medium text-slate-700 mb-1">Logs</label>
          <pre className="bg-slate-900 text-green-200 rounded p-3 h-40 overflow-auto text-xs whitespace-pre-wrap">{logs}</pre>
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium text-slate-700 mb-1">Result</label>
          <pre className="bg-slate-100 text-slate-800 rounded p-3 h-20 overflow-auto text-sm whitespace-pre-wrap">{result}</pre>
        </div>
      </div>
      <div className="mt-8 text-slate-400 text-xs text-center max-w-lg">
        <p>Note: This UI simulates build, query, and evaluate operations. To connect to your Python backend, you'll need to implement an API and update the UI accordingly.</p>
      </div>
    </div>
  );
}

// Render the app
ReactDOM.createRoot(document.getElementById('root')).render(<GraphRAGControlPanel />);
