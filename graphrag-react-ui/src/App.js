import React, { useState, useEffect, useCallback } from 'react';
import { Play, Settings, BarChart2, FileText, Loader2, AlertTriangle, CheckCircle2, Info } from 'lucide-react';

// Mock list of available RAG methods (stem of the YAML file names)
const RAG_METHODS = [
  "LGraphRAG",
  "GGraphRAG",
  "LightRAG",
  "GR",
  "Dalk",
  "ToG",
  "KGP",
  "RAPTOR",
  "HippoRAG",
];

// --- UI Components (Assume these are unchanged from previous version) ---

const Header = () => (
  <header className="bg-slate-800 text-white p-6 shadow-lg">
    <div className="container mx-auto">
      <h1 className="text-4xl font-bold tracking-tight">GraphRAG Control Panel</h1>
      <p className="text-slate-300 mt-1">Manage and experiment with your GraphRAG project.</p>
    </div>
  </header>
);

const Card = ({ title, icon, children }) => (
  <div className="bg-white rounded-xl shadow-lg overflow-hidden">
    <div className="p-6">
      <div className="flex items-center text-slate-700 mb-4">
        {icon && React.cloneElement(icon, { className: "w-6 h-6 mr-3 text-indigo-600" })}
        <h2 className="text-2xl font-semibold">{title}</h2>
      </div>
      {children}
    </div>
  </div>
);

const Label = ({ htmlFor, children }) => (
  <label htmlFor={htmlFor} className="block text-sm font-medium text-slate-700 mb-1">
    {children}
  </label>
);

const Input = (props) => (
  <input
    {...props}
    className="mt-1 block w-full px-4 py-3 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm transition-shadow"
  />
);

const Select = ({ children, ...props }) => (
  <select
    {...props}
    className="mt-1 block w-full pl-4 pr-10 py-3 text-base border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-lg shadow-sm transition-shadow"
  >
    {children}
  </select>
);

const Button = ({ children, onClick, isLoading = false, variant = 'primary', ...props }) => {
  const baseStyles = "w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-150 ease-in-out";
  const variants = {
    primary: "text-white bg-indigo-600 hover:bg-indigo-700 focus:ring-indigo-500",
    secondary: "text-indigo-700 bg-indigo-100 hover:bg-indigo-200 focus:ring-indigo-500",
    danger: "text-white bg-red-600 hover:bg-red-700 focus:ring-red-500",
  };
  const disabledStyles = "opacity-50 cursor-not-allowed";

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={isLoading}
      className={`${baseStyles} ${variants[variant]} ${isLoading ? disabledStyles : ''}`}
      {...props}
    >
      {isLoading && <Loader2 className="mr-2 h-5 w-5 animate-spin" />}
      {children}
    </button>
  );
};

const TextArea = (props) => (
  <textarea
    {...props}
    rows={props.rows || 3}
    className="mt-1 block w-full px-4 py-3 border border-slate-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm transition-shadow"
  />
);

const LogDisplay = ({ logs }) => (
  <div className="mt-4 bg-slate-50 p-4 rounded-lg max-h-60 overflow-y-auto shadow">
    <h3 className="text-sm font-semibold text-slate-600 mb-2">Operation Log:</h3>
    <pre className="text-xs text-slate-500 whitespace-pre-wrap">
      {logs.length > 0 ? logs.join('\n') : 'No operations yet.'}
    </pre>
  </div>
);

const ResultDisplay = ({ title, data, icon }) => {
  if (!data) return null;
  return (
    <div className="mt-6 bg-slate-50 p-6 rounded-lg shadow">
      <div className="flex items-center text-slate-700 mb-3">
        {icon && React.cloneElement(icon, { className: "w-5 h-5 mr-2 text-indigo-500" })}
        <h3 className="text-lg font-semibold">{title}</h3>
      </div>
      <pre className="text-sm text-slate-600 bg-white p-4 rounded-md shadow-inner whitespace-pre-wrap overflow-x-auto">
        {typeof data === 'object' ? JSON.stringify(data, null, 2) : data}
      </pre>
    </div>
  );
};

// --- Main App Component ---

export default function App() {
  const [datasetName, setDatasetName] = useState('MySampleTexts');
  const [selectedMethod, setSelectedMethod] = useState(RAG_METHODS[0]);
  const [question, setQuestion] = useState('What were the key causes of the American Revolution?');
  
  const [isBuilding, setIsBuilding] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  
  const [buildResult, setBuildResult] = useState(null);
  const [queryResult, setQueryResult] = useState(null);
  const [evaluationResult, setEvaluationResult] = useState(null);
  
  const [logs, setLogs] = useState([]);
  const [lastOperationStatus, setLastOperationStatus] = useState({ type: '', message: ''});

  const API_BASE_URL = 'http://localhost:5000/api'; // Your Flask API URL

  const addLog = useCallback((message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prevLogs => [`[${timestamp}] ${message}`, ...prevLogs.slice(0, 99)]);
  }, []);

  // Function to make actual API calls to the backend
  const callBackendAPI = async (endpoint, payload, operationName) => {
    addLog(`Starting ${operationName}...`);
    addLog(`Calling backend endpoint: ${API_BASE_URL}/${endpoint} with payload: ${JSON.stringify(payload)}`);
    setLastOperationStatus({type: 'info', message: `${operationName} in progress...`});

    try {
      const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const responseData = await response.json();

      if (!response.ok) {
        // If response is not OK (e.g., 4xx, 5xx), throw an error with backend message
        const errorMsg = `Error during ${operationName}: ${responseData.error || response.statusText}`;
        addLog(errorMsg);
        setLastOperationStatus({type: 'error', message: errorMsg});
        return { success: false, data: responseData, message: errorMsg };
      }

      addLog(`${operationName} completed successfully.`);
      setLastOperationStatus({type: 'success', message: `${operationName} completed successfully.`});
      return { success: true, data: responseData, message: `${operationName} successful.` };

    } catch (error) {
      // Network errors or other issues with the fetch call
      const errorMsg = `Network or system error during ${operationName}: ${error.message}`;
      addLog(errorMsg);
      console.error("API Call Error:", error);
      setLastOperationStatus({type: 'error', message: errorMsg});
      return { success: false, data: null, message: errorMsg };
    }
  };
  
  const handleBuild = async () => {
    if (!datasetName || !selectedMethod) {
      setLastOperationStatus({type: 'error', message: 'Dataset name and method must be selected.'});
      return;
    }
    setIsBuilding(true);
    setBuildResult(null);
    
    const payload = {
      datasetName: datasetName,
      selectedMethod: selectedMethod, // The API will derive the .yaml path from this
    };
    const result = await callBackendAPI('build', payload, 'Build Artifacts');
    
    setBuildResult(result.data?.message || result.message); // Display message from backend or error
    setIsBuilding(false);
  };

  const handleQuery = async () => {
    if (!datasetName || !selectedMethod || !question) {
      setLastOperationStatus({type: 'error', message: 'Dataset, method, and question must be provided.'});
      return;
    }
    setIsQuerying(true);
    setQueryResult(null);

    const payload = {
      datasetName: datasetName,
      selectedMethod: selectedMethod,
      question: question,
    };
    const result = await callBackendAPI('query', payload, 'Query');

    if (result.success && result.data && result.data.answer) {
      setQueryResult(result.data.answer);
    } else {
      setQueryResult(result.data?.error || result.message || "Failed to get answer from backend.");
    }
    setIsQuerying(false);
  };

  const handleEvaluate = async () => {
    if (!datasetName || !selectedMethod) {
      setLastOperationStatus({type: 'error', message: 'Dataset name and method must be selected for evaluation.'});
      return;
    }
    setIsEvaluating(true);
    setEvaluationResult(null);

    const payload = {
      datasetName: datasetName,
      selectedMethod: selectedMethod,
    };
    const result = await callBackendAPI('evaluate', payload, 'Evaluation');

    if (result.success && result.data) {
      // Assuming the backend /evaluate endpoint returns metrics directly or a success message
      // For now, let's assume it returns a message, and actual metrics would be in backend logs/files
      setEvaluationResult(result.data?.metrics || result.data?.message || result.message);
    } else {
      setEvaluationResult(result.data?.error || result.message || "Failed to get evaluation results.");
    }
    setIsEvaluating(false);
  };
  
  useEffect(() => {
    setBuildResult(null);
    setQueryResult(null);
    setEvaluationResult(null);
    setLastOperationStatus({type: '', message: ''});
  }, [datasetName, selectedMethod]);

  return (
    <div className="min-h-screen bg-slate-100 font-sans">
      <Header />
      <main className="container mx-auto p-4 md:p-8">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
          
          <Card title="Configuration" icon={<Settings />}>
            <div className="space-y-6">
              <div>
                <Label htmlFor="datasetName">Dataset Name</Label>
                <Input 
                  type="text" 
                  id="datasetName" 
                  value={datasetName} 
                  onChange={(e) => setDatasetName(e.target.value)}
                  placeholder="e.g., MySampleTexts"
                />
                <p className="mt-1 text-xs text-slate-500">Corresponds to a folder in your `./Data/` directory.</p>
              </div>
              <div>
                <Label htmlFor="selectedMethod">RAG Method</Label>
                <Select 
                  id="selectedMethod" 
                  value={selectedMethod} 
                  onChange={(e) => setSelectedMethod(e.target.value)}
                >
                  {RAG_METHODS.map(method => (
                    <option key={method} value={method}>{method}</option>
                  ))}
                </Select>
                 <p className="mt-1 text-xs text-slate-500">Uses `Option/Method/{selectedMethod}.yaml`.</p>
              </div>
            </div>
          </Card>

          <Card title="Operations" icon={<Play />}>
            <div className="space-y-4">
              <Button onClick={handleBuild} isLoading={isBuilding}>
                Build Artifacts
              </Button>
              <div>
                <Label htmlFor="question">Question for Query Mode</Label>
                <TextArea 
                  id="question"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={4}
                />
              </div>
              <Button onClick={handleQuery} isLoading={isQuerying} variant="secondary">
                Run Query
              </Button>
              <Button onClick={handleEvaluate} isLoading={isEvaluating}>
                Run Evaluation
              </Button>
            </div>
          </Card>

          <div className="md:col-span-2 lg:col-span-1">
            <Card title="Status & Logs" icon={<Info />}>
              {lastOperationStatus.message && (
                <div className={`p-3 mb-4 rounded-md text-sm ${
                  lastOperationStatus.type === 'success' ? 'bg-green-50 text-green-700 ring-1 ring-inset ring-green-600/20' :
                  lastOperationStatus.type === 'error' ? 'bg-red-50 text-red-700 ring-1 ring-inset ring-red-600/20' :
                  'bg-blue-50 text-blue-700 ring-1 ring-inset ring-blue-600/20'
                }`}>
                  <div className="flex items-center">
                    {lastOperationStatus.type === 'success' && <CheckCircle2 className="w-5 h-5 mr-2" />}
                    {lastOperationStatus.type === 'error' && <AlertTriangle className="w-5 h-5 mr-2" />}
                    {lastOperationStatus.type === 'info' && <Loader2 className="w-5 h-5 mr-2 animate-spin" />}
                    <p className="font-medium">{lastOperationStatus.message}</p>
                  </div>
                </div>
              )}
              <LogDisplay logs={logs} />
            </Card>
          </div>

          {(buildResult || queryResult || evaluationResult) && (
             <div className="md:col-span-2 lg:col-span-3">
                <Card title="Results" icon={<FileText />}>
                    <ResultDisplay title="Build Output" data={buildResult} icon={<Settings />} />
                    <ResultDisplay title="Query Answer" data={queryResult} icon={<Play />} />
                    <ResultDisplay title="Evaluation Metrics" data={evaluationResult} icon={<BarChart2 />} />
                </Card>
            </div>
          )}
        </div>
        
        <div className="mt-8 p-4 bg-amber-50 border border-amber-300 rounded-lg text-amber-700 text-sm shadow">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-amber-500" />
            <strong>Note:</strong> This UI now attempts to call a backend API at <code>http://localhost:5000/api</code>. Ensure your Python Flask server (api.py) is running in WSL and accessible.
          </div>
        </div>
      </main>
      <footer className="text-center p-6 text-sm text-slate-500 border-t border-slate-200 mt-12">
        GraphRAG Control Panel | Your PhD Research Tool
      </footer>
    </div>
  );
}
