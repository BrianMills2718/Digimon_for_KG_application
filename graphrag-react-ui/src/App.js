import React, { useState, useEffect } from 'react';
import { Settings, Info } from 'lucide-react';
import ChatComponent from './ChatComponent';
import OntologyDisplay from './OntologyDisplay';

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


// --- Main App Component ---

export default function App() {
  const [queryResult, setQueryResult] = useState(null);
  const [evaluateResult, setEvaluateResult] = useState(null);
  const [datasetName, setDatasetName] = useState('MySampleTexts');
  const [selectedMethod, setSelectedMethod] = useState(RAG_METHODS[0]);

  // Ontology management states
  const [activeOntologyJson, setActiveOntologyJson] = useState(null); // Last accepted ontology from chat
  const [currentSavedOntology, setCurrentSavedOntology] = useState(null); // Ontology fetched from backend
  const [buildStatus, setBuildStatus] = useState(''); // Build status message

  // Fetch current ontology from backend
  const fetchCurrentOntology = async () => {
    try {
      const response = await fetch('/api/ontology');
      if (response.ok) {
        const data = await response.json();
        console.log("Fetched current ontology from /api/ontology:", data);
        setCurrentSavedOntology(data);
        // Initialize activeOntologyJson only if it's currently null
        // and the fetched ontology is not just an empty/message placeholder
        if (activeOntologyJson === null && (data.entities?.length > 0 || data.relations?.length > 0)) {
             setActiveOntologyJson(data);
             console.log("Initialized activeOntologyJson with data from /api/ontology.");
        }
      } else {
        console.error("Failed to fetch current ontology, status:", response.status);
        const errorData = await response.json().catch(() => null);
        setCurrentSavedOntology({ entities: [], relations: [], message: `Failed to load: ${errorData?.error || response.statusText}` });
      }
    } catch (error) {
      console.error("Error fetching current ontology:", error);
      setCurrentSavedOntology({ entities: [], relations: [], message: `Error loading: ${error.message}` });
    }
  };

  useEffect(() => {
    fetchCurrentOntology();
    // eslint-disable-next-line
  }, []);

  // Accept ontology from chat
  const handleOntologySuggestionFromChat = (ontologyJson) => {
    console.log("ACCEPTION: Ontology suggestion accepted from chat via onOntologySuggested:", ontologyJson);
    setActiveOntologyJson(ontologyJson); 
    setCurrentSavedOntology(ontologyJson); // Update display immediately
    alert("Ontology suggestion accepted and set as active! It will be saved when you click 'Build Graph'.");
  };

  // Build handler: save ontology first, then build
  const handleBuildGraph = async () => {
    setBuildStatus('Starting build process...');
    console.log("BUILD_CLICKED: handleBuildGraph called.");
    console.log("BUILD_CLICKED: Current activeOntologyJson:", activeOntologyJson);
    console.log("BUILD_CLICKED: Current datasetName:", datasetName, "selectedMethod:", selectedMethod);

    if (!datasetName || !selectedMethod) {
      alert("Please select a dataset and method for building.");
      setBuildStatus('Build failed: Dataset or method not selected.');
      return;
    }

    if (activeOntologyJson && (activeOntologyJson.entities?.length > 0 || activeOntologyJson.relations?.length > 0) ) {
      try {
        setBuildStatus('Saving active ontology to /api/ontology...');
        console.log("BUILD_CLICKED: Attempting to POST activeOntologyJson:", JSON.stringify(activeOntologyJson, null, 2));
        const saveResponse = await fetch('/api/ontology', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(activeOntologyJson),
        });
        if (!saveResponse.ok) {
          const errorData = await saveResponse.json().catch(() => ({error: "Unknown error during ontology save."}));
          throw new Error(errorData.error || `Failed to save ontology. Status: ${saveResponse.status}`);
        }
        const saveData = await saveResponse.json();
        setBuildStatus(`Ontology saved: ${saveData.message}. Now triggering graph build...`);
        console.log("BUILD_CLICKED: Ontology POSTed successfully:", saveData);
        await fetchCurrentOntology(); // Re-fetch to ensure consistency for display
      } catch (error) {
        console.error("BUILD_CLICKED: Error saving ontology before build:", error);
        setBuildStatus(`Error saving ontology: ${error.message}. Build aborted.`);
        alert(`Error saving ontology: ${error.message}. Build aborted.`);
        return; 
      }
    } else {
      setBuildStatus('No new active ontology from chat, build will use existing custom_ontology.json (if any). Triggering graph build...');
      console.log("BUILD_CLICKED: No new active ontology to save. Proceeding with existing.");
    }

    try {
      console.log("BUILD_CLICKED: Calling /api/build with:", { datasetName, selectedMethod });
      setBuildStatus(`Building graph for ${datasetName} using ${selectedMethod}...`);
      const buildApiResponse = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ datasetName, selectedMethod }),
      });
      const buildData = await buildApiResponse.json();
      if (!buildApiResponse.ok) {
        throw new Error(buildData.error || `Build API error status: ${buildApiResponse.status}`);
      }
      setBuildStatus(`Build successful for ${datasetName} using ${selectedMethod}: ${buildData.message || buildData.details}`);
      alert(`Build successful: ${buildData.message || buildData.details}`);
      console.log("BUILD_CLICKED: Build response:", buildData);
    } catch (error) {
      console.error("BUILD_CLICKED: Error during /api/build call:", error);
      setBuildStatus(`Build failed: ${error.message}`);
      alert(`Build failed: ${error.message}`);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 font-sans">
      <main className="container mx-auto p-4 md:p-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
          {/* Left panel: Build controls and ontology display */}
          <div className="md:col-span-1">
            <Card title="Build & Ontology" icon={<Settings />}>
              <div className="space-y-6">
                <div>
                  <Label htmlFor="datasetName">Dataset Name</Label>
                  <Input
                    type="text"
                    id="datasetName"
                    value={datasetName}
                    onChange={e => setDatasetName(e.target.value)}
                    placeholder="e.g., MySampleTexts"
                  />
                </div>
                <div>
                  <Label htmlFor="selectedMethod">RAG Method</Label>
                  <Select
                    id="selectedMethod"
                    value={selectedMethod}
                    onChange={e => setSelectedMethod(e.target.value)}
                  >
                    {RAG_METHODS.map(method => (
                      <option key={method} value={method}>{method}</option>
                    ))}
                  </Select>
                </div>
                <button onClick={handleBuildGraph} className="w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                  Build Graph with Active Ontology
                </button>
                {buildStatus && <p className="mt-2 text-sm text-slate-600">Status: {buildStatus}</p>}
              </div>
              <div className="space-y-4 mt-4">
                <button
                  onClick={async () => {
                    const question = window.prompt('Enter your query/question for the graph:');
                    if (!question) return;
                    try {
                      const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          datasetName,
                          selectedMethod,
                          question
                        })
                      });
                      const data = await response.json();
                      if (response.ok && data.answer) {
                        setQueryResult({ type: 'answer', content: data.answer });
                      } else {
                        setQueryResult({ type: 'error', content: data.error || 'Unknown error from /api/query' });
                      }
                    } catch (error) {
                      setQueryResult({ type: 'error', content: error.message });
                    }
                  }}
                  className="w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500"
                >
                  Query Graph
                </button>
                <button
                  onClick={async () => {
                    try {
                      const response = await fetch('/api/evaluate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          datasetName,
                          selectedMethod
                        })
                      });
                      const data = await response.json();
                      if (response.ok && (data.metrics || data.message)) {
                        setEvaluateResult({ type: 'success', content: data.metrics || data.message });
                      } else {
                        setEvaluateResult({ type: 'error', content: data.error || 'Unknown error from /api/evaluate' });
                      }
                    } catch (error) {
                      setEvaluateResult({ type: 'error', content: error.message });
                    }
                  }}
                  className="w-full flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  Evaluate Graph
                </button>
                {/* Results Display */}
                {queryResult && (
                  <div className={`mt-2 p-3 rounded text-sm ${queryResult.type === 'error' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-800'}`}>
                    <strong>Query Result:</strong> {queryResult.content}
                  </div>
                )}
                {evaluateResult && (
                  <div className={`mt-2 p-3 rounded text-sm ${evaluateResult.type === 'error' ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-800'}`}>
                    <strong>Evaluation Result:</strong> {evaluateResult.content}
                  </div>
                )}
              </div>
              <OntologyDisplay ontology={currentSavedOntology} />
            </Card>
          </div>
          {/* Right panel: Chat */}
          <div className="md:col-span-2">
            <Card title="LLM Guidance Chat" icon={<Info />}>
              <ChatComponent onOntologySuggested={handleOntologySuggestionFromChat} />
            </Card>
          </div>
        </div>
      </main>
      <footer className="text-center p-6 text-sm text-slate-500 border-t border-slate-200 mt-12">
        GraphRAG Control Panel | Your PhD Research Tool
      </footer>
    </div>
  );
}

