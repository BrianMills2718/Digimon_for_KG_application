import React, { useState, useEffect, useRef } from 'react';
import './ChatComponent.css';

const CHAT_MODES = {
    ONTOLOGY_DESIGN: 'ONTOLOGY_DESIGN',
    ANALYSIS_GUIDANCE: 'ANALYSIS_GUIDANCE',
};

function ChatComponent({ onOntologySuggested }) {
    const [currentMode, setCurrentMode] = useState(CHAT_MODES.ONTOLOGY_DESIGN); // Default mode
    const [messages, setMessages] = useState([
        { text: "Hello! I'm here to help. Select a mode: 'Ontology Design' to define your graph structure, or 'Analysis Guidance' for help with queries and methods.", isUser: false, id: Date.now() }
    ]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleModeChange = (event) => {
        const newMode = event.target.value;
        setCurrentMode(newMode);
        setMessages([ // Reset messages or provide a mode-specific welcome
            { text: `Switched to ${newMode === CHAT_MODES.ONTOLOGY_DESIGN ? 'Ontology Design' : 'Analysis Guidance'} mode. How can I assist?`, isUser: false, id: Date.now() }
        ]);
    };

    const handleSendMessage = async () => {
        if (inputText.trim() === '') return;

        const userMessage = { text: inputText, isUser: true, id: Date.now(), suggestedOntologyJson: null };
        const updatedMessages = [...messages, userMessage];
        setMessages(updatedMessages);
        setInputText('');
        setIsLoading(true);

        let endpoint = '';
        if (currentMode === CHAT_MODES.ONTOLOGY_DESIGN) {
            endpoint = '/api/ontology_chat';
        } else if (currentMode === CHAT_MODES.ANALYSIS_GUIDANCE) {
            endpoint = '/api/chat_guidance';
        }

        try {
            const response = await fetch(endpoint, { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json', },
                body: JSON.stringify({ 
                    message: userMessage.text,
                    history: updatedMessages.slice(0, -1).map(msg => ({
                        text: msg.text, // Send full text for history context
                        isUser: msg.isUser 
                    })) 
                }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: "Unknown error structure" }));
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorData.detail || response.statusText}`);
            }

            const data = await response.json();
            let botTextReply = "";
            let actualSuggestedJson = null;

            if (currentMode === CHAT_MODES.ONTOLOGY_DESIGN && data.descriptive_reply) {
                botTextReply = data.descriptive_reply;
                if (data.suggested_ontology_json && !data.suggested_ontology_json.error) {
                    actualSuggestedJson = data.suggested_ontology_json; // This is now a direct JSON object
                    console.log("Received structured ontology JSON:", actualSuggestedJson);
                } else if (data.suggested_ontology_json && data.suggested_ontology_json.error) {
                    console.error("Backend reported an error generating ontology JSON:", data.suggested_ontology_json.error);
                    botTextReply += "\n\n[Error generating structured ontology - see console for details]";
                }
            } else {
                botTextReply = data.reply || "Sorry, I couldn't get a response.";
            }

            const botMessage = { 
                text: botTextReply, 
                isUser: false, 
                id: Date.now() + 1,
                suggestedOntologyJson: actualSuggestedJson // Store the direct JSON object
            };
            setMessages(prevMessages => [...prevMessages, botMessage]);

        } catch (error) {
            console.error(`Failed to send message to ${endpoint}:`, error);
            const errorReply = { text: `Error: ${error.message || "Could not connect to server."}`, isUser: false, id: Date.now() + 1 };
            setMessages(prevMessages => [...prevMessages, errorReply]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="chat-container">
            <div className="chat-mode-selector">
                <label htmlFor="chatMode">Chat Mode: </label>
                <select id="chatMode" value={currentMode} onChange={handleModeChange}>
                    <option value={CHAT_MODES.ONTOLOGY_DESIGN}>Ontology Design</option>
                    <option value={CHAT_MODES.ANALYSIS_GUIDANCE}>Analysis Guidance</option>
                </select>
            </div>
            <div className="chat-messages">
                {messages.map((msg) => (
                    <div key={msg.id} className={`message ${msg.isUser ? 'user-message' : 'bot-message'}`}>
                        {msg.text.split('\n').map((line, i, arr) => (
                            <span key={i}>{line}{i === arr.length - 1 ? '' : <br />}</span>
                        ))}
                        {msg.suggestedOntologyJson && (
                            <div className="suggested-ontology-actions">
                                <p><strong>Ontology suggestion available:</strong></p>
                                <button 
                                    onClick={() => onOntologySuggested(msg.suggestedOntologyJson)}
                                    className="use-ontology-button"
                                >
                                    Accept & View/Edit Suggested Ontology
                                </button>
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && <div className="message bot-message typing-indicator"><span></span><span></span><span></span></div>}
                <div ref={messagesEndRef} />
            </div>
            <div className="chat-input-area">
                <input
                    type="text"
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyPress={(e) => {
                        if (e.key === 'Enter' && !isLoading) {
                            e.preventDefault(); 
                            handleSendMessage();
                        }
                    }}
                    placeholder={currentMode === CHAT_MODES.ONTOLOGY_DESIGN ? "Describe your KG goal for ontology help..." : "Ask for analysis/method guidance..."}
                    disabled={isLoading}
                />
                <button onClick={handleSendMessage} disabled={isLoading}>
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
}

export default ChatComponent;
