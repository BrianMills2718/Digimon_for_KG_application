import React from 'react';
import './OntologyDisplay.css'; // Create this CSS file next

function OntologyDisplay({ ontology }) {
    if (!ontology || (!ontology.entities?.length && !ontology.relations?.length)) {
        if (ontology?.message) { // Display message if backend returns one for empty/error
            return <div className="ontology-display-container"><p>{ontology.message}</p></div>;
        }
        return <div className="ontology-display-container"><p>No ontology loaded or defined.</p></div>;
    }

    return (
        <div className="ontology-display-container">
            <h3>Current Ontology Structure</h3>

            <h4>Entities:</h4>
            {ontology.entities && ontology.entities.length > 0 ? (
                <ul>
                    {ontology.entities.map((entity, index) => (
                        <li key={`entity-${index}`} className="ontology-item">
                            <strong>{entity.name}</strong>
                            {entity.properties && entity.properties.length > 0 && (
                                <ul className="properties-list">
                                    {entity.properties.map((prop, pIndex) => (
                                        <li key={`entity-${index}-prop-${pIndex}`}>
                                            {prop.name} (<em>{prop.type || 'any'}</em>): {prop.description || 'No description'}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </li>
                    ))}
                </ul>
            ) : <p>No entity types defined.</p>}

            <h4>Relations:</h4>
            {ontology.relations && ontology.relations.length > 0 ? (
                <ul>
                    {ontology.relations.map((relation, index) => (
                        <li key={`relation-${index}`} className="ontology-item">
                            <strong>{relation.name}</strong>
                            <div className="relation-details">
                                <span>Source: {relation.source_entity || 'Any'}</span>
                                <span>Target: {relation.target_entity || 'Any'}</span>
                                {relation.description && <span>Description: {relation.description}</span>}
                            </div>
                            {relation.properties && relation.properties.length > 0 && (
                                <ul className="properties-list">
                                    {relation.properties.map((prop, pIndex) => (
                                        <li key={`relation-${index}-prop-${pIndex}`}>
                                            {prop.name} (<em>{prop.type || 'any'}</em>): {prop.description || 'No description'}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </li>
                    ))}
                </ul>
            ) : <p>No relation types defined.</p>}
        </div>
    );
}

export default OntologyDisplay;
