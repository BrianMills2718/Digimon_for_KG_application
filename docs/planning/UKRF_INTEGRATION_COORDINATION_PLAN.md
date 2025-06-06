# UKRF Integration Coordination Plan
## Universal Knowledge Reasoning Framework

**Distribution**: This document should be shared with all Claude Code agents working on DIGIMON, Autocoder, and StructGPT for coordinated integration planning.

---

## Executive Summary

We are integrating three AI systems into a unified Universal Knowledge Reasoning Framework (UKRF):

- **DIGIMON**: Master orchestrator and knowledge graph engine
- **StructGPT**: Structured data reasoning (SQL, tables, cross-modal linking)
- **Autocoder**: Dynamic code generation and tool creation

**Goal**: Create a seamless system where users can ask complex questions spanning multiple data types, and the system automatically selects tools, generates new capabilities as needed, and provides comprehensive answers.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    DIGIMON                                  ││
│  │              (Master Orchestrator)                         ││
│  │                                                             ││
│  │  • Query understanding & decomposition                     ││
│  │  • Tool selection & routing                                ││
│  │  • Knowledge graph reasoning                               ││
│  │  • Result synthesis & presentation                         ││
│  │  • Context management across tools                         ││
│  └─────────────────┬───────────────┬───────────────────────────┘│
│                    │               │                            │
│  ┌─────────────────▼─────────────  │  ──▼─────────────────────┐ │
│  │        StructGPT              │  │  │      Autocoder       │ │
│  │   (Structured Data)           │  │  │   (Code Generation)  │ │
│  │                               │  │  │                      │ │
│  │  • Text-to-SQL generation     │  │  │ • Tool creation      │ │
│  │  • Table QA & analysis        │  │  │ • API integration    │ │
│  │  • Entity extraction          │  │  │ • Custom workflows   │ │
│  │  • Cross-modal linking        │  │  │ • Dynamic adaptation │ │
│  └───────────────────────────────┘  │  └──────────────────────┘ │
│                                     │                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              MCP Protocol Layer                             ││
│  │        (Model Context Protocol)                            ││
│  │                                                             ││
│  │  • Standardized tool communication                         ││
│  │  • Shared context management                               ││
│  │  • Cross-system entity linking                             ││
│  │  • Unified error handling                                  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Integration Strategy: Federation Approach

**Decision**: We will use a **Federation approach** initially, evolving toward tighter integration as interfaces stabilize.

### Repository Structure
```
ukrf-integration/
├── docker-compose.yml          # Orchestrates all services
├── shared/                     # Shared libraries & protocols
│   ├── mcp-protocol/          # MCP implementation
│   ├── common-types/          # Shared data models
│   └── monitoring/            # Unified observability
├── services/
│   ├── digimon/              # Git submodule or symlink
│   ├── structgpt/            # Git submodule or symlink  
│   └── autocoder/            # Git submodule or symlink
├── integration-tests/         # End-to-end testing
├── docs/                     # Combined documentation
└── deployment/               # Production deployment configs
```

### Communication Protocol: MCP (Model Context Protocol)

All systems will communicate via MCP servers on designated ports:
- **DIGIMON**: Port 8765 (Master orchestrator)
- **StructGPT**: Port 8766 (Structured data tools)
- **Autocoder**: Port 8767 (Code generation tools)

## StructGPT System Overview

### Core Capabilities
1. **Text-to-SQL Generation**: Convert natural language to SQL queries
2. **Table Question Answering**: Analyze tabular data and answer questions
3. **Knowledge Graph QA**: Multi-hop reasoning over knowledge graphs
4. **Cross-Modal Entity Linking**: Connect database records to knowledge entities

### Performance Characteristics
- **Query Latency**: <1 second (p50), <5 seconds (p99)
- **Accuracy**: 78.5% on Spider benchmark (SQL), 76.8% on TabFact
- **Concurrency**: 100+ simultaneous queries supported
- **Memory**: <100MB per session
- **Error Recovery**: 95%+ success rate with retry mechanisms

### Infrastructure Components
- **Configuration**: Hierarchical config with environment variable support
- **LLM Abstraction**: OpenAI, Anthropic providers with fallback
- **Monitoring**: Real-time metrics, dashboards, alerting
- **Caching**: Multi-level (LLM responses, embeddings, SQL results)
- **Parallel Processing**: BatchProcessor, TaskPool for high throughput
- **Retry Logic**: Exponential backoff, circuit breakers

### MCP Tools Exposed
1. **`structgpt.sql_generation`**: Generate SQL from natural language
2. **`structgpt.table_qa`**: Answer questions about tables
3. **`structgpt.extract_entities`**: Extract entities from SQL results

### Data Formats
```python
# SQL Generation Input
{
    "question": "Find all customers who bought more than $1000",
    "database_schema": {"db_id": "ecommerce", "tables": [...], "columns": [...]},
    "dialect": "postgresql"
}

# SQL Generation Output
{
    "sql": "SELECT customer_id, SUM(amount) FROM orders...",
    "confidence": 0.92,
    "explanation": "Query finds customers with total purchases > $1000",
    "entities": [{"name": "Customer", "type": "Entity", "ids": [...]}]
}
```

## Information Needed from DIGIMON Team

### 1. Architecture & Components
- [ ] **Core orchestration logic**: How does DIGIMON route queries to tools?
- [ ] **Knowledge graph structure**: What graph database/format used?
- [ ] **Query decomposition**: How are complex queries broken down?
- [ ] **Context management**: How is conversation context maintained?
- [ ] **Tool registration**: How are new tools discovered and registered?

### 2. MCP Implementation Status
- [ ] **MCP server**: Is MCP server already implemented? On what port?
- [ ] **Tool calling pattern**: How does DIGIMON invoke external tools?
- [ ] **Context sharing**: What shared context format is used?
- [ ] **Error handling**: How are tool errors propagated and handled?
- [ ] **Performance requirements**: Latency/throughput expectations?

### 3. Data Models & Interfaces
- [ ] **Entity representation**: How are entities represented in knowledge graph?
- [ ] **Query result format**: Expected format for tool responses?
- [ ] **Cross-modal linking**: How should SQL entities link to graph entities?
- [ ] **Schema management**: How are database schemas shared/discovered?

### 4. Integration Points
- [ ] **Tool discovery**: How should StructGPT tools be registered?
- [ ] **Entity registration**: When StructGPT extracts entities, how to register them?
- [ ] **Schema updates**: How to notify DIGIMON of new database schemas?
- [ ] **Result aggregation**: How are results from multiple tools combined?

### 5. Testing & Deployment
- [ ] **Integration test framework**: What testing approach is used?
- [ ] **Configuration management**: How are service configs managed?
- [ ] **Monitoring integration**: Is there a unified monitoring system?
- [ ] **Deployment strategy**: Docker, Kubernetes, or other approach?

## Information Needed from Autocoder Team

### 1. Architecture & Capabilities
- [ ] **Code generation scope**: What types of code can Autocoder generate?
- [ ] **Tool creation process**: How are new tools dynamically created?
- [ ] **API integration**: Can Autocoder create API wrappers automatically?
- [ ] **Database adapters**: Can it generate database connection code?
- [ ] **Validation**: How is generated code tested/validated?

### 2. Integration with StructGPT
- [ ] **Schema-driven generation**: Can Autocoder use DB schemas to generate code?
- [ ] **SQL optimization**: Can it generate optimized SQL variants?
- [ ] **Data pipeline creation**: Can it create ETL/data processing pipelines?
- [ ] **Custom tool creation**: Generate new StructGPT tools based on requirements?

### 3. MCP Integration
- [ ] **Tool generation via MCP**: Can tools be requested via MCP calls?
- [ ] **Code execution**: How is generated code safely executed?
- [ ] **Dependency management**: How are dependencies handled for generated code?
- [ ] **Caching**: Are generated tools cached for reuse?

### 4. DIGIMON Integration
- [ ] **Dynamic tool registration**: How are new tools registered with DIGIMON?
- [ ] **Capability expansion**: How does DIGIMON request new capabilities?
- [ ] **Quality control**: How is generated code quality ensured?

## Integration Scenarios to Plan For

### Scenario 1: SQL-to-Knowledge Graph Bridge
**User Query**: "Find all papers by Einstein and show their citation network"

**Workflow**:
1. DIGIMON receives query, identifies need for both SQL and graph data
2. Routes to StructGPT: "Find papers by Einstein" → SQL generation
3. StructGPT extracts entities from SQL results
4. DIGIMON uses entities to query knowledge graph for citations
5. Results combined and presented

**Requirements**:
- Entity linking between SQL results and graph entities
- Shared context for maintaining entity mappings
- Result format compatibility

### Scenario 2: Dynamic Tool Creation
**User Query**: "Analyze the correlation between weather data and sales"

**Workflow**:
1. DIGIMON identifies need for weather API integration (not existing)
2. Routes to Autocoder: "Create weather API tool"
3. Autocoder generates weather data fetcher
4. DIGIMON registers new tool, uses it alongside StructGPT for sales analysis
5. StructGPT analyzes correlation between datasets

**Requirements**:
- Dynamic tool creation and registration
- Data format standardization
- Tool versioning and updates

### Scenario 3: Multi-Modal Complex Query
**User Query**: "Compare our Q4 sales performance with industry benchmarks and predict Q1 trends"

**Workflow**:
1. DIGIMON decomposes into: sales analysis + benchmark lookup + prediction
2. StructGPT handles internal sales data analysis
3. Autocoder creates benchmark data scraper if needed
4. DIGIMON synthesizes results and handles trend prediction
5. Combined response with visualizations

**Requirements**:
- Complex query decomposition
- Tool orchestration
- Result synthesis across modalities

## Implementation Timeline

### Phase 1: Basic Integration (Weeks 1-2)
- [ ] **Week 1**: MCP server setup in all services
- [ ] **Week 2**: Basic tool calling between services

### Phase 2: Entity Linking (Weeks 3-4)
- [ ] **Week 3**: Cross-modal entity extraction and registration
- [ ] **Week 4**: Shared context management

### Phase 3: Dynamic Capabilities (Weeks 5-6)
- [ ] **Week 5**: Autocoder integration for tool creation
- [ ] **Week 6**: Dynamic tool registration in DIGIMON

### Phase 4: Production Ready (Weeks 7-8)
- [ ] **Week 7**: Performance optimization, error handling
- [ ] **Week 8**: Monitoring, deployment, documentation

## Action Items for Each Team

### DIGIMON Team
1. **Document current architecture** and MCP implementation status
2. **Define tool registration protocol** for external services
3. **Specify entity representation format** for cross-modal linking
4. **Create integration test framework** for multi-service workflows

### StructGPT Team (Us)
1. **Complete MCP server implementation** on port 8766
2. **Enhance entity extraction** for cross-modal registration
3. **Document all tool APIs** and data formats
4. **Create Docker deployment** configuration

### Autocoder Team
1. **Document code generation capabilities** and limitations
2. **Define tool creation API** accessible via MCP
3. **Specify generated code standards** for integration
4. **Create dynamic tool registration** mechanism

## Risk Mitigation

### Technical Risks
- **Protocol version conflicts**: Lock MCP versions, maintain compatibility
- **Performance degradation**: Early benchmarking, caching strategies
- **Context synchronization**: Atomic updates, conflict resolution

### Integration Risks
- **Service dependencies**: Circuit breakers, graceful degradation
- **Data format mismatches**: Schema validation, transformation layers
- **Dynamic tool quality**: Code review, automated testing

## Success Metrics

### Functional Goals
- [ ] All three services communicate via MCP
- [ ] Cross-modal entity linking working (90%+ accuracy)
- [ ] Dynamic tool creation and registration functional
- [ ] End-to-end complex queries working

### Performance Goals
- [ ] Query latency <2 seconds for simple queries
- [ ] System handles 50+ concurrent users
- [ ] Tool creation time <30 seconds
- [ ] 99.9% uptime across all services

### Integration Goals
- [ ] Zero-downtime deployments
- [ ] Unified monitoring and alerting
- [ ] Comprehensive integration test suite
- [ ] Documentation for all interfaces

## Next Steps

1. **Immediate** (This Week):
   - Share this document with all teams
   - Schedule integration planning meeting
   - Begin documenting current architectures

2. **Short Term** (Next 2 Weeks):
   - Complete architecture documentation exchange
   - Start MCP server implementations
   - Create shared development environment

3. **Medium Term** (Weeks 3-8):
   - Execute integration phases
   - Regular cross-team synchronization
   - Iterative testing and refinement

## Communication Protocol

- **Weekly sync meetings**: All teams, integration status
- **Shared documentation**: Real-time updates on interfaces
- **Integration testing**: Continuous testing across services
- **Issue tracking**: Unified issue tracking for integration problems

---

**Document Status**: Initial Draft  
**Last Updated**: January 2025  
**Next Review**: Weekly during integration phase

**Contacts**:
- StructGPT Lead: [Current Claude Code session]
- DIGIMON Lead: [To be provided]
- Autocoder Lead: [To be provided]