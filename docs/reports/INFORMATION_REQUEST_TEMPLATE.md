# Information Request Template for UKRF Integration

**To**: Claude Code Agent working on [DIGIMON/Autocoder]  
**From**: StructGPT Integration Team  
**Subject**: Integration Planning Information Request

---

## Background

We are coordinating the integration of DIGIMON, StructGPT, and Autocoder into a Universal Knowledge Reasoning Framework (UKRF). To create an optimal integration plan, each team needs detailed information about the other systems.

**Please review the attached `UKRF_INTEGRATION_COORDINATION_PLAN.md` for full context.**

## Your Task

Please analyze your current codebase and provide the following information to enable optimal integration planning:

---

## Section A: System Architecture

### A1. Core Components
- [ ] **Main orchestration/control logic**: Which files contain the core system logic?
- [ ] **API/interface layer**: How does your system expose functionality?
- [ ] **Data models**: What are your key data structures and schemas?
- [ ] **Configuration system**: How is your system configured?
- [ ] **Error handling**: How are errors handled and propagated?

### A2. Performance Characteristics
- [ ] **Typical latency**: How long do operations take?
- [ ] **Memory usage**: What are typical memory requirements?
- [ ] **Concurrency**: How many simultaneous operations supported?
- [ ] **Bottlenecks**: What are the main performance limitations?

### A3. Dependencies
- [ ] **External services**: What external APIs/services do you use?
- [ ] **Libraries/frameworks**: Key dependencies and versions
- [ ] **Infrastructure requirements**: Database, cache, etc.

---

## Section B: Integration Readiness

### B1. Communication Interfaces
- [ ] **Existing APIs**: What APIs does your system already expose?
- [ ] **Input/output formats**: What data formats do you use?
- [ ] **Async capability**: Do you support asynchronous operations?
- [ ] **Streaming**: Can you handle streaming requests/responses?

### B2. MCP (Model Context Protocol) Status
- [ ] **Current implementation**: Do you have MCP server already?
- [ ] **Port assignment**: What port should your MCP server use?
- [ ] **Tool registration**: How are tools/capabilities exposed?
- [ ] **Context sharing**: How do you maintain conversation context?

### B3. Extension Points
- [ ] **Plugin architecture**: How can new capabilities be added?
- [ ] **Tool registration**: How are new tools discovered/registered?
- [ ] **Configuration updates**: Can configuration be updated at runtime?
- [ ] **Hot reload**: Can components be updated without restart?

---

## Section C: Cross-System Integration

### C1. For DIGIMON (Master Orchestrator)
- [ ] **Query routing**: How do you decide which tools to use for a query?
- [ ] **Tool selection**: What information do you need about available tools?
- [ ] **Result synthesis**: How do you combine results from multiple tools?
- [ ] **Context management**: How do you maintain state across tool calls?
- [ ] **Entity management**: How do you handle entities discovered by tools?

### C2. For Autocoder (Code Generation)
- [ ] **Generation triggers**: What prompts tool/code generation?
- [ ] **Code validation**: How do you ensure generated code is safe/correct?
- [ ] **Tool packaging**: How are generated tools made available?
- [ ] **Dependency management**: How do you handle dependencies for generated code?
- [ ] **Versioning**: How are different versions of generated tools managed?

### C3. For StructGPT Integration
Based on StructGPT's capabilities, please indicate:
- [ ] **SQL generation needs**: Would you benefit from text-to-SQL capabilities?
- [ ] **Table analysis needs**: Do you need tabular data analysis?
- [ ] **Entity extraction needs**: Would extracted entities from SQL be useful?
- [ ] **Database integration**: What database systems do you work with?

---

## Section D: Integration Scenarios

For each scenario below, please describe how your system would participate:

### D1. Complex Cross-Modal Query
**Example**: "Compare our Q4 sales with industry benchmarks and predict trends"

**Your role**: How would your system contribute to this workflow?
**Input needed**: What information would you need from other systems?
**Output provided**: What would you provide to other systems?

### D2. Dynamic Capability Creation
**Example**: User needs analysis of a new data source type

**Your role**: How would your system handle this requirement?
**Integration points**: How would you coordinate with other systems?

### D3. Real-Time Collaborative Analysis
**Example**: Multiple users working on related queries simultaneously

**Your role**: How does your system handle concurrent operations?
**State sharing**: How would you share state with other systems?

---

## Section E: Technical Constraints & Concerns

### E1. Current Limitations
- [ ] **Known bottlenecks**: What would slow down integration?
- [ ] **Missing capabilities**: What would you need to add for integration?
- [ ] **Breaking changes**: What changes might break existing functionality?

### E2. Integration Concerns
- [ ] **Security concerns**: What security considerations are important?
- [ ] **Data privacy**: Any restrictions on data sharing between systems?
- [ ] **Backwards compatibility**: What compatibility must be maintained?

### E3. Resource Requirements
- [ ] **Development time**: How much work would integration require?
- [ ] **Infrastructure changes**: What infrastructure changes needed?
- [ ] **Testing requirements**: What testing would be required?

---

## Section F: Proposed Integration Points

Based on the coordination plan, please comment on these proposed integration points:

### F1. MCP Protocol Communication
**Proposal**: All systems communicate via MCP servers on designated ports
- [ ] **Feasibility**: Can your system implement MCP server?
- [ ] **Timeline**: How long would MCP implementation take?
- [ ] **Concerns**: Any issues with this approach?

### F2. Shared Context Management
**Proposal**: Systems share context for entities, schemas, and state
- [ ] **Current context handling**: How do you currently manage context?
- [ ] **Sharing mechanism**: How could context be shared with other systems?
- [ ] **Conflicts**: How would context conflicts be resolved?

### F3. Federation Architecture
**Proposal**: Keep separate repositories, integrate via Docker Compose
- [ ] **Deployment preferences**: Do you prefer monorepo or federation?
- [ ] **Configuration management**: How should configs be managed across services?
- [ ] **Monitoring**: Should monitoring be unified or separate?

---

## Section G: Implementation Preferences

### G1. Timeline Preferences
- [ ] **Integration urgency**: How soon do you need integration capabilities?
- [ ] **Phase preferences**: Which integration phases are most important?
- [ ] **Resource availability**: How much development time can you allocate?

### G2. Technical Preferences
- [ ] **Communication protocols**: Any preferences beyond MCP?
- [ ] **Data formats**: Preferred data serialization formats?
- [ ] **Error handling**: Preferred error handling patterns?

### G3. Testing Approach
- [ ] **Current testing**: What testing framework do you use?
- [ ] **Integration testing**: How should cross-system testing work?
- [ ] **Continuous integration**: What CI/CD approach do you prefer?

---

## Deliverables Requested

Please provide:

1. **Architecture Document**: Overview of your system's architecture
2. **API Specification**: Current and planned APIs
3. **Integration Proposal**: Your preferred integration approach
4. **Risk Assessment**: Potential integration challenges
5. **Timeline Estimate**: Development time for integration work

## Next Steps

After receiving this information:

1. **Analysis Phase**: We'll analyze all responses and create detailed integration plan
2. **Design Review**: Cross-team review of integration architecture
3. **Proof of Concept**: Implement basic integration between systems
4. **Iterative Development**: Phased implementation with regular sync points

## Questions & Clarifications

If any sections are unclear or you need more context:
- Review the full `UKRF_INTEGRATION_COORDINATION_PLAN.md`
- Ask specific questions about StructGPT capabilities
- Request clarification on integration requirements

---

**Response Deadline**: [To be specified]  
**Contact**: StructGPT Integration Team  
**Priority**: High - Integration planning depends on this information