"""
Pattern Learning System for Cognitive Architecture

This module implements a comprehensive pattern learning system that can:
- Recognize patterns from experiences and data
- Store and index patterns for efficient retrieval
- Match new situations against existing patterns
- Apply learned patterns to new contexts
- Learn and adapt patterns based on feedback
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import hashlib

from .memory_architecture import MemoryArchitecture, EpisodicMemory, SemanticMemory, MemoryType

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns that can be learned"""
    SEQUENCE = "sequence"           # Temporal sequences of events
    STRUCTURE = "structure"         # Structural relationships
    CAUSAL = "causal"              # Cause-effect relationships
    BEHAVIORAL = "behavioral"       # Behavioral patterns
    LINGUISTIC = "linguistic"      # Language patterns
    NUMERICAL = "numerical"        # Numerical patterns
    CATEGORICAL = "categorical"    # Categorical associations


@dataclass
class Pattern:
    """Represents a learned pattern"""
    id: str
    pattern_type: PatternType
    name: str
    structure: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    counter_examples: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    support: int = 0  # Number of supporting instances
    frequency: float = 0.0
    last_used: Optional[datetime] = None
    creation_time: datetime = field(default_factory=datetime.utcnow)
    success_rate: float = 0.5
    applications: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['pattern_type'] = self.pattern_type.value
        data['creation_time'] = self.creation_time.isoformat()
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """Create from dictionary"""
        data['pattern_type'] = PatternType(data['pattern_type'])
        data['creation_time'] = datetime.fromisoformat(data['creation_time'])
        if data.get('last_used'):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        return cls(**data)


@dataclass
class PatternMatch:
    """Represents a pattern match result"""
    pattern: Pattern
    confidence: float
    match_data: Dict[str, Any]
    similarity_score: float
    context_match: float
    
    @property
    def overall_score(self) -> float:
        """Calculate overall match score"""
        return (self.confidence * 0.4 + 
                self.similarity_score * 0.4 + 
                self.context_match * 0.2)


class PatternRecognizer(ABC):
    """Abstract base class for pattern recognizers"""
    
    @abstractmethod
    async def extract_patterns(
        self,
        data: List[Dict[str, Any]],
        pattern_type: PatternType
    ) -> List[Pattern]:
        """Extract patterns from data"""
        pass
    
    @abstractmethod
    async def match_pattern(
        self,
        pattern: Pattern,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[PatternMatch]:
        """Match a pattern against new data"""
        pass


class SequencePatternRecognizer(PatternRecognizer):
    """Recognizes sequential patterns in temporal data"""
    
    def __init__(self, min_sequence_length: int = 2, min_support: int = 2):
        self.min_sequence_length = min_sequence_length
        self.min_support = min_support
    
    async def extract_patterns(
        self,
        data: List[Dict[str, Any]],
        pattern_type: PatternType = PatternType.SEQUENCE
    ) -> List[Pattern]:
        """Extract sequence patterns using sliding window approach"""
        patterns = []
        
        # Extract sequences from data
        sequences = self._extract_sequences(data)
        
        # Find frequent subsequences
        frequent_subsequences = self._find_frequent_subsequences(sequences)
        
        # Create pattern objects
        for seq, support in frequent_subsequences.items():
            if len(seq) >= self.min_sequence_length and support >= self.min_support:
                pattern_id = self._generate_pattern_id(seq)
                pattern = Pattern(
                    id=pattern_id,
                    pattern_type=pattern_type,
                    name=f"Sequence: {' → '.join(seq)}",
                    structure={
                        "sequence": list(seq),
                        "length": len(seq),
                        "transitions": self._get_transitions(seq)
                    },
                    support=support,
                    frequency=support / len(sequences) if sequences else 0,
                    confidence=min(0.9, support / max(1, self.min_support * 2))
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_sequences(self, data: List[Dict[str, Any]]) -> List[List[str]]:
        """Extract event sequences from data"""
        sequences = []
        
        for item in data:
            if isinstance(item, dict):
                # Handle episodic memory format
                if 'event_sequence' in item:
                    seq = [event.get('event', str(event)) for event in item['event_sequence']]
                    if seq:
                        sequences.append(seq)
                # Handle simple sequence format
                elif 'sequence' in item:
                    sequences.append(item['sequence'])
                # Handle single event
                elif 'event' in item:
                    sequences.append([item['event']])
        
        return sequences
    
    def _find_frequent_subsequences(
        self,
        sequences: List[List[str]]
    ) -> Dict[Tuple[str, ...], int]:
        """Find frequent subsequences using sliding window"""
        subsequence_counts = defaultdict(int)
        
        for sequence in sequences:
            # Generate all subsequences
            for length in range(self.min_sequence_length, len(sequence) + 1):
                for start in range(len(sequence) - length + 1):
                    subseq = tuple(sequence[start:start + length])
                    subsequence_counts[subseq] += 1
        
        return subsequence_counts
    
    def _get_transitions(self, sequence: Tuple[str, ...]) -> List[Dict[str, str]]:
        """Get state transitions in sequence"""
        transitions = []
        for i in range(len(sequence) - 1):
            transitions.append({
                "from": sequence[i],
                "to": sequence[i + 1]
            })
        return transitions
    
    def _generate_pattern_id(self, sequence: Tuple[str, ...]) -> str:
        """Generate unique ID for pattern"""
        seq_str = "->".join(sequence)
        return f"seq_{hashlib.md5(seq_str.encode()).hexdigest()[:8]}"
    
    async def match_pattern(
        self,
        pattern: Pattern,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[PatternMatch]:
        """Match sequence pattern against new data"""
        if pattern.pattern_type != PatternType.SEQUENCE:
            return None
        
        target_sequence = pattern.structure.get("sequence", [])
        if not target_sequence:
            return None
        
        # Extract sequence from data
        data_sequence = []
        if 'event_sequence' in data:
            data_sequence = [event.get('event', str(event)) for event in data['event_sequence']]
        elif 'sequence' in data:
            data_sequence = data['sequence']
        elif 'event' in data:
            data_sequence = [data['event']]
        
        if not data_sequence:
            return None
        
        # Calculate similarity
        similarity = self._calculate_sequence_similarity(target_sequence, data_sequence)
        
        if similarity < 0.3:  # Minimum threshold
            return None
        
        # Calculate context match
        context_match = self._calculate_context_match(pattern, context)
        
        return PatternMatch(
            pattern=pattern,
            confidence=pattern.confidence,
            match_data={
                "matched_sequence": data_sequence,
                "target_sequence": target_sequence,
                "similarity_details": self._get_similarity_details(target_sequence, data_sequence)
            },
            similarity_score=similarity,
            context_match=context_match
        )
    
    def _calculate_sequence_similarity(
        self,
        target: List[str],
        candidate: List[str]
    ) -> float:
        """Calculate similarity between two sequences"""
        if not target or not candidate:
            return 0.0
        
        # Exact match bonus
        if target == candidate:
            return 1.0
        
        # Subsequence matching
        target_set = set(target)
        candidate_set = set(candidate)
        
        # Jaccard similarity for elements
        intersection = len(target_set & candidate_set)
        union = len(target_set | candidate_set)
        element_similarity = intersection / union if union > 0 else 0
        
        # Order similarity (longest common subsequence)
        lcs_length = self._longest_common_subsequence(target, candidate)
        order_similarity = (2 * lcs_length) / (len(target) + len(candidate))
        
        # Combined similarity
        return (element_similarity * 0.4 + order_similarity * 0.6)
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _get_similarity_details(
        self,
        target: List[str],
        candidate: List[str]
    ) -> Dict[str, Any]:
        """Get detailed similarity analysis"""
        return {
            "target_length": len(target),
            "candidate_length": len(candidate),
            "common_elements": list(set(target) & set(candidate)),
            "unique_to_target": list(set(target) - set(candidate)),
            "unique_to_candidate": list(set(candidate) - set(target)),
            "lcs_length": self._longest_common_subsequence(target, candidate)
        }
    
    def _calculate_context_match(
        self,
        pattern: Pattern,
        context: Dict[str, Any]
    ) -> float:
        """Calculate how well context matches pattern's typical context"""
        if not context or not pattern.metadata:
            return 0.5
        
        # Simple context matching based on shared keys
        pattern_context = pattern.metadata.get('typical_context', {})
        if not pattern_context:
            return 0.5
        
        shared_keys = set(context.keys()) & set(pattern_context.keys())
        total_keys = set(context.keys()) | set(pattern_context.keys())
        
        if not total_keys:
            return 0.5
        
        return len(shared_keys) / len(total_keys)


class StructuralPatternRecognizer(PatternRecognizer):
    """Recognizes structural patterns in relational data"""
    
    async def extract_patterns(
        self,
        data: List[Dict[str, Any]],
        pattern_type: PatternType = PatternType.STRUCTURE
    ) -> List[Pattern]:
        """Extract structural patterns from relational data"""
        patterns = []
        
        # Extract relationships and structures
        structures = self._extract_structures(data)
        
        # Find common structural patterns
        pattern_frequencies = self._find_structural_patterns(structures)
        
        for structure_key, frequency in pattern_frequencies.items():
            if frequency >= 2:  # Minimum support
                structure = json.loads(structure_key)
                pattern_id = self._generate_structure_id(structure)
                pattern = Pattern(
                    id=pattern_id,
                    pattern_type=pattern_type,
                    name=f"Structure: {structure['name']}",
                    structure=structure,
                    support=frequency,
                    confidence=min(0.9, frequency / max(1, len(structures))),
                    frequency=frequency / len(structures) if structures else 0
                )
                patterns.append(pattern)
        
        return patterns
    
    def _extract_structures(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract structural information from data"""
        structures = []
        
        for item in data:
            if isinstance(item, dict):
                # Extract relationships
                if 'relations' in item:
                    for relation_type, targets in item['relations'].items():
                        structures.append({
                            "type": "relation",
                            "relation": relation_type,
                            "source": item.get('concept', 'unknown'),
                            "targets": targets,
                            "name": f"{item.get('concept', 'X')} {relation_type} {targets}"
                        })
                
                # Extract properties
                if 'properties' in item:
                    structures.append({
                        "type": "properties",
                        "entity": item.get('concept', 'unknown'),
                        "properties": item['properties'],
                        "name": f"Properties of {item.get('concept', 'X')}"
                    })
        
        return structures
    
    def _find_structural_patterns(
        self,
        structures: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Find common structural patterns"""
        pattern_counts = defaultdict(int)
        
        for structure in structures:
            # Create pattern key based on structure type and properties
            if structure['type'] == 'relation':
                key = f"relation:{structure['relation']}"
                pattern_counts[key] += 1
            elif structure['type'] == 'properties':
                for prop_key in structure['properties'].keys():
                    key = f"property:{prop_key}"
                    pattern_counts[key] += 1
        
        # Convert to proper structure format
        result = {}
        for key, count in pattern_counts.items():
            parts = key.split(':', 1)
            structure = {
                "type": parts[0],
                "name": parts[1],
                "pattern_key": key
            }
            result[json.dumps(structure, sort_keys=True)] = count
        
        return result
    
    def _generate_structure_id(self, structure: Dict[str, Any]) -> str:
        """Generate unique ID for structural pattern"""
        struct_str = json.dumps(structure, sort_keys=True)
        return f"struct_{hashlib.md5(struct_str.encode()).hexdigest()[:8]}"
    
    async def match_pattern(
        self,
        pattern: Pattern,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[PatternMatch]:
        """Match structural pattern against new data"""
        if pattern.pattern_type != PatternType.STRUCTURE:
            return None
        
        # Extract structure from data
        data_structures = self._extract_structures([data])
        
        # Check for matches
        for structure in data_structures:
            similarity = self._calculate_structural_similarity(
                pattern.structure,
                structure
            )
            
            if similarity > 0.5:
                return PatternMatch(
                    pattern=pattern,
                    confidence=pattern.confidence,
                    match_data={
                        "matched_structure": structure,
                        "pattern_structure": pattern.structure
                    },
                    similarity_score=similarity,
                    context_match=0.5
                )
        
        return None
    
    def _calculate_structural_similarity(
        self,
        pattern_struct: Dict[str, Any],
        data_struct: Dict[str, Any]
    ) -> float:
        """Calculate similarity between structures"""
        if pattern_struct.get('type') != data_struct.get('type'):
            return 0.0
        
        # Type-specific similarity calculation
        if pattern_struct['type'] == 'relation':
            if pattern_struct.get('relation') == data_struct.get('relation'):
                return 0.8
        elif pattern_struct['type'] == 'properties':
            pattern_props = set(pattern_struct.get('properties', {}).keys())
            data_props = set(data_struct.get('properties', {}).keys())
            if pattern_props & data_props:
                return len(pattern_props & data_props) / len(pattern_props | data_props)
        
        return 0.0


class PatternStore:
    """Stores and indexes patterns for efficient retrieval"""
    
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}
        self.type_index: Dict[PatternType, Set[str]] = defaultdict(set)
        self.name_index: Dict[str, Set[str]] = defaultdict(set)
        self.confidence_index: Dict[float, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def store_pattern(self, pattern: Pattern) -> str:
        """Store a pattern"""
        async with self._lock:
            self.patterns[pattern.id] = pattern
            
            # Update indices
            self.type_index[pattern.pattern_type].add(pattern.id)
            
            # Index by name keywords
            name_words = pattern.name.lower().split()
            for word in name_words:
                self.name_index[word].add(pattern.id)
            
            # Index by confidence range
            conf_bucket = round(pattern.confidence, 1)
            self.confidence_index[conf_bucket].add(pattern.id)
            
            return pattern.id
    
    async def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Retrieve a specific pattern"""
        return self.patterns.get(pattern_id)
    
    async def search_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        min_confidence: float = 0.0,
        name_keywords: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Pattern]:
        """Search for patterns matching criteria"""
        candidates = set(self.patterns.keys())
        
        # Filter by type
        if pattern_type:
            candidates &= self.type_index.get(pattern_type, set())
        
        # Filter by confidence
        if min_confidence > 0:
            conf_candidates = set()
            for conf_bucket, pattern_ids in self.confidence_index.items():
                if conf_bucket >= min_confidence:
                    conf_candidates.update(pattern_ids)
            candidates &= conf_candidates
        
        # Filter by name keywords
        if name_keywords:
            name_candidates = set()
            for keyword in name_keywords:
                name_candidates.update(self.name_index.get(keyword.lower(), set()))
            candidates &= name_candidates
        
        # Get pattern objects and sort by confidence
        result_patterns = [self.patterns[pid] for pid in candidates]
        result_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return result_patterns[:limit]
    
    async def update_pattern(
        self,
        pattern_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a pattern"""
        async with self._lock:
            if pattern_id not in self.patterns:
                return False
            
            pattern = self.patterns[pattern_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(pattern, key):
                    setattr(pattern, key, value)
            
            return True
    
    async def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern"""
        async with self._lock:
            if pattern_id not in self.patterns:
                return False
            
            pattern = self.patterns[pattern_id]
            del self.patterns[pattern_id]
            
            # Remove from indices
            self.type_index[pattern.pattern_type].discard(pattern_id)
            
            name_words = pattern.name.lower().split()
            for word in name_words:
                self.name_index[word].discard(pattern_id)
            
            conf_bucket = round(pattern.confidence, 1)
            self.confidence_index[conf_bucket].discard(pattern_id)
            
            return True
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get pattern store statistics"""
        stats = {
            "total_patterns": len(self.patterns),
            "patterns_by_type": {
                pt.value: len(ids) for pt, ids in self.type_index.items()
            },
            "average_confidence": 0.0,
            "high_confidence_patterns": 0
        }
        
        if self.patterns:
            confidences = [p.confidence for p in self.patterns.values()]
            stats["average_confidence"] = sum(confidences) / len(confidences)
            stats["high_confidence_patterns"] = sum(1 for c in confidences if c > 0.7)
        
        return stats


class PatternLearningSystem:
    """Main pattern learning system coordinating all components"""
    
    def __init__(self, memory_architecture: Optional[MemoryArchitecture] = None):
        self.memory = memory_architecture or MemoryArchitecture()
        self.pattern_store = PatternStore()
        
        # Pattern recognizers
        self.recognizers = {
            PatternType.SEQUENCE: SequencePatternRecognizer(),
            PatternType.STRUCTURE: StructuralPatternRecognizer()
        }
        
        self.learning_rate = 0.1
        self.confidence_threshold = 0.6
        
    async def learn_patterns_from_memory(
        self,
        memory_type: MemoryType = MemoryType.EPISODIC,
        time_window: Optional[timedelta] = None
    ) -> List[Pattern]:
        """Learn patterns from existing memories"""
        # Get memories to analyze
        search_criteria = {'memory_type': memory_type}
        if time_window:
            end_time = datetime.utcnow()
            start_time = end_time - time_window
            search_criteria.update({
                'start_time': start_time,
                'end_time': end_time
            })
        
        if memory_type == MemoryType.EPISODIC:
            memories = await self.memory.episodic_store.search(search_criteria, limit=100)
        else:
            memories = await self.memory.semantic_store.search(search_criteria, limit=100)
        
        # Convert memories to pattern learning format
        data = []
        for memory in memories:
            if hasattr(memory, 'to_dict'):
                data.append(memory.to_dict())
            else:
                data.append(memory)
        
        # Extract patterns using appropriate recognizers
        all_patterns = []
        
        for pattern_type, recognizer in self.recognizers.items():
            try:
                patterns = await recognizer.extract_patterns(data, pattern_type)
                for pattern in patterns:
                    await self.pattern_store.store_pattern(pattern)
                all_patterns.extend(patterns)
            except Exception as e:
                logger.error(f"Error extracting {pattern_type} patterns: {e}")
        
        logger.info(f"Learned {len(all_patterns)} patterns from {len(memories)} memories")
        return all_patterns
    
    async def match_patterns(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        pattern_types: Optional[List[PatternType]] = None,
        min_confidence: float = 0.3
    ) -> List[PatternMatch]:
        """Match input data against learned patterns"""
        matches = []
        
        # Get candidate patterns
        search_types = pattern_types or list(self.recognizers.keys())
        
        for pattern_type in search_types:
            if pattern_type not in self.recognizers:
                continue
            
            recognizer = self.recognizers[pattern_type]
            patterns = await self.pattern_store.search_patterns(
                pattern_type=pattern_type,
                min_confidence=min_confidence
            )
            
            for pattern in patterns:
                try:
                    match = await recognizer.match_pattern(pattern, data, context)
                    if match and match.overall_score >= min_confidence:
                        matches.append(match)
                except Exception as e:
                    logger.error(f"Error matching pattern {pattern.id}: {e}")
        
        # Sort by overall score
        matches.sort(key=lambda m: m.overall_score, reverse=True)
        return matches
    
    async def apply_pattern(
        self,
        pattern_match: PatternMatch,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a matched pattern to generate predictions or actions"""
        pattern = pattern_match.pattern
        
        # Update pattern usage statistics
        await self.pattern_store.update_pattern(pattern.id, {
            'applications': pattern.applications + 1,
            'last_used': datetime.utcnow()
        })
        
        # Generate application based on pattern type
        if pattern.pattern_type == PatternType.SEQUENCE:
            return await self._apply_sequence_pattern(pattern_match, context)
        elif pattern.pattern_type == PatternType.STRUCTURE:
            return await self._apply_structural_pattern(pattern_match, context)
        else:
            return {
                "pattern_id": pattern.id,
                "pattern_name": pattern.name,
                "application_type": "generic",
                "confidence": pattern_match.overall_score,
                "context": context
            }
    
    async def _apply_sequence_pattern(
        self,
        pattern_match: PatternMatch,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply sequence pattern to predict next events"""
        pattern = pattern_match.pattern
        sequence = pattern.structure.get("sequence", [])
        matched_seq = pattern_match.match_data.get("matched_sequence", [])
        
        # Predict next events in sequence
        predictions = []
        if len(matched_seq) < len(sequence):
            next_events = sequence[len(matched_seq):]
            predictions = next_events[:3]  # Predict up to 3 next events
        
        return {
            "pattern_id": pattern.id,
            "pattern_name": pattern.name,
            "application_type": "sequence_prediction",
            "predictions": predictions,
            "confidence": pattern_match.overall_score,
            "current_position": len(matched_seq),
            "total_sequence_length": len(sequence),
            "context": context
        }
    
    async def _apply_structural_pattern(
        self,
        pattern_match: PatternMatch,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply structural pattern to infer relationships or properties"""
        pattern = pattern_match.pattern
        structure = pattern.structure
        
        inferences = []
        if structure.get('type') == 'relation':
            inferences.append({
                "type": "relationship",
                "relation": structure.get('relation'),
                "confidence": pattern_match.overall_score
            })
        elif structure.get('type') == 'properties':
            inferences.append({
                "type": "property_inference",
                "properties": structure.get('properties', {}),
                "confidence": pattern_match.overall_score
            })
        
        return {
            "pattern_id": pattern.id,
            "pattern_name": pattern.name,
            "application_type": "structural_inference",
            "inferences": inferences,
            "confidence": pattern_match.overall_score,
            "context": context
        }
    
    async def provide_feedback(
        self,
        pattern_id: str,
        feedback: Dict[str, Any]
    ) -> bool:
        """Provide feedback on pattern application to improve learning"""
        pattern = await self.pattern_store.get_pattern(pattern_id)
        if not pattern:
            return False
        
        # Extract feedback components
        success = feedback.get('success', False)
        accuracy = feedback.get('accuracy', 0.5)
        usefulness = feedback.get('usefulness', 0.5)
        
        # Update pattern based on feedback
        new_success_rate = self._update_success_rate(
            pattern.success_rate,
            pattern.applications,
            success
        )
        
        new_confidence = self._update_confidence(
            pattern.confidence,
            accuracy,
            usefulness
        )
        
        await self.pattern_store.update_pattern(pattern_id, {
            'success_rate': new_success_rate,
            'confidence': new_confidence
        })
        
        # Store feedback as memory for future learning
        await self.memory.store_fact(
            concept=f"pattern_feedback_{pattern_id}",
            category="feedback",
            properties={
                "pattern_id": pattern_id,
                "success": success,
                "accuracy": accuracy,
                "usefulness": usefulness,
                "timestamp": datetime.utcnow().isoformat()
            },
            relations={"feedback_for": [pattern_id]},
            confidence=0.8
        )
        
        return True
    
    def _update_success_rate(
        self,
        current_rate: float,
        applications: int,
        success: bool
    ) -> float:
        """Update success rate using exponential smoothing"""
        if applications == 0:
            return 1.0 if success else 0.0
        
        alpha = min(self.learning_rate, 1.0 / applications)
        new_value = 1.0 if success else 0.0
        return current_rate * (1 - alpha) + new_value * alpha
    
    def _update_confidence(
        self,
        current_confidence: float,
        accuracy: float,
        usefulness: float
    ) -> float:
        """Update confidence based on feedback"""
        feedback_score = (accuracy + usefulness) / 2
        alpha = self.learning_rate
        return current_confidence * (1 - alpha) + feedback_score * alpha
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        pattern_stats = await self.pattern_store.get_statistics()
        memory_stats = await self.memory.get_memory_stats()
        
        return {
            "patterns": pattern_stats,
            "memory": memory_stats,
            "recognizers": list(self.recognizers.keys()),
            "learning_rate": self.learning_rate,
            "confidence_threshold": self.confidence_threshold
        }
    
    async def shutdown(self):
        """Shutdown the pattern learning system"""
        await self.memory.shutdown()