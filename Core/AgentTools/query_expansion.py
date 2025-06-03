"""
Query Expansion Module for improving entity search relevance
"""
from typing import List, Set
import re
from Core.Common.Logger import logger

class QueryExpander:
    """Expands queries with synonyms and related terms to improve entity search"""
    
    def __init__(self):
        # Domain-specific expansions based on the Fictional_Test corpus
        self.term_expansions = {
            # Crystal-related terms
            "crystal plague": ["the crystal plague", "great crystal plague", "plague", "crystalline virus", "crystal disease", "crystal corruption"],
            "crystal technology": ["crystal tech", "crystalline technology", "crystal-based technology", "crystal based technology", "levitite technology"],
            "levitite": ["levitite crystals", "levitite stones", "anti-gravity crystals", "blue crystals"],
            
            # Empire and location terms
            "zorathian empire": ["zorathian", "zorathians", "empire", "zorathian civilization"],
            "aerophantis": ["aerophantis", "capital city", "floating city", "capital of zorathian empire"],
            "shadowpeak": ["shadowpeak mountains", "shadow peak", "levitite mines"],
            
            # People and roles
            "emperor zorthak": ["emperor", "zorthak", "emperor zorthak iii", "zorthak the luminous", "emperor zorthak the luminous"],
            "crystal keepers": ["crystal keeper", "keepers", "scientists and engineers"],
            "sky warriors": ["sky warrior", "warriors", "military", "zorathian military"],
            
            # Events and concepts
            "fall": ["downfall", "collapse", "destruction", "end", "decline"],
            "floating cities": ["floating city", "levitating cities", "aerial cities", "sky cities"],
            
            # Time periods
            "1850 bce": ["1850 bc", "eighteen fifty bce", "crystal plague era"],
            "1823 bce": ["1823 bc", "fall of aerophantis", "empire's end"]
        }
        
        # Common query patterns that need expansion
        self.query_patterns = {
            r"what (?:is|are) (.+?)(?:\?|$)": self._expand_definition_query,
            r"tell me about (.+?)(?:\?|$)": self._expand_information_query,
            r"who (?:was|is|were) (.+?)(?:\?|$)": self._expand_person_query,
            r"what (?:caused|led to) (.+?)(?:\?|$)": self._expand_causal_query,
            r"describe (.+?)(?:\?|$)": self._expand_description_query
        }
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple search terms
        
        Args:
            query: The original user query
            
        Returns:
            List of expanded search terms
        """
        expanded_terms = set()
        
        # Always include the original query
        expanded_terms.add(query.lower())
        
        # Apply pattern-based expansions
        for pattern, expansion_func in self.query_patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                pattern_expansions = expansion_func(match.group(1))
                expanded_terms.update(pattern_expansions)
                break
        
        # Apply term-based expansions
        query_lower = query.lower()
        for key_term, expansions in self.term_expansions.items():
            if key_term in query_lower:
                expanded_terms.update(expansions)
                # Also add individual words from the key term
                expanded_terms.update(key_term.split())
        
        # Add individual significant words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'when', 
                     'where', 'why', 'how', 'of', 'to', 'in', 'for', 'on', 'at', 'by'}
        words = query_lower.split()
        significant_words = [w for w in words if w not in stop_words and len(w) > 2]
        expanded_terms.update(significant_words)
        
        result = list(expanded_terms)
        logger.info(f"Query expansion: '{query}' -> {len(result)} terms: {result[:10]}...")
        return result
    
    def _expand_definition_query(self, subject: str) -> Set[str]:
        """Expand 'what is X' type queries"""
        expansions = {subject}
        # Add subject without articles
        cleaned = re.sub(r'^(the|a|an)\s+', '', subject)
        expansions.add(cleaned)
        # Add plural/singular variations
        if subject.endswith('s'):
            expansions.add(subject[:-1])
        else:
            expansions.add(subject + 's')
        return expansions
    
    def _expand_information_query(self, subject: str) -> Set[str]:
        """Expand 'tell me about X' type queries"""
        expansions = self._expand_definition_query(subject)
        # Add "X information", "X details"
        expansions.add(f"{subject} information")
        expansions.add(f"{subject} details")
        return expansions
    
    def _expand_person_query(self, person: str) -> Set[str]:
        """Expand 'who was X' type queries"""
        expansions = {person}
        # Handle titles
        if "emperor" in person:
            name_parts = person.replace("emperor", "").strip()
            expansions.add(name_parts)
            expansions.add(f"emperor {name_parts}")
        # Add role-based expansions
        expansions.add(f"{person} biography")
        expansions.add(f"{person} role")
        return expansions
    
    def _expand_causal_query(self, event: str) -> Set[str]:
        """Expand 'what caused X' type queries"""
        expansions = {event}
        # Add causal terms
        expansions.add(f"{event} cause")
        expansions.add(f"{event} reason")
        expansions.add(f"{event} led to")
        expansions.add(f"{event} resulted in")
        # Common causal events in the corpus
        if "fall" in event or "downfall" in event:
            expansions.update(["crystal plague", "1850 bce", "collapse", "crashed"])
        return expansions
    
    def _expand_description_query(self, subject: str) -> Set[str]:
        """Expand 'describe X' type queries"""
        return self._expand_information_query(subject)

# Global instance
query_expander = QueryExpander()